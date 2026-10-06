"""TrOCR: the 4th, fine-tunable recognizer.

Base model: microsoft/trocr-small-printed (~62M params; fits CPU training).
Fine-tuning targets are the NORMALIZED ground truth (long-s -> s), so the
model learns to read the ſ glyph directly as "s" - the form the primary
metric scores - instead of the "f" every off-the-shelf engine outputs.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .metrics import normalize

BASE = "microsoft/trocr-small-printed"
MAX_LEN = 64


def load(checkpoint: str | None = None):
    src = checkpoint or BASE
    processor = TrOCRProcessor.from_pretrained(src)
    model = VisionEncoderDecoderModel.from_pretrained(src)
    tok = processor.tokenizer
    # Keep the checkpoint's own special-token config and only fill gaps.
    # Overriding decoder_start_token_id with the tokenizer's CLS id (0), as
    # some fine-tuning recipes do, broke this checkpoint completely: it was
    # trained to start decoding from id 2, and zero-shot output collapsed to
    # "." / ".TY R.J.J.J..." with a first training loss of 13.4.
    # For this checkpoint the top-level config leaves these blank; the real
    # values (start=2, eos=2, pad=1) live in the decoder's own config.
    dec = model.config.decoder
    for k in ("decoder_start_token_id", "pad_token_id", "eos_token_id"):
        if getattr(model.config, k) is None:
            setattr(model.config, k, getattr(dec, k, None))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    for k in ("decoder_start_token_id", "pad_token_id", "eos_token_id"):
        setattr(model.generation_config, k, getattr(model.config, k))
    return model, processor


def save(model, processor, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)


@torch.no_grad()
def read_batch(model, processor, images, batch_size: int = 8) -> list[tuple[str, float]]:
    """Returns (text, confidence) per image; confidence = mean token probability."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        pix = processor(images=images[i:i + batch_size], return_tensors="pt").pixel_values
        gen = model.generate(pix, max_new_tokens=MAX_LEN, num_beams=1,
                             output_scores=True, return_dict_in_generate=True)
        texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)
        trans = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
        for t, s in zip(texts, trans):
            probs = torch.exp(s[torch.isfinite(s)])
            out.append((t.strip(), float(probs.mean()) if len(probs) else 0.0))
    return out


def train(model, processor, samples: list[tuple], epochs: int = 1, lr: float = 5e-5,
          batch_size: int = 4, seed: int = 42, log_every: int = 20, freeze_encoder: bool = True) -> list[float]:
    """samples: [(image or zero-arg callable returning an image, ground-truth text)].

    freeze_encoder: train only the text decoder. On this 16 GB machine,
    full fine-tuning (batch 8) pushed Windows into ~37k pages/s of swapping;
    a frozen vision encoder needs no gradients or optimizer state for about
    a third of the weights and no stored encoder activations. The decoder is
    where the mappings that matter here are learned (receipt-style CAPITALS
    -> normal case, long-s -> s). Callables let images load per batch
    instead of holding every reviewed line image in memory.
    """
    rng = random.Random(seed)
    tok = processor.tokenizer
    for p in model.encoder.parameters():
        p.requires_grad = not freeze_encoder
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    model.train()
    losses = []
    for epoch in range(epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for step, i in enumerate(range(0, len(order), batch_size), 1):
            batch = [samples[j] for j in order[i:i + batch_size]]
            images = [b[0]() if callable(b[0]) else b[0] for b in batch]
            if freeze_encoder:
                with torch.no_grad():
                    enc = model.encoder(pixel_values=processor(images=images, return_tensors="pt").pixel_values)
            pix = None if freeze_encoder else processor(images=images, return_tensors="pt").pixel_values
            # Pad to the longest label in the batch, not a fixed 64 tokens -
            # most lines are far shorter, so this cuts decoder compute too.
            labels = tok([normalize(b[1]) for b in batch], padding="longest", max_length=MAX_LEN,
                         truncation=True, return_tensors="pt").input_ids
            labels[labels == tok.pad_token_id] = -100  # ignore padding in the loss
            if freeze_encoder:
                loss = model(encoder_outputs=enc, labels=labels).loss
            else:
                loss = model(pixel_values=pix, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(float(loss.detach()))
            if step % log_every == 0:
                print(f"    epoch {epoch + 1} step {step}: loss {np.mean(losses[-log_every:]):.3f}", flush=True)
    model.eval()
    return losses
