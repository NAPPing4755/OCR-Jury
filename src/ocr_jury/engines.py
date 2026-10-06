"""OCR engine adapters behind one interface.

Each engine takes a single text-line image and returns a Reading. The jury
logic only ever sees Readings, so no part of it depends on a specific OCR
library. Engines are loaded lazily: EasyOCR and docTR pull model weights on
first use, and that cost shouldn't be paid by code that never runs them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Reading:
    engine: str
    text: str
    confidence: float | None  # engine's own 0-1 confidence, if it reports one


class Engine(Protocol):
    name: str

    def read(self, image: Image.Image) -> Reading: ...


class TesseractEngine:
    name = "tesseract"

    def __init__(self):
        import pytesseract
        from dotenv import load_dotenv

        load_dotenv()
        cmd = os.getenv("TESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self._pt = pytesseract

    def read(self, image: Image.Image) -> Reading:
        # psm 7 = "treat the image as a single text line", which matches
        # this dataset's pre-cropped line images.
        data = self._pt.image_to_data(image.convert("RGB"), lang="eng", config="--psm 7",
                                      output_type=self._pt.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data["text"], data["conf"]):
            if word.strip():
                words.append(word)
                conf = float(conf)
                if conf >= 0:
                    confs.append(conf / 100)
        return Reading(self.name, " ".join(words), float(np.mean(confs)) if confs else None)


class EasyOCREngine:
    name = "easyocr"

    def __init__(self):
        import easyocr

        # quantize=False: EasyOCR's default CPU dynamic quantization crashed on
        # ~98% of lines ("ChooseQuantizationParams, min should be less than or
        # equal to max") once per-process thread limits were set.
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)

    def read(self, image: Image.Image) -> Reading:
        results = self._reader.readtext(np.array(image.convert("RGB")), detail=1, paragraph=False)
        # EasyOCR may split one line into several boxes; order them left to right.
        results.sort(key=lambda r: min(p[0] for p in r[0]))
        text = " ".join(r[1] for r in results)
        confs = [float(r[2]) for r in results]
        return Reading(self.name, text, float(np.mean(confs)) if confs else None)


class DocTREngine:
    name = "doctr"

    def __init__(self):
        from doctr.models import ocr_predictor

        # Full detection + recognition. docTR's recognizers are trained on
        # single-word crops and never emit spaces, so feeding them a whole
        # line returns one run-on string ("bothcivilandecclefiaftical").
        # Detecting word boxes first and joining them restores word breaks.
        # Light detector: on 8 dev lines db_mobilenet_v3_large ran ~2x faster
        # than db_resnet50 (3.8 vs 7.7 s/line on CPU) at a small accuracy cost
        # (mean line CER 0.078 vs 0.070). For a jury, engine diversity matters
        # more than one engine's peak accuracy.
        self._model = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="crnn_vgg16_bn", pretrained=True)

    def read(self, image: Image.Image) -> Reading:
        doc = self._model([np.array(image.convert("RGB"))])
        words = [w for block in doc.pages[0].blocks for line in block.lines for w in line.words]
        # One physical line, so reading order is simply left to right.
        words.sort(key=lambda w: w.geometry[0][0])
        text = " ".join(w.value for w in words)
        conf = float(np.mean([w.confidence for w in words])) if words else None
        return Reading(self.name, text, conf)


ENGINES = {"tesseract": TesseractEngine, "easyocr": EasyOCREngine, "doctr": DocTREngine}


def load_engines(names: list[str]) -> list[Engine]:
    return [ENGINES[n]() for n in names]
