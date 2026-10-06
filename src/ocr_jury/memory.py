"""Correction memory: a Qdrant vector DB (local mode, no server/Docker) of
human-corrected lines.

Embedding = character n-gram TF-IDF (2-4 chars) reduced to 256 dims with SVD,
fit on the training pool only. Chosen over a semantic sentence-embedding model
on purpose: those are trained on clean text and scatter OCR misspellings
("prifoner"), while character n-grams still overlap heavily between a misread
line and its correction - which is exactly the lookup this memory performs.

Only lines that have been "human-corrected" (pool ground truth, or lines
added during the review loop) are ever stored. Dev/test ground truth never is.
"""
from __future__ import annotations

import os
import pickle
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .data import ROOT
from .metrics import normalize

DIM = 256
COLLECTION = "corrections"


class CorrectionMemory:
    def __init__(self, path: str | None = None, fresh: bool = False):
        self.path = path or os.path.join(ROOT, "cache", "qdrant")
        self._encoder_path = os.path.join(self.path, "encoder.pkl")
        os.makedirs(self.path, exist_ok=True)
        self.client = QdrantClient(path=self.path)
        self.encoder = None
        if fresh and self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        if not fresh and os.path.exists(self._encoder_path):
            with open(self._encoder_path, "rb") as f:
                self.encoder = pickle.load(f)

    def fit_encoder(self, texts: list[str]) -> None:
        """Fit on pool text only; the encoder is frozen afterwards so vectors
        stay comparable as the memory grows."""
        self.encoder = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
            TruncatedSVD(n_components=DIM, random_state=42),
            Normalizer(),
        )
        self.encoder.fit([normalize(t) for t in texts])
        with open(self._encoder_path, "wb") as f:
            pickle.dump(self.encoder, f)
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(COLLECTION, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.transform([normalize(t) for t in texts]).astype(np.float32)

    def add(self, texts: list[str], source: str = "pool") -> None:
        if not texts:
            return
        vecs = self._embed(texts)
        points = [PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload={"text": normalize(t), "source": source})
                  for t, v in zip(texts, vecs)]
        for i in range(0, len(points), 512):
            self.client.upsert(COLLECTION, points=points[i:i + 512])

    def count(self) -> int:
        return self.client.count(COLLECTION).count

    def search(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        res = self.client.query_points(COLLECTION, query=self._embed([text])[0].tolist(), limit=k).points
        return [(p.payload["text"], float(p.score)) for p in res]

    def close(self):
        self.client.close()
