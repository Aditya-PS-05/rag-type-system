"""Local cross-encoder reranker (ONNX, runs offline once the model is downloaded to models/)."""

import math
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

DEFAULT_MODEL = "jinaai/jina-reranker-v1-turbo-en"
CACHE_DIR = Path(__file__).resolve().parent.parent / "models"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path = CACHE_DIR):
        try:
            self.model = TextCrossEncoder(model_name, cache_dir=str(cache_dir), local_files_only=True)
        except Exception:
            self.model = TextCrossEncoder(model_name, cache_dir=str(cache_dir))

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance of each document to the query, in [0, 1]."""
        if not documents:
            return []
        return [1 / (1 + math.exp(-s)) for s in self.model.rerank(query, documents, batch_size=16)]
