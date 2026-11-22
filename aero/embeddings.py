"""Local embedding model (no API quota). 384-dim vectors, normalised."""
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    vecs = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]
