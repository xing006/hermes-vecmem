"""嵌入引擎 — 支持 API 和本地两种嵌入方式。"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class EmbedEngine:
    """Text embedding engine.

    Supports two modes (configurable via config):
      - api: calls an OpenAI-compatible embeddings endpoint
      - local: uses sentence-transformers with a local model

    Falls back to a simple hash-based vector if neither is available.
    """

    def __init__(self, config: dict):
        self._config = config
        self._mode = config.get("embed_mode", "api").lower()
        self._dimension = 384  # default, updated after first real embedding

        # API mode config
        self._api_base = config.get("api_base", "").rstrip("/")
        self._api_key = config.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
        self._model = config.get("model", "deepseek-embedding")

        # Local mode config
        self._local_model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self._local_model = None

        # In-memory embedding cache (SHA-256 → vector)
        # Mirrors the SQLite cache for hot-path speed
        self._mem_cache: dict[str, List[float]] = {}

        # Optional external cache (wired to VecStore when available)
        self._external_cache_get: Optional[Callable] = None
        self._external_cache_set: Optional[Callable] = None

    def set_external_cache(self, get_fn: Callable, set_fn: Callable) -> None:
        """Wire in a persistent cache store (e.g. VecStore)."""
        self._external_cache_get = get_fn
        self._external_cache_set = set_fn

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        """Convert text to embedding vector. Returns a list of floats."""
        if not text.strip():
            return [0.0] * self._dimension

        # Check in-memory cache first
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._mem_cache.get(cache_key)
        if cached is not None:
            return cached

        # Check external cache (VecStore)
        if self._external_cache_get is not None:
            cached = self._external_cache_get(text, self._model)
            if cached is not None:
                self._mem_cache[cache_key] = cached
                return cached

        # Compute embedding
        if self._mode == "api":
            emb = self._embed_api(text)
        elif self._mode == "local":
            emb = self._embed_local(text)
        else:
            emb = self._embed_fallback(text)

        # Cache result
        self._mem_cache[cache_key] = emb
        if self._external_cache_set is not None and self._mode == "api":
            self._external_cache_set(text, emb, self._model)

        return emb

    def probe_dimension(self) -> int:
        """Return the actual embedding dimension by making a probe call.

        Safe to call before VecStore is initialized — does not require a DB.
        """
        probe = self.embed("probe")
        return len(probe)

    # ------------------------------------------------------------------ #
    # API mode
    # ------------------------------------------------------------------ #

    def _embed_api(self, text: str) -> List[float]:
        import httpx

        url = f"{self._api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": text,
        }

        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]

            # Update dimension from real API response
            if len(embedding) != self._dimension:
                logger.info(
                    "Embedding dimension updated: %d → %d (model=%s)",
                    self._dimension, len(embedding), self._model,
                )
                self._dimension = len(embedding)
            return embedding
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Embedding API returned %s: %s",
                e.response.status_code, e.response.text[:200],
            )
            return self._embed_fallback(text)
        except Exception as e:
            logger.warning("Embedding API call failed: %s", e)
            return self._embed_fallback(text)

    # ------------------------------------------------------------------ #
    # Local mode (sentence-transformers)
    # ------------------------------------------------------------------ #

    def _embed_local(self, text: str) -> List[float]:
        if self._local_model is None:
            self._load_local_model()
        try:
            emb = self._local_model.encode(text)
            return emb.tolist()
        except Exception as e:
            logger.warning("Local embedding failed: %s", e)
            return self._embed_fallback(text)

    def _load_local_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", self._local_model_name)
            self._local_model = SentenceTransformer(self._local_model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            self._mode = "fallback"
        except Exception as e:
            logger.warning("Failed to load local model: %s", e)
            self._mode = "fallback"

    # ------------------------------------------------------------------ #
    # Fallback mode (hash-based deterministic vector)
    # ------------------------------------------------------------------ #

    def _embed_fallback(self, text: str) -> List[float]:
        """Create a simple hash-based deterministic vector when no real embedding is available.

        This is NOT semantically meaningful — it just produces consistent
        vectors for the same text so identical phrases match.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(self._dimension):
            idx = i % len(digest)
            val = (digest[idx] / 255.0) * 2.0 - 1.0  # normalize to [-1, 1]
            vec.append(val)
        return vec

    @property
    def dimension(self) -> int:
        return self._dimension
