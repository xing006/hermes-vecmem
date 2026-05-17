"""嵌入引擎 — 支持 API、本地、TF-IDF 三种嵌入方式。"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Regex for tokenizing mixed Chinese + English text
_TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+|[^\s\w]")


class EmbedEngine:
    """Text embedding engine.

    Supports three modes:
      - api: calls an OpenAI-compatible embeddings endpoint
      - local: uses sentence-transformers with a local model
      - fallback: TF-IDF weighted feature hashing (always works, no deps)

    The fallback uses TF-IDF + feature hashing into a fixed-dimension vector.
    Unlike pure hash, TF-IDF captures term importance and co-occurrence:
    two texts sharing important terms will have similar vectors.
    """

    def __init__(self, config: dict):
        self._config = config
        self._mode = config.get("embed_mode", "api").lower()
        self._dimension = 384

        # API mode config
        self._api_base = config.get("api_base", "").rstrip("/")
        self._api_key = config.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
        self._model = config.get("model", "deepseek-embedding")

        # Local mode config
        self._local_model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self._local_model = None

        # Caches
        self._mem_cache: dict[str, List[float]] = {}
        self._external_cache_get: Optional[Callable] = None
        self._external_cache_set: Optional[Callable] = None

        # TF-IDF corpus — tracks document frequencies for IDF weighting
        self._df: dict[str, int] = {}    # term → doc frequency
        self._total_docs = 0

    def set_external_cache(self, get_fn: Callable, set_fn: Callable) -> None:
        self._external_cache_get = get_fn
        self._external_cache_set = set_fn

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self._dimension

        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._mem_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._external_cache_get is not None:
            cached = self._external_cache_get(text, self._model)
            if cached is not None:
                self._mem_cache[cache_key] = cached
                return cached

        if self._mode == "api":
            emb = self._embed_api(text)
        elif self._mode == "local":
            emb = self._embed_local(text)
        else:
            emb = self._embed_fallback(text)

        self._mem_cache[cache_key] = emb
        if self._external_cache_set is not None and self._mode == "api":
            self._external_cache_set(text, emb, self._model)

        return emb

    def probe_dimension(self) -> int:
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
        payload = {"model": self._model, "input": text}

        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            if len(embedding) != self._dimension:
                logger.info("Embedding dimension updated: %d → %d (model=%s)",
                            self._dimension, len(embedding), self._model)
                self._dimension = len(embedding)
            return embedding
        except Exception as e:
            logger.warning("Embedding API failed: %s", e)
            return self._embed_fallback(text)

    # ------------------------------------------------------------------ #
    # Local mode
    # ------------------------------------------------------------------ #

    def _embed_local(self, text: str) -> List[float]:
        if self._local_model is None:
            self._load_local_model()
        try:
            return self._local_model.encode(text).tolist()
        except Exception as e:
            logger.warning("Local embedding failed: %s", e)
            return self._embed_fallback(text)

    def _load_local_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local model: %s", self._local_model_name)
            self._local_model = SentenceTransformer(self._local_model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed")
            self._mode = "fallback"
        except Exception as e:
            logger.warning("Failed to load local model: %s", e)
            self._mode = "fallback"

    # ------------------------------------------------------------------ #
    # TF-IDF fallback mode
    # ------------------------------------------------------------------ #

    def _embed_fallback(self, text: str) -> List[float]:
        """TF-IDF weighted feature hashing into a fixed-dimension vector.

        How it works:
          1. Tokenize text into unigrams + bigrams
          2. Hash each n-gram to 2 positions in the vector (signed hash)
          3. Weight by TF * IDF
          4. L2-normalize

        Unlike pure hash vectors, this captures term importance:
        "前端框架 Vue.js" and "Vue.js 前端开发" share important terms
        (前端, vue, js) so their vectors will be similar.
        """
        # 1. Tokenize
        terms = self._tokenize(text)
        if not terms:
            return [0.0] * self._dimension

        # 2. Count term frequencies (TF)
        tf: dict[str, float] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        max_tf = max(tf.values()) if tf else 1

        # 3. Build vector via feature hashing with TF-IDF weighting
        vec = [0.0] * self._dimension
        n_docs = max(self._total_docs, 1)

        for term, freq in tf.items():
            # TF: augmented frequency (0.5 + 0.5 * tf / max_tf)
            tf_weight = 0.5 + 0.5 * (freq / max_tf)

            # IDF: log(N / df), smooth with +1
            df = self._df.get(term, 1)
            idf = math.log((n_docs + 1) / (df + 1)) + 1

            weight = tf_weight * idf

            # Feature hashing: map term to 2 positions
            h1 = int(hashlib.md5(term.encode()).hexdigest()[:8], 16)
            h2 = int(hashlib.sha256(term.encode()).hexdigest()[:8], 16)

            pos1 = h1 % self._dimension
            pos2 = h2 % self._dimension
            sign1 = 1 if (h1 & 1) == 0 else -1
            sign2 = 1 if (h2 & 1) == 0 else -1

            vec[pos1] += sign1 * weight
            vec[pos2] += sign2 * weight

        # 4. L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]

        # 5. Update corpus stats for future IDF computation
        self._total_docs += 1
        seen_terms = set(terms)
        for t in seen_terms:
            self._df[t] = self._df.get(t, 0) + 1

        return vec

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize mixed Chinese + English text into n-grams (unigrams + bigrams)."""
        # Extract all tokens
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return []

        # Generate unigrams and bigrams
        ngrams = []
        ngrams.extend(tokens)  # unigrams
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]} {tokens[i+1]}")  # bigrams

        return ngrams

    @property
    def dimension(self) -> int:
        return self._dimension
