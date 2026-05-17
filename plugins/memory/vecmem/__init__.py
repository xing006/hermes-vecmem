"""
sqlite-vec memory provider — SQLite向量记忆插件

使用 sqlite-vec 扩展在 SQLite 中存储和检索向量化的记忆。
支持语义搜索（向量相似度）和关键词搜索（FTS5）的混合检索。

Config in config.yaml:
  memory:
    provider: vecmem
    vecmem:
      embed_mode: api           # api | local
      top_k: 5
      min_score: 0.3
      # API 模式:
      api_base: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model: deepseek-embedding
      # 本地模式:
      # model_name: all-MiniLM-L6-v2
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


VECMEM_SCHEMA = {
    "name": "vecmem",
    "description": (
        "Vector memory — semantically search stored facts. "
        "Use when you need to recall information from past conversations "
        "by meaning, not just keywords.\n\n"
        "Actions:\n"
        "  search <query> — Semantic search (finds similar meanings)\n"
        "  keyword <query> — Exact keyword match via FTS5\n"
        "  add <content> — Store a fact (auto-embedded)\n"
        "  delete <id> — Remove a fact\n"
        "  list [limit] — Recent facts\n"
        "  build_index — Train IVF index for faster search (data > 100 items)\n"
        "  set_probe <n> — Set IVF probe count (higher = more accurate but slower)\n"
        "  stats — Memory stats"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "keyword", "add", "delete", "list", "build_index", "set_probe", "stats"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search'/'keyword')."},
            "id": {"type": "integer", "description": "Fact ID (required for 'delete')."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
            "n": {"type": "integer", "description": "IVF probe count (for 'set_probe')."},
        },
        "required": ["action"],
    },
}


class VecMemProvider(MemoryProvider):
    """Vector memory provider backed by sqlite-vec."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._store = None
        self._embed = None
        self._top_k = int(self._config.get("top_k", 5))
        self._min_score = float(self._config.get("min_score", 0.3))
        self._sync_interval = int(self._config.get("sync_interval", 3))
        self._turn_count = 0
        self._message_buffer: List[str] = []
        # LLM extraction config
        self._llm_extract = self._config.get("llm_extract", True)
        self._llm_api_base = self._config.get("llm_api_base",
                            self._config.get("api_base", "https://api.deepseek.com")).rstrip("/")
        self._llm_api_key = self._config.get("llm_api_key",
                            self._config.get("api_key", ""))
        self._llm_model = self._config.get("llm_model", "deepseek-chat")
        if not self._llm_api_key:
            import os
            self._llm_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        # LRU cache for prefetch results
        self._lru_max = int(self._config.get("lru_size", 100))
        self._lru_ttl = int(self._config.get("lru_ttl", 60))
        self._lru_cache: dict = {}  # query_hash → (timestamp, result_string)
        self._lru_order: list[str] = []  # ordered keys for LRU eviction

    @property
    def name(self) -> str:
        return "vecmem"

    # ------------------------------------------------------------------ #
    # Core lifecycle
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        try:
            import sqlite_vec  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        from .store import VecStore
        from .embed import EmbedEngine

        logger.info("Initializing vecmem provider for session %s", session_id)
        hermes_home = kwargs.get("hermes_home", None)

        # 1. Create EmbedEngine first, probe real dimension
        self._embed = EmbedEngine(config=self._config)
        probe_dim = self._embed.probe_dimension()
        logger.info("Probed embedding dimension: %d", probe_dim)

        # 2. Create VecStore with probed dimension (or existing table's dim)
        self._store = VecStore(hermes_home=hermes_home, dimension=probe_dim)
        self._store.initialize()

        # 3. Wire cache: EmbedEngine → VecStore
        self._embed.set_external_cache(
            get_fn=self._store.get_cached_embedding,
            set_fn=self._store.set_cached_embedding,
        )

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
        if self._embed:
            self._embed = None
        logger.info("vecmem provider shut down")

    # ------------------------------------------------------------------ #
    # System prompt
    # ------------------------------------------------------------------ #

    def system_prompt_block(self) -> str:
        return (
            "You have vector memory (vecmem). It stores facts as embeddings and "
            "retrieves them by semantic similarity. Use the `vecmem` tool to search "
            "or store facts manually. Memories are also auto-prefetched each turn."
        )

    # ------------------------------------------------------------------ #
    # Prefetch & sync
    # ------------------------------------------------------------------ #

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not self._embed:
            return ""
        if not query.strip():
            return ""

        # LRU cache check
        import hashlib, time
        qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        now = time.time()

        cached = self._lru_cache.get(qhash)
        if cached is not None:
            ts, result = cached
            if now - ts < self._lru_ttl:
                # Move to end (most recently used)
                if qhash in self._lru_order:
                    self._lru_order.remove(qhash)
                self._lru_order.append(qhash)
                logger.debug("LRU prefetch cache HIT: %.20s", query)
                return result
            else:
                # Expired
                del self._lru_cache[qhash]
                self._lru_order.remove(qhash)

        try:
            embedding = self._embed.embed(query)
            results = self._store.search(embedding, limit=self._top_k)
            if not results:
                return ""
            lines = ["[From vecmem vector memory:]"]
            for r in results:
                score = r.get("score", 0)
                if score < self._min_score:
                    continue
                lines.append(f"  • {r['content']} (score: {score:.2f})")
            result = "\n".join(lines) if len(lines) > 1 else ""

            # Store in LRU cache
            self._lru_cache[qhash] = (now, result)
            self._lru_order.append(qhash)
            # Evict oldest if over limit
            while len(self._lru_order) > self._lru_max:
                oldest = self._lru_order.pop(0)
                self._lru_cache.pop(oldest, None)

            return result
        except Exception as e:
            logger.warning("vecmem prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Auto-extract facts from user messages on a configurable interval.

        Uses lightweight heuristics to identify factual statements
        (paths, preferences, config, keys) and stores them as vector memories.
        Non-blocking — runs synchronously but is very fast.
        """
        if not self._store or not self._embed:
            return
        if not user_content.strip():
            return

        self._turn_count += 1
        self._message_buffer.append(user_content)

        # Check if it's time to extract
        if self._turn_count % self._sync_interval != 0:
            return

        # Extract facts from buffered messages
        facts = self._extract_facts_from_buffer()
        self._message_buffer.clear()

        for fact in facts:
            try:
                embedding = self._embed.embed(fact)
                self._store.add(fact, embedding, category="auto")
            except Exception as e:
                logger.debug("sync_turn auto-store failed: %s", e)

    def _extract_facts_from_buffer(self) -> List[str]:
        """Extract concise factual statements from buffered messages.

        Uses LLM extraction when available (llm_extract=True + API key),
        falls back to regex heuristics otherwise.
        """
        if not self._message_buffer:
            return []

        # Try LLM extraction first
        if self._llm_extract and self._llm_api_key:
            try:
                facts = self._extract_with_llm()
                if facts:
                    return facts
            except Exception as e:
                logger.debug("LLM extraction failed, falling back to regex: %s", e)

        # Fallback: regex heuristics
        return self._extract_with_regex()

    def _extract_with_llm(self) -> List[str]:
        """Call LLM to extract concise facts from buffered messages.

        Returns a list of fact strings, or empty list if nothing to extract.
        """
        import json
        import httpx

        conversation = "\n".join(
            f"用户: {msg}" for msg in self._message_buffer
        )

        prompt = (
            "你是一个记忆提取助手。从以下用户对话中，提取所有可以长期记忆的事实性信息。\n\n"
            "包括但不限于：\n"
            "- 项目路径、文件位置\n"
            "- 技术栈偏好（语言、框架、工具）\n"
            "- 配置信息（服务器、端口、账号）\n"
            "- API Key / Token（只记类型，不记具体值）\n"
            "- 工作习惯、个人偏好\n"
            "- 重要的人名、项目名\n\n"
            "规则：\n"
            "1. 每个事实用简洁的一句话表达\n"
            "2. 只提取明确陈述的事实，不猜测\n"
            "3. API Key/Token 只记\"已配置 xxx\"，不记原文\n"
            "4. 如果没有值得记忆的内容，返回空数组\n"
            "5. 返回 JSON 格式：[\"事实1\", \"事实2\"]\n\n"
            f"对话内容：\n{conversation}"
        )

        resp = httpx.post(
            f"{self._llm_api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Parse JSON from LLM response
        # Try direct JSON parse first
        try:
            facts = json.loads(content)
            if isinstance(facts, list):
                return [f.strip() for f in facts if f.strip()]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from markdown code block
        import re
        m = re.search(r'\[.*?\]', content, re.DOTALL)
        if m:
            try:
                facts = json.loads(m.group(0))
                if isinstance(facts, list):
                    return [f.strip() for f in facts if f.strip()]
            except json.JSONDecodeError:
                pass

        # Last resort: split by newlines, filter
        lines = [l.strip().lstrip("- ").lstrip("* ") for l in content.split("\n")
                 if l.strip() and not l.strip().startswith(("```", "json"))]
        return lines[:10]

    def _extract_with_regex(self) -> List[str]:
        """Fallback: extract facts using regex heuristics."""
        import re
        facts: List[str] = []
        seen = set()

        path_pattern = re.compile(
            r'(?:路径|目录|在|地址|位置)[：:\s]*([a-zA-Z]:[/\\][^\s,，。；;]{5,})'
            r'|([a-zA-Z]:[/\\][^\s,，。；;]{5,})'
        )
        pref_pattern = re.compile(
            r'(?:用|使用|偏好|喜欢|默认)(?:的|是)?[：:\s]*([^，。,\.]{2,40})'
        )
        config_pattern = re.compile(
            r'(?:设|设置|配置|改为|改成|切换)[：:\s]*([^，。,\.]{3,60})'
        )
        key_pattern = re.compile(
            r'(key|token|密钥|密码|ap[pi]_key|api_key|PAT|令牌)[：:\s]*["\'`]?([a-zA-Z0-9_\-]{8,})["\'`]?',
            re.IGNORECASE,
        )
        general_pattern = re.compile(
            r'(?:我(?:的|用|在|有|是))([^，。！？\n]{4,60})'
        )

        for msg in self._message_buffer:
            for m in path_pattern.finditer(msg):
                fact = m.group(1) or m.group(2)
                if fact and fact not in seen:
                    seen.add(fact)
                    facts.append(f"路径: {fact.strip()}")

            for m in config_pattern.finditer(msg):
                fact = m.group(1).strip()
                if fact and fact not in seen:
                    seen.add(fact)
                    facts.append(fact)

            for m in pref_pattern.finditer(msg):
                fact = m.group(1).strip()
                if fact and fact not in seen and len(fact) > 4:
                    seen.add(fact)
                    facts.append(f"偏好: {fact}")

            for m in key_pattern.finditer(msg):
                key_type = m.group(1)
                if key_type and key_type not in seen:
                    seen.add(key_type)
                    facts.append(f"已配置 {key_type}")

            for m in general_pattern.finditer(msg):
                fact = m.group(1).strip()
                if fact and fact not in seen and len(fact) > 6:
                    if not any(f.startswith("路径:") and f.endswith(fact[:20]) for f in facts):
                        seen.add(fact)
                        facts.append(fact)

        return facts

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes into vecmem storage."""
        if action != "add" or not self._store or not self._embed or not content:
            return
        try:
            category = "user_pref" if target == "user" else "memory"
            embedding = self._embed.embed(content)
            self._store.add(content, embedding, category=category)
            logger.debug("Mirrored memory write to vecmem: %s", content[:50])
        except Exception as e:
            logger.debug("on_memory_write mirror failed: %s", e)

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [VECMEM_SCHEMA]

    def handle_tool_call(self, args: Dict[str, Any]) -> str:
        import json

        action = args.get("action", "")
        try:
            if action == "add":
                return self._handle_add(args)
            elif action == "search":
                return self._handle_search(args)
            elif action == "keyword":
                return self._handle_keyword(args)
            elif action == "delete":
                return self._handle_delete(args)
            elif action == "list":
                return self._handle_list(args)
            elif action == "stats":
                return self._handle_stats()
            elif action == "build_index":
                return self._handle_build_index()
            elif action == "set_probe":
                return self._handle_set_probe(args)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            logger.exception("vecmem tool call failed")
            return json.dumps({"error": str(e)})

    def _handle_add(self, args: dict) -> str:
        import json
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content required"})
        embedding = self._embed.embed(content)
        fid = self._store.add(content, embedding)
        return json.dumps({"id": fid, "content": content, "status": "stored"})

    def _handle_search(self, args: dict) -> str:
        import json
        query = args.get("query", "").strip()
        limit = int(args.get("limit", self._top_k))
        if not query:
            return json.dumps({"error": "query required"})
        embedding = self._embed.embed(query)
        results = self._store.search(embedding, limit=limit)
        results = [r for r in results if r.get("score", 0) >= self._min_score]
        return json.dumps({"results": results, "count": len(results)})

    def _handle_keyword(self, args: dict) -> str:
        import json
        query = args.get("query", "").strip()
        limit = int(args.get("limit", self._top_k))
        if not query:
            return json.dumps({"error": "query required"})
        results = self._store.keyword_search(query, limit=limit)
        return json.dumps({"results": results, "count": len(results)})

    def _handle_delete(self, args: dict) -> str:
        import json
        fid = args.get("id")
        if fid is None:
            return json.dumps({"error": "id required"})
        self._store.delete(int(fid))
        return json.dumps({"status": "deleted", "id": fid})

    def _handle_list(self, args: dict) -> str:
        import json
        limit = int(args.get("limit", 20))
        results = self._store.list_all(limit=limit)
        return json.dumps({"results": results, "count": len(results)})

    def _handle_stats(self) -> str:
        import json
        stats = self._store.stats()
        return json.dumps(stats)

    def _handle_build_index(self) -> str:
        import json
        result = self._store.build_index(force=True)
        return json.dumps(result)

    def _handle_set_probe(self, args: dict) -> str:
        import json
        n = int(args.get("n", 2))
        self._store.set_ivf_probe(n)
        return json.dumps({"status": "ok", "probe": n})


# ------------------------------------------------------------------ #
# Plugin entry point
# ------------------------------------------------------------------ #

def register(ctx) -> None:
    """Register vecmem memory provider with the plugin system."""
    from hermes_cli.config import cfg_get
    from hermes_constants import get_hermes_home
    import yaml

    config_path = get_hermes_home() / "config.yaml"
    plugin_config = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            plugin_config = cfg_get(all_config, "memory", "vecmem", default={}) or {}
        except Exception:
            pass

    provider = VecMemProvider(config=plugin_config)
    ctx.register_memory_provider(provider)
