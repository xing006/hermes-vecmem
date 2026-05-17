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
        "  stats — Memory stats"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "keyword", "add", "delete", "list", "stats"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search'/'keyword')."},
            "id": {"type": "integer", "description": "Fact ID (required for 'delete')."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
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
        # Accumulate user messages for batch extraction
        self._message_buffer: List[str] = []

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
            return "\n".join(lines) if len(lines) > 1 else ""
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

        Uses heuristic patterns rather than an LLM to keep it fast and free.
        """
        import re
        facts: List[str] = []
        seen = set()

        # Pattern 1: Windows/Unix paths
        path_pattern = re.compile(
            r'(?:路径|目录|在|地址|位置)[：:\s]*([a-zA-Z]:[/\\][^\s,，。；;]{5,})'
            r'|([a-zA-Z]:[/\\][^\s,，。；;]{5,})'
        )

        # Pattern 2: Preferences (用/偏好/喜欢 + noun)
        pref_pattern = re.compile(
            r'(?:用|使用|偏好|喜欢|默认)(?:的|是)?[：:\s]*([^，。,\.]{2,40})'
        )

        # Pattern 3: Config statements (设/配置/设置 + value)
        config_pattern = re.compile(
            r'(?:设|设置|配置|改为|改成|切换)[：:\s]*([^，。,\.]{3,60})'
        )

        # Pattern 4: Key/token patterns
        key_pattern = re.compile(
            r'(key|token|密钥|密码|ap[pi]_key|api_key|PAT|令牌)[：:\s]*["\'`]?([a-zA-Z0-9_\-]{8,})["\'`]?',
            re.IGNORECASE,
        )

        # Pattern 5: General facts (I use / I prefer / my X is)
        general_pattern = re.compile(
            r'(?:我(?:的|用|在|有|是))([^，。！？\n]{4,60})'
        )

        for msg in self._message_buffer:
            # Paths
            for m in path_pattern.finditer(msg):
                fact = m.group(1) or m.group(2)
                if fact and fact not in seen:
                    seen.add(fact)
                    facts.append(f"路径: {fact.strip()}")

            # Config
            for m in config_pattern.finditer(msg):
                fact = m.group(1).strip()
                if fact and fact not in seen:
                    seen.add(fact)
                    facts.append(fact)

            # Preferences
            for m in pref_pattern.finditer(msg):
                fact = m.group(1).strip()
                if fact and fact not in seen and len(fact) > 4:
                    seen.add(fact)
                    facts.append(f"偏好: {fact}")

            # Keys/tokens (store masked version only)
            for m in key_pattern.finditer(msg):
                key_type = m.group(1)
                if key_type and key_type not in seen:
                    seen.add(key_type)
                    facts.append(f"已配置 {key_type}")

            # General facts
            for m in general_pattern.finditer(msg):
                fact = m.group(1).strip()
                if fact and fact not in seen and len(fact) > 6:
                    # Deduplicate with path facts
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
