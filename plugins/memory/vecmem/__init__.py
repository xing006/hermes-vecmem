"""
sqlite-vec memory provider — SQLite向量记忆插件

使用 sqlite-vec 扩展在 SQLite 中存储和检索向量化的记忆。
支持语义搜索（向量相似度）和关键词搜索（FTS5）的混合检索。

Config in config.yaml:
  memory:
    provider: vecmem
    vecmem:
      embed_mode: api           # api | local
      top_k: 3
      min_score: 0.55
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


VECMEM_LIGHT_ACTIONS = ["search", "keyword", "add", "list", "archive", "restore", "stats"]

VECMEM_ADVANCED_ACTIONS = [
    "search", "keyword", "add", "list", "archive", "restore", "stats",
    "preview_add", "review_list", "approve", "reject",
    "cleanup_dry_run", "cleanup_report",
    "preview_archive", "preview_restore", "preview_approve", "preview_reject",
    "create_plan", "get_plan", "list_plans", "apply_plan", "events",
    "build_index", "set_probe",
    "explain",
    "review_bulk_plan",
    "health_report",
]

VECMEM_LIGHT_DESCRIPTION = (
    "Vector memory — semantically search stored facts. "
    "Use when you need to recall information from past conversations "
    "by meaning, not just keywords. Default personal light mode exposes only everyday actions.\n\n"
    "Actions:\n"
    "  search <query> — Semantic search (finds similar meanings)\n"
    "  keyword <query> — Exact keyword match via FTS5\n"
    "  add <content> — Store or upsert a fact (auto-embedded, deduplicated)\n"
    "  archive/restore <id> — Hide or restore a fact without deleting it\n"
    "  list [limit] — Recent facts\n"
    "  stats — Memory stats\n\n"
    "Advanced governance actions are hidden unless memory.vecmem.governance_mode is 'advanced'."
)

VECMEM_ADVANCED_DESCRIPTION = (
    "Vector memory — semantically search stored facts. "
    "Use when you need to recall information from past conversations "
    "by meaning, not just keywords.\n\n"
    "Actions:\n"
    "  search <query> — Semantic search (finds similar meanings)\n"
    "  keyword <query> — Exact keyword match via FTS5\n"
    "  add <content> — Store or upsert a fact (auto-embedded, deduplicated)\n"
    "  preview_add <content> — Dry-run add/upsert/LLM decision without writing\n"
    "  archive/restore <id> — Hide or restore a fact without deleting it\n"
    "  preview_archive/preview_restore <id> — Dry-run status changes\n"
    "  review_list — List records waiting for manual review\n"
    "  approve/reject <id> — Approve or reject a review record\n"
    "  cleanup_dry_run — Generate low-risk cleanup report + dry-run archive plan\n"
    "  cleanup_report — Alias of cleanup_dry_run\n"
    "  preview_approve/preview_reject <id> — Dry-run review decisions\n"
    "  create_plan <operations> — Save a dry-run governance plan JSON without applying it\n"
    "  get_plan/list_plans/apply_plan — Inspect or execute saved plans; apply requires confirm_text\n"
    "  events [id] — Show audit events for memory changes\n"
    "  list [limit] — Recent facts\n"
    "  build_index — Train IVF index for faster search (data > 100 items)\n"
    "  set_probe <n> — Set IVF probe count (higher = more accurate but slower)\n"
    "  explain <query> — Show why each memory was/wasn't retrieved (scores, filters, paths)\n"
    "  review_bulk_plan — Auto-generate a batch approve/reject plan for review records\n"
    "  health_report — Generate a cron-friendly memory health report\n"
    "  stats — Memory stats"
)

VECMEM_SCHEMA = {
    "name": "vecmem",
    "description": VECMEM_ADVANCED_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": VECMEM_ADVANCED_ACTIONS,
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search'/'keyword')."},
            "id": {"type": "integer", "description": "Fact ID (required for 'delete')."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
            "n": {"type": "integer", "description": "IVF probe count (for 'set_probe')."},
            "operations": {"type": "array", "description": "Plan operations for create_plan."},
            "plan_id": {"type": "string", "description": "Plan ID for get_plan/apply_plan."},
            "confirm_text": {"type": "string", "description": "Required exact text for apply_plan: 我确认执行计划."},
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
        self._top_k = int(self._config.get("top_k", 3))
        self._min_score = float(self._config.get("min_score", 0.55))
        self._sync_interval = int(self._config.get("sync_interval", 3))
        self._semantic_dedupe = bool(self._config.get("semantic_dedupe", False))
        self._semantic_duplicate_score = float(self._config.get("semantic_duplicate_score", 0.90))
        self._llm_merge = bool(self._config.get("llm_merge", False))
        self._llm_merge_min_score = float(self._config.get("llm_merge_min_score", 0.75))
        self._auto_filter_questions = bool(self._config.get("auto_filter_questions", True))
        self._governance_mode = str(self._config.get("governance_mode", "light")).strip().lower()
        if self._governance_mode not in {"light", "advanced"}:
            logger.warning("Unknown vecmem governance_mode=%r; falling back to light", self._governance_mode)
            self._governance_mode = "light"
        self._turn_count = 0
        self._message_buffer: List[str] = []
        self._hermes_home: Optional[str] = None
        # LLM extraction config
        self._llm_extract = self._config.get("llm_extract", True)
        self._llm_api_base = self._config.get("llm_api_base",
                            self._config.get("api_base", "https://api.deepseek.com")).rstrip("/")
        self._llm_api_key = self._config.get("llm_api_key",
                            self._config.get("api_key", ""))
        self._llm_model = self._config.get("llm_model", "deepseek-chat")
        self._llm_extract_min_confidence = float(self._config.get("llm_extract_min_confidence", 0.75))
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
        self._hermes_home = hermes_home

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

        # Extract fact candidates from buffered messages
        candidates = self._extract_facts_from_buffer()
        self._message_buffer.clear()

        for candidate in candidates:
            try:
                fact = self._candidate_content(candidate)
                if not fact:
                    continue
                if not self._candidate_is_long_term(candidate):
                    continue
                if not self._should_store_fact(fact):
                    continue
                metadata = candidate if isinstance(candidate, dict) else None
                self._upsert_memory(fact, category="auto", source="sync_turn", metadata=metadata)
            except Exception as e:
                logger.debug("sync_turn auto-store failed: %s", e)

    def _candidate_content(self, candidate: Any) -> str:
        if isinstance(candidate, dict):
            return self._normalize_fact_text(str(candidate.get("content") or ""))
        return self._normalize_fact_text(str(candidate or ""))

    def _candidate_is_long_term(self, candidate: Any) -> bool:
        """Accept legacy string facts and only high-confidence long-term structured candidates."""
        if not isinstance(candidate, dict):
            return True
        durability = str(candidate.get("durability") or "").strip().lower()
        if durability != "long_term":
            return False
        try:
            confidence = float(candidate.get("confidence", 0))
        except (TypeError, ValueError):
            return False
        return confidence >= self._llm_extract_min_confidence

    def _coerce_extraction_candidates(self, raw: Any) -> List[Any]:
        """Normalize LLM extraction output while preserving legacy string tests."""
        if not isinstance(raw, list):
            return []
        candidates: List[Any] = []
        for item in raw:
            if isinstance(item, str):
                text = self._normalize_fact_text(item)
                if text:
                    candidates.append(text)
                continue
            if not isinstance(item, dict):
                continue
            content = self._normalize_fact_text(str(item.get("content") or ""))
            if not content:
                continue
            candidate = dict(item)
            candidate["content"] = content
            candidates.append(candidate)
        return candidates

    def _extract_facts_from_buffer(self) -> List[Any]:
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

    def _extract_with_llm(self) -> List[Any]:
        """Call LLM to extract structured fact candidates from buffered messages.

        Returns candidates shaped as:
        {content, memory_type, durability, confidence, reason}.
        Legacy string arrays are still accepted for compatibility.
        """
        import json
        import httpx

        conversation = "\n".join(
            f"用户: {msg}" for msg in self._message_buffer
        )

        prompt = (
            "你是 Hermes Agent 的长期记忆候选提取器。请从用户消息中提取结构化候选，"
            "目标是减少污染：只让稳定、明确、未来仍有用的事实进入长期记忆。\n\n"
            "允许提取的 long_term 类型：\n"
            "- preference：用户稳定偏好、工作习惯、表达风格\n"
            "- environment：稳定环境信息、安装路径、长期配置\n"
            "- project：长期项目索引、仓库/文档位置、项目约定\n"
            "- workflow：可复用流程或稳定操作习惯\n"
            "- credential_hint：只记录某类凭据已配置，绝不记录具体密钥值\n"
            "- tool_quirk：稳定工具坑点或环境差异\n\n"
            "必须拒绝或标 temporary/reject 的内容：\n"
            "- 疑问句、反问、请求帮助、要你执行的命令\n"
            "- 临时任务进度、今天/本轮/刚才/下一步/已完成等流水日志\n"
            "- 半成型想法、草稿、未确认计划、一次性决策\n"
            "- 裸 URL、裸文件路径、无上下文片段、过短偏好片段\n"
            "- API Key/Token/密码原文；如确有必要，只写‘已配置 xxx 凭据’\n\n"
            "输出要求：\n"
            "1. 只返回 JSON 数组，不要 markdown，不要解释。\n"
            "2. 每个对象必须包含：content, memory_type, subject, predicate, object, durability, confidence, reason。\n"
            "3. memory_type 只能从允许类型中选择；subject 用稳定英文/点分命名，predicate 用 is/prefers/uses/path/configured 等短谓词，object 写事实值。\n"
            "4. durability 只能是 long_term、temporary、reject。\n"
            "5. confidence 为 0 到 1。只有明确陈述且未来一周后仍有用的事实才给 long_term。\n"
            "6. 没有候选时返回 []。\n\n"
            "示例输出：\n"
            "[{\"content\":\"用户偏好：回答简洁务实\",\"memory_type\":\"preference\","
            "\"subject\":\"user.response_style\",\"predicate\":\"prefers\",\"object\":\"回答简洁务实\","
            "\"durability\":\"long_term\",\"confidence\":0.95,\"reason\":\"稳定沟通偏好\"}]\n\n"
            f"用户消息：\n{conversation}"
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
            return self._coerce_extraction_candidates(facts)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from markdown code block
        import re
        m = re.search(r'\[.*?\]', content, re.DOTALL)
        if m:
            try:
                facts = json.loads(m.group(0))
                return self._coerce_extraction_candidates(facts)
            except json.JSONDecodeError:
                pass

        # Last resort: split by newlines, filter as legacy string candidates.
        lines = [l.strip().lstrip("- ").lstrip("* ") for l in content.split("\n")
                 if l.strip() and not l.strip().startswith(("```", "json"))]
        return self._coerce_extraction_candidates(lines[:10])

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

    def _normalize_fact_text(self, text: str) -> str:
        import re
        return re.sub(r"\s+", " ", (text or "").strip())

    def _should_store_fact(self, fact: str) -> bool:
        """Filter low-value auto-extracted facts before they pollute vector memory."""
        import re

        text = self._normalize_fact_text(fact)
        if not text or len(text) < 10:
            return False
        if len(text) > 1000:
            return False

        lowered = text.casefold()
        explicit_prefixes = (
            "active_context:", "current_project:", "idea_backlog:",
            "paused_project:", "abandoned_project:", "mvp_backlog:",
        )

        # Bare paths/URLs are usually extraction artifacts. Keep them only when the
        # caller intentionally writes a structured project index with an explicit prefix.
        if not text.startswith(explicit_prefixes):
            if re.fullmatch(r"路径[:：]\s*\S+", text):
                return False
            if re.fullmatch(r"(?:https?://|s://|www\.)\S+", lowered):
                return False
            if re.fullmatch(r"[a-zA-Z]:[/\\][^\s]+", text):
                return False
            if lowered.startswith(("路径: s://", "路径：https://", "路径: http://", "路径: www.")):
                return False

        request_markers = (
            "请", "帮我", "帮忙", "麻烦", "读取", "检查", "执行", "创建", "修改", "生成", "发布",
            "please", "can you", "could you",
        )
        if any(m in lowered for m in request_markers) and not text.startswith(explicit_prefixes):
            return False

        secret_patterns = (
            r"(?:密码|口令|password|passwd|pwd)[:：=\s]+\S+",
            r"(?:token|api[_-]?key|secret|密钥|令牌)[:：=\s]+[a-zA-Z0-9_\-]{8,}",
        )
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in secret_patterns):
            return False

        if self._auto_filter_questions:
            question_markers = ("?", "？")
            question_words = ("吗", "是不是", "有没有", "如何", "怎么", "为啥", "为什么", "能不能", "可不可以", "要不要", "哪个", "是否")
            if text.endswith(question_markers) or any(w in text for w in question_words):
                return False

        # Reject malformed preference fragments produced by regex fallback.
        bad_prefixes = (
            "偏好: 是", "偏好: 有没有", "偏好: 如何", "偏好: 怎么",
            "偏好: 还是", "偏好: 这个", "偏好: 来", "偏好: 户",
            "用户问", "用户询问",
        )
        if text.startswith(bad_prefixes):
            return False
        if text.startswith("偏好:"):
            pref = text.split(":", 1)[1].strip()
            if len(pref) < 8:
                return False
            if re.search(r"[？?]", pref):
                return False
            if pref.startswith(("是", "了", "这个", "那个", "还是", "来", "户")):
                return False

        # Avoid storing obvious task progress; use project STATUS/INDEX or active_context explicitly instead.
        transient_markers = (
            "刚才", "这次", "本轮", "当前", "今天", "明天", "下一步",
            "已经完成", "已完成", "正在做", "进行中", "待办", "todo",
            "真实订单", "订单", "order_id", "item_id", "proc_", "发布结果",
            "smoke test", "验证通过", "跑通", "点击成功", "后台进程",
        )
        if any(m in lowered for m in transient_markers) and not text.startswith(explicit_prefixes):
            return False

        # Reject conversational half-sentences and ideation fragments.
        fragment_prefixes = ("想法是", "想如果", "想既然", "目的是", "场景可能", "刚发现", "新会话里", "一下就能用")
        if text.startswith(fragment_prefixes):
            return False

        # Secret/value breadcrumbs without durable context are noisy and risky.
        if text in ("已配置 KEY", "已配置 key", "已配置 token", "已配置 API_KEY"):
            return False

        return True

    def _unique_prefix_for(self, content: str) -> Optional[str]:
        text = content.strip()
        singleton_prefixes = ("active_context:", "current_project:")
        for prefix in singleton_prefixes:
            if text.startswith(prefix):
                return prefix
        named_prefixes = ("idea_backlog:", "paused_project:", "abandoned_project:", "mvp_backlog:")
        for prefix in named_prefixes:
            if text.startswith(prefix):
                # Keep one record per prefix + name before first comma/semicolon/Chinese punctuation.
                rest = text[len(prefix):].strip()
                name = rest.split("，", 1)[0].split(",", 1)[0].split("；", 1)[0].split(";", 1)[0].strip()
                return f"{prefix} {name}" if name else prefix
        return None

    def _semantic_duplicate(self, content: str, embedding: List[float], category: str) -> Optional[Dict[str, Any]]:
        if not self._semantic_dedupe:
            return None
        try:
            results = self._store.search(embedding, limit=3)
        except Exception:
            return None
        normalized = self._normalize_fact_text(content).casefold()
        for r in results:
            if r.get("category") != category:
                continue
            score = float(r.get("score", 0))
            existing = self._normalize_fact_text(r.get("content", "")).casefold()
            if score >= self._semantic_duplicate_score or existing == normalized:
                return r
        return None

    def _llm_candidates(self, embedding: List[float], category: str) -> List[Dict[str, Any]]:
        if not self._llm_merge:
            return []
        try:
            results = self._store.search(embedding, limit=5)
        except Exception:
            return []
        candidates = []
        for r in results:
            if r.get("category") != category:
                continue
            if float(r.get("score", 0)) >= self._llm_merge_min_score:
                candidates.append(r)
        return candidates

    def _parse_llm_decision_json(self, content: str) -> Dict[str, Any]:
        import json, re
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if not m:
                return {"action": "store", "reason": "LLM returned non-JSON decision"}
            data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {"action": "store", "reason": "LLM decision was not an object"}

    def _llm_memory_decision(self, new_fact: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates or not self._llm_api_key:
            return {"action": "store", "reason": "no candidates or no llm api key"}
        import httpx
        compact = [
            {"id": c.get("id"), "content": c.get("content"), "score": c.get("score"), "category": c.get("category")}
            for c in candidates
        ]
        prompt = (
            "你是长期记忆合并助手。判断新记忆与候选旧记忆的关系，只返回 JSON。\n"
            "允许 action: duplicate, update, merge, review, store。\n"
            "duplicate=语义完全重复，保留旧记录；update=新事实是旧事实的更完整版本，更新 target_id；"
            "merge=把多个事实合成一条新内容，旧记录标记 superseded；review=冲突/不确定，存为待人工确认；store=无关，直接新增。\n"
            "JSON 字段：action, target_id, target_ids, content, confidence, reason。\n"
            f"新记忆：{new_fact}\n候选：{compact}"
        )
        resp = httpx.post(
            f"{self._llm_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self._llm_api_key}", "Content-Type": "application/json"},
            json={
                "model": self._llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_llm_decision_json(data["choices"][0]["message"]["content"].strip())

    def _apply_llm_decision(self, text: str, embedding: List[float], category: str,
                            source: str, candidates: List[Dict[str, Any]],
                            decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(decision.get("action") or "store").strip().lower()
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        reason = str(decision.get("reason") or "").strip() or None
        candidate_ids = {int(c["id"]) for c in candidates if c.get("id") is not None}

        if action == "duplicate":
            target_id = int(decision.get("target_id") or (next(iter(candidate_ids)) if candidate_ids else 0))
            if target_id in candidate_ids:
                self._store.touch(target_id)
                existing = self._store.get(target_id)
                return {"status": "llm_duplicate", "id": target_id, "content": existing.get("content") if existing else text, "reason": reason}

        if action == "update":
            target_id = int(decision.get("target_id") or 0)
            if target_id in candidate_ids:
                merged_content = self._normalize_fact_text(decision.get("content") or text)
                merged_embedding = self._embed.embed(merged_content)
                self._store.update(target_id, merged_content, merged_embedding, category=category, source=source,
                                   status="active", confidence=confidence or 1.0, decision_reason=reason)
                return {"status": "llm_updated", "id": target_id, "content": merged_content, "reason": reason}

        if action == "merge":
            ids = [int(x) for x in decision.get("target_ids") or [] if int(x) in candidate_ids]
            if ids:
                merged_content = self._normalize_fact_text(decision.get("content") or text)
                merged_embedding = self._embed.embed(merged_content)
                fid = self._store.add(merged_content, merged_embedding, category=category, source=source,
                                      status="active", confidence=confidence or 1.0, decision_reason=reason)
                for old_id in ids:
                    self._store.mark_status(old_id, "superseded", reason)
                return {"status": "llm_merged", "id": fid, "content": merged_content, "merged_from": ids, "reason": reason}

        if action == "review":
            fid = self._store.add(text, embedding, category=category, source=source, status="review",
                                  confidence=confidence, decision_reason=reason)
            return {"status": "review", "id": fid, "content": text, "reason": reason}

        return None

    def _preview_llm_decision(self, text: str, category: str, candidates: List[Dict[str, Any]],
                              decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action = str(decision.get("action") or "store").strip().lower()
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        reason = str(decision.get("reason") or "").strip() or None
        candidate_ids = {int(c["id"]) for c in candidates if c.get("id") is not None}

        if action == "duplicate":
            target_id = int(decision.get("target_id") or (next(iter(candidate_ids)) if candidate_ids else 0))
            if target_id in candidate_ids:
                existing = self._store.get(target_id)
                return {"dry_run": True, "would_action": "llm_duplicate", "target_id": target_id,
                        "preview_content": existing.get("content") if existing else text,
                        "reason": reason, "confidence": confidence, "candidates": candidates}

        if action == "update":
            target_id = int(decision.get("target_id") or 0)
            if target_id in candidate_ids:
                merged_content = self._normalize_fact_text(decision.get("content") or text)
                return {"dry_run": True, "would_action": "llm_updated", "target_id": target_id,
                        "preview_content": merged_content, "reason": reason,
                        "confidence": confidence, "candidates": candidates}

        if action == "merge":
            ids = [int(x) for x in decision.get("target_ids") or [] if int(x) in candidate_ids]
            if ids:
                merged_content = self._normalize_fact_text(decision.get("content") or text)
                return {"dry_run": True, "would_action": "llm_merged", "target_ids": ids,
                        "preview_content": merged_content, "reason": reason,
                        "confidence": confidence, "candidates": candidates}

        if action == "review":
            return {"dry_run": True, "would_action": "review", "preview_content": text,
                    "reason": reason, "confidence": confidence, "candidates": candidates}

        return None

    def _preview_add(self, content: str, category: str = "general") -> Dict[str, Any]:
        text = self._normalize_fact_text(content)
        embedding = self._embed.embed(text)
        semantic_dup = self._semantic_duplicate(text, embedding, category)
        if semantic_dup:
            return {"dry_run": True, "would_action": "semantic_duplicate", "target_id": semantic_dup["id"],
                    "preview_content": semantic_dup["content"]}

        existing = self._store.find_by_hash(text, category=category)
        if existing:
            return {"dry_run": True, "would_action": "duplicate", "target_id": existing["id"],
                    "preview_content": existing["content"]}

        unique_prefix = self._unique_prefix_for(text)
        if unique_prefix:
            matches = self._store.find_by_prefix(unique_prefix, category=category, limit=50)
            if matches:
                return {"dry_run": True, "would_action": "updated", "target_id": matches[0]["id"],
                        "preview_content": text, "removed": max(0, len(matches) - 1),
                        "unique_prefix": unique_prefix}

        candidates = self._llm_candidates(embedding, category)
        if candidates:
            try:
                decision = self._llm_memory_decision(text, candidates)
                preview = self._preview_llm_decision(text, category, candidates, decision)
                if preview:
                    return preview
            except Exception as e:
                logger.debug("LLM memory merge preview failed: %s", e)

        return {"dry_run": True, "would_action": "stored", "preview_content": text}

    def _same_topic_resolution(self, text: str, embedding: List[float], category: str, source: str,
                               topic_key: Optional[str], memory_type: Optional[str],
                               subject: Optional[str], predicate: Optional[str],
                               object_value: Optional[str]) -> Optional[Dict[str, Any]]:
        if not topic_key:
            return None
        derived = self._store.derive_metadata(text, category=category)
        memory_type = memory_type or derived.get("memory_type")
        subject = subject or derived.get("subject")
        predicate = predicate or derived.get("predicate")
        object_value = object_value or derived.get("object")
        matches = self._store.find_by_topic_key(topic_key, category=category, limit=10)
        if not matches:
            return None
        normalized_new = self._normalize_fact_text(text).casefold()
        object_norm = self._normalize_fact_text(str(object_value or "")).casefold()
        for match in matches:
            existing_text = self._normalize_fact_text(match.get("content", "")).casefold()
            existing_object = self._normalize_fact_text(str(match.get("object") or "")).casefold()
            if existing_text == normalized_new or (object_norm and object_norm == existing_object):
                self._store.touch(int(match["id"]), action="topic_duplicate", source=source, reason=f"same topic_key: {topic_key}")
                return {"status": "topic_duplicate", "id": match["id"], "content": match["content"], "topic_key": topic_key}

        reason = f"same topic_key conflict: {topic_key}"
        fid = self._store.add(
            text, embedding, category=category, source=source, status="review",
            topic_key=topic_key, memory_type=memory_type, subject=subject,
            predicate=predicate, object=object_value, confidence=0.5, decision_reason=reason,
        )
        return {"status": "review", "id": fid, "content": text, "topic_key": topic_key, "reason": reason}

    def _upsert_memory(self, content: str, category: str = "general", source: str = "manual",
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        text = self._normalize_fact_text(content)
        embedding = self._embed.embed(text)
        metadata = metadata or {}
        memory_type = metadata.get("memory_type")
        subject = metadata.get("subject")
        predicate = metadata.get("predicate")
        object_value = metadata.get("object") if "object" in metadata else metadata.get("value")
        topic_key = metadata.get("topic_key")
        if not topic_key:
            derived = self._store.derive_metadata(text, category=category)
            topic_key = self._store.derive_topic_key(
                text, category=category,
                memory_type=memory_type or derived.get("memory_type"),
                subject=subject or derived.get("subject"),
                predicate=predicate or derived.get("predicate"),
                object=object_value or derived.get("object"),
            )
        semantic_dup = self._semantic_duplicate(text, embedding, category)
        if semantic_dup:
            self._store.touch(int(semantic_dup["id"]), action="semantic_duplicate", source=source)
            return {"status": "semantic_duplicate", "id": semantic_dup["id"], "content": semantic_dup["content"]}

        existing = self._store.find_by_hash(text, category=category)
        if existing:
            self._store.touch(existing["id"], action="duplicate", source=source)
            return {"status": "duplicate", "id": existing["id"], "content": existing["content"]}

        unique_prefix = self._unique_prefix_for(text)
        if unique_prefix:
            return self._store.upsert(
                text, embedding, category=category, source=source, unique_prefix=unique_prefix,
                topic_key=topic_key, memory_type=memory_type, subject=subject, predicate=predicate, object=object_value,
            )

        topic_resolution = self._same_topic_resolution(
            text, embedding, category, source, topic_key, memory_type, subject, predicate, object_value
        )
        if topic_resolution:
            return topic_resolution

        candidates = self._llm_candidates(embedding, category)
        if candidates:
            try:
                decision = self._llm_memory_decision(text, candidates)
                applied = self._apply_llm_decision(text, embedding, category, source, candidates, decision)
                if applied:
                    return applied
            except Exception as e:
                logger.debug("LLM memory merge decision failed: %s", e)

        return self._store.upsert(
            text,
            embedding,
            category=category,
            source=source,
            unique_prefix=None,
            topic_key=topic_key,
            memory_type=memory_type,
            subject=subject,
            predicate=predicate,
            object=object_value,
        )


    def _plan_dir(self):
        from pathlib import Path
        base = Path(self._hermes_home) if self._hermes_home else Path.cwd()
        path = base / "vecmem_plans"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _plan_path(self, plan_id: str):
        safe = "".join(ch for ch in str(plan_id) if ch.isalnum() or ch in ("-", "_"))
        if not safe:
            raise ValueError("invalid plan_id")
        return self._plan_dir() / f"{safe}.json"

    def _preview_operation(self, op: Dict[str, Any]) -> Dict[str, Any]:
        action = str(op.get("action") or "").strip().lower()
        if action in ("archive", "restore", "approve", "reject"):
            after_status = {
                "archive": "archived",
                "restore": "active",
                "approve": "active",
                "reject": "archived",
            }[action]
            fid = op.get("id")
            if fid is None:
                return {"error": "id required", "action": action}
            existing = self._store.get(int(fid))
            if not existing:
                return {"error": f"memory id not found: {fid}", "action": action, "id": int(fid)}
            return {
                "dry_run": True,
                "would_action": action,
                "id": int(fid),
                "before_status": existing.get("status") or "active",
                "after_status": after_status,
                "content": existing.get("content"),
                "reason": op.get("reason"),
            }
        if action == "add":
            content = str(op.get("content") or "").strip()
            if not content:
                return {"error": "content required", "action": action}
            preview = self._preview_add(content, category=op.get("category", "general"))
            preview["action"] = action
            return preview
        return {"error": f"unsupported plan operation: {action}", "action": action}

    def _load_plan(self, plan_id: str) -> Dict[str, Any]:
        import json
        path = self._plan_path(plan_id)
        if not path.exists():
            raise FileNotFoundError(f"plan not found: {plan_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_plan(self, plan: Dict[str, Any]) -> str:
        import json
        path = self._plan_path(plan["plan_id"])
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _create_plan(self, operations: List[Dict[str, Any]], reason: Optional[str] = None) -> Dict[str, Any]:
        import time, uuid
        previews = [self._preview_operation(op) for op in operations]
        plan_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        plan = {
            "plan_id": plan_id,
            "status": "planned",
            "dry_run": True,
            "reason": reason,
            "created_at": time.time(),
            "applied_at": None,
            "applied": 0,
            "count": len(previews),
            "operations": previews,
        }
        plan["path"] = self._save_plan(plan)
        return plan

    def _apply_plan(self, plan_id: str) -> Dict[str, Any]:
        import time
        plan = self._load_plan(plan_id)
        if plan.get("status") == "applied":
            return {"error": f"plan already applied: {plan_id}"}
        applied = 0
        results = []
        for op in plan.get("operations") or []:
            if op.get("error"):
                results.append({"skipped": True, "error": op.get("error"), "operation": op})
                continue
            action = op.get("would_action")
            fid = op.get("id")
            reason = op.get("reason") or plan.get("reason")
            if action in ("archive", "restore", "approve", "reject"):
                after_status = op.get("after_status")
                self._store.mark_status(int(fid), after_status, reason, action=action, source="plan")
                applied += 1
                results.append({"status": "applied", "action": action, "id": int(fid)})
            else:
                results.append({"skipped": True, "error": f"unsupported apply action: {action}", "operation": op})
        plan["status"] = "applied"
        plan["dry_run"] = False
        plan["applied_at"] = time.time()
        plan["applied"] = applied
        plan["results"] = results
        plan["path"] = self._save_plan(plan)
        return {"status": "applied", "plan_id": plan_id, "applied": applied, "results": results, "path": plan["path"]}

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
            result = self._upsert_memory(content, category=category, source="memory_write")
            logger.debug("Mirrored memory write to vecmem: %s (%s)", content[:50], result.get("status"))
        except Exception as e:
            logger.debug("on_memory_write mirror failed: %s", e)

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        import copy

        schema = copy.deepcopy(VECMEM_SCHEMA)
        if self._governance_mode == "advanced":
            schema["description"] = VECMEM_ADVANCED_DESCRIPTION
            schema["parameters"]["properties"]["action"]["enum"] = list(VECMEM_ADVANCED_ACTIONS)
        else:
            schema["description"] = VECMEM_LIGHT_DESCRIPTION
            schema["parameters"]["properties"]["action"]["enum"] = list(VECMEM_LIGHT_ACTIONS)
        return [schema]

    def _is_action_allowed(self, action: str) -> bool:
        allowed = VECMEM_ADVANCED_ACTIONS if self._governance_mode == "advanced" else VECMEM_LIGHT_ACTIONS
        return action in allowed

    def handle_tool_call(self, tool_name_or_args: Any, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> str:
        import json

        # Hermes MemoryManager calls provider.handle_tool_call(tool_name, args, **kwargs).
        # Older project-local tests called provider.handle_tool_call(args). Support both
        # forms so the provider works in real Agent sessions and remains backward compatible.
        if args is None and isinstance(tool_name_or_args, dict):
            args = tool_name_or_args
        elif args is None:
            return json.dumps({"error": "args required"})
        elif not isinstance(args, dict):
            return json.dumps({"error": "args must be a dict"})

        action = args.get("action", "")
        if not self._is_action_allowed(action):
            return json.dumps({"error": f"Action '{action}' requires governance_mode=advanced"}, ensure_ascii=False)
        try:
            if action == "add":
                return self._handle_add(args)
            elif action == "preview_add":
                return self._handle_preview_add(args)
            elif action == "search":
                return self._handle_search(args)
            elif action == "keyword":
                return self._handle_keyword(args)
            elif action == "delete":
                return self._handle_delete(args)
            elif action == "archive":
                return self._handle_archive(args)
            elif action == "restore":
                return self._handle_restore(args)
            elif action == "preview_archive":
                return self._handle_preview_status(args, "archive", "archived")
            elif action == "preview_restore":
                return self._handle_preview_status(args, "restore", "active")
            elif action == "review_list":
                return self._handle_review_list(args)
            elif action in ("cleanup_dry_run", "cleanup_report"):
                return self._handle_cleanup_dry_run(args)
            elif action == "review_bulk_plan":
                return self._handle_review_bulk_plan(args)
            elif action == "health_report":
                return self._handle_health_report(args)
            elif action == "approve":
                return self._handle_approve(args)
            elif action == "reject":
                return self._handle_reject(args)
            elif action == "preview_approve":
                return self._handle_preview_status(args, "approve", "active")
            elif action == "preview_reject":
                return self._handle_preview_status(args, "reject", "archived")
            elif action == "create_plan":
                return self._handle_create_plan(args)
            elif action == "get_plan":
                return self._handle_get_plan(args)
            elif action == "list_plans":
                return self._handle_list_plans(args)
            elif action == "apply_plan":
                return self._handle_apply_plan(args)
            elif action == "events":
                return self._handle_events(args)
            elif action == "explain":
                return self._handle_explain(args)
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
        result = self._upsert_memory(content, category=args.get("category", "general"), source="tool")
        return json.dumps(result, ensure_ascii=False)

    def _handle_preview_add(self, args: dict) -> str:
        import json
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content required"})
        result = self._preview_add(content, category=args.get("category", "general"))
        return json.dumps(result, ensure_ascii=False)

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

    def _handle_archive(self, args: dict) -> str:
        import json
        fid = args.get("id")
        if fid is None:
            return json.dumps({"error": "id required"})
        self._store.mark_status(int(fid), "archived", args.get("reason"), action="archive", source="tool")
        return json.dumps({"status": "archived", "id": fid}, ensure_ascii=False)

    def _handle_preview_status(self, args: dict, action: str, after_status: str) -> str:
        import json
        fid = args.get("id")
        if fid is None:
            return json.dumps({"error": "id required"})
        existing = self._store.get(int(fid))
        if not existing:
            return json.dumps({"error": f"memory id not found: {fid}"}, ensure_ascii=False)
        return json.dumps({
            "dry_run": True,
            "would_action": action,
            "id": int(fid),
            "before_status": existing.get("status") or "active",
            "after_status": after_status,
            "content": existing.get("content"),
            "reason": args.get("reason"),
        }, ensure_ascii=False)

    def _handle_restore(self, args: dict) -> str:
        import json
        fid = args.get("id")
        if fid is None:
            return json.dumps({"error": "id required"})
        self._store.mark_status(int(fid), "active", args.get("reason"), action="restore", source="tool")
        return json.dumps({"status": "restored", "id": fid}, ensure_ascii=False)

    def _handle_review_list(self, args: dict) -> str:
        import json
        limit = int(args.get("limit", 20))
        category = args.get("category")
        memory_type = args.get("memory_type")
        topic_key = args.get("topic_key")
        results = self._store.list_by_status(
            "review", limit=limit,
            category=category, memory_type=memory_type, topic_key=topic_key,
        )
        # Enrich each result with related events and conflict info
        for r in results:
            fid = int(r["id"])
            events = self._store.list_events(memory_id=fid, limit=10)
            r["events"] = events
            # Find conflicting records: same topic_key different content
            tk = r.get("topic_key")
            if tk:
                conflicts = self._store.find_by_topic_key(tk, limit=5)
                conflicts = [c for c in conflicts if int(c["id"]) != fid]
                if conflicts:
                    r["conflicts"] = [{
                        "id": int(c["id"]),
                        "content": c["content"],
                        "status": c.get("status", "active"),
                        "confidence": c.get("confidence"),
                    } for c in conflicts]
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)

    def _handle_review_bulk_plan(self, args: dict) -> str:
        """Auto-generate a batch archive/reject plan for all review records.
        
        Uses confidence threshold: records with confidence >= auto_approve_min
        get approved, others get rejected (archived).
        """
        import json
        auto_approve_min = float(args.get("auto_approve_min", 0.7))
        category = args.get("category")
        memory_type = args.get("memory_type")
        topic_key = args.get("topic_key")
        limit = int(args.get("limit", 100))
        records = self._store.list_by_status(
            "review", limit=limit,
            category=category, memory_type=memory_type, topic_key=topic_key,
        )
        operations = []
        for r in records:
            confidence = r.get("confidence") or 0.5
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                conf = 0.5
            if conf >= auto_approve_min:
                operations.append({
                    "action": "approve",
                    "id": int(r["id"]),
                    "reason": f"review_bulk_plan: auto-approved (confidence={conf:.2f} >= {auto_approve_min})",
                })
            else:
                operations.append({
                    "action": "reject",
                    "id": int(r["id"]),
                    "reason": f"review_bulk_plan: auto-rejected (confidence={conf:.2f} < {auto_approve_min})",
                })
        if not operations:
            return json.dumps({"dry_run": True, "plan": None, "operations": [], "count": 0}, ensure_ascii=False)
        plan = self._create_plan(operations, reason=args.get("reason", "review bulk plan"))
        return json.dumps({"dry_run": True, "plan": plan, "operations": operations, "count": len(operations)}, ensure_ascii=False)

    def _classify_cleanup_candidate(self, row: Dict[str, Any], seen: Dict[tuple, int]) -> Optional[Dict[str, Any]]:
        content = str(row.get("content") or "")
        lowered = content.casefold()
        category = str(row.get("category") or "general")
        normalized = self._normalize_fact_text(content).casefold()
        status = row.get("status") or "active"
        if status != "active":
            return None

        def operation(reason: str, risk: str = "low") -> Dict[str, Any]:
            return {
                "action": "archive",
                "id": int(row["id"]),
                "risk": risk,
                "category": category,
                "content": content,
                "reason": reason,
            }

        key = (normalized, category)
        if key in seen:
            return operation("cleanup: exact_duplicate_same_category")
        seen[key] = int(row["id"])

        cross_category_key = (normalized, "*")
        if cross_category_key in seen and category in {"auto", "general"}:
            return operation("cleanup: exact_duplicate_less_stable_category")
        seen[cross_category_key] = int(row["id"])

        if category == "auto":
            return operation("cleanup: auto_noise")
        if lowered.startswith("active_context:"):
            return operation("cleanup: stale_active_context")
        if content.strip().endswith(("?", "？")) or any(marker in content for marker in ("是不是", "有没有", "如何", "怎么", "能不能", "要不要")):
            return operation("cleanup: question_or_request_noise")
        if any(marker in lowered for marker in ("dummy", "待办：", "待办:", "明天继续", "下一步", "order_id", "item_id", "proc_", "smoke test")):
            return operation("cleanup: temporary_progress_noise")
        return None

    def _cleanup_consistency(self) -> Dict[str, Any]:
        db = getattr(self._store, "_db", None)
        if db is None:
            return {"error": "store not initialized"}
        checks = {
            "mem_count": db.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
            "vec_count": db.execute("SELECT COUNT(*) FROM vec_memories").fetchone()[0],
            "fts_count": db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0],
            "orphan_vec": db.execute("SELECT COUNT(*) FROM vec_memories WHERE rowid NOT IN (SELECT id FROM memories)").fetchone()[0],
            "missing_vec": db.execute("SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT rowid FROM vec_memories)").fetchone()[0],
        }
        checks["ok"] = checks["orphan_vec"] == 0 and checks["missing_vec"] == 0
        return checks

    def _write_cleanup_report(self, report: Dict[str, Any]) -> str:
        import time
        from pathlib import Path
        reports_dir = Path("E:/projects/hermes-vecmem/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = reports_dir / f"cleanup-dryrun-{stamp}.md"
        lines = [
            "# vecmem cleanup dry-run report",
            "",
            f"- dry_run: `{report.get('dry_run')}`",
            f"- plan_id: `{report.get('plan', {}).get('plan_id')}`",
            f"- plan_path: `{report.get('plan', {}).get('path')}`",
            f"- low_risk_count: `{report.get('summary', {}).get('low_risk_count')}`",
            f"- review_count: `{report.get('summary', {}).get('review_count')}`",
            "",
            "## Low-risk archive operations",
        ]
        for op in report.get("operations", []):
            lines.append(f"- #{op.get('id')} `{op.get('reason')}` {op.get('content')}")
        lines.extend(["", "## Review candidates"])
        for item in report.get("review", []):
            lines.append(f"- #{item.get('id')} `{item.get('status')}` {item.get('content')}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def _cleanup_dry_run(self, reason: Optional[str] = None) -> Dict[str, Any]:
        rows = self._store.list_all(limit=10000, include_inactive=True)
        seen: Dict[tuple, int] = {}
        operations: List[Dict[str, Any]] = []
        review: List[Dict[str, Any]] = []
        for row in rows:
            op = self._classify_cleanup_candidate(row, seen)
            if op:
                operations.append(op)
                continue
            if (row.get("status") or "active") == "review":
                review.append({
                    "id": int(row["id"]),
                    "status": row.get("status"),
                    "topic_key": row.get("topic_key"),
                    "content": row.get("content"),
                    "reason": row.get("decision_reason"),
                })
        plan = self._create_plan(operations, reason=reason or "cleanup dry-run low-risk archive plan") if operations else {
            "dry_run": True,
            "plan_id": None,
            "path": None,
            "count": 0,
            "operations": [],
        }
        report = {
            "dry_run": True,
            "reason": reason,
            "summary": {
                "scanned": len(rows),
                "low_risk_count": len(operations),
                "review_count": len(review),
            },
            "operations": operations,
            "review": review,
            "plan": plan,
            "consistency": self._cleanup_consistency(),
        }
        report["report_path"] = self._write_cleanup_report(report)
        return report

    def _handle_cleanup_dry_run(self, args: dict) -> str:
        import json
        result = self._cleanup_dry_run(reason=args.get("reason"))
        return json.dumps(result, ensure_ascii=False)

    def _recent_events_since(self, since_ts: float) -> Dict[str, int]:
        db = getattr(self._store, "_db", None)
        if db is None:
            return {}
        cur = db.execute(
            "SELECT action, COUNT(*) as cnt FROM memory_events WHERE created_at >= ? GROUP BY action",
            (since_ts,),
        )
        return {row["action"]: row["cnt"] for row in cur.fetchall()}

    def _topic_conflicts(self, limit: int = 20) -> List[Dict[str, Any]]:
        db = getattr(self._store, "_db", None)
        if db is None:
            return []
        cur = db.execute(
            """
            SELECT topic_key, COUNT(*) as cnt
            FROM memories
            WHERE COALESCE(status, 'active') = 'active'
              AND topic_key IS NOT NULL
              AND topic_key != ''
            GROUP BY topic_key
            HAVING cnt > 1
            ORDER BY cnt DESC, topic_key ASC
            LIMIT ?
            """,
            (limit,),
        )
        conflicts = []
        for row in cur.fetchall():
            records = self._store.find_by_topic_key(row["topic_key"], include_inactive=False, limit=10)
            conflicts.append({
                "topic_key": row["topic_key"],
                "count": row["cnt"],
                "records": [
                    {"id": r.get("id"), "content": r.get("content"), "confidence": r.get("confidence")}
                    for r in records
                ],
            })
        return conflicts

    def _write_health_report(self, report: Dict[str, Any]) -> str:
        import time
        from pathlib import Path
        reports_dir = Path("E:/projects/hermes-vecmem/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = reports_dir / f"health-report-{stamp}.md"
        summary = report.get("summary", {})
        stats = report.get("stats", {})
        lines = [
            "# vecmem health report",
            "",
            f"- generated_at: `{report.get('generated_at')}`",
            f"- status: `{report.get('status')}`",
            f"- total: `{stats.get('total')}`",
            f"- active: `{stats.get('active')}`",
            f"- new_records: `{summary.get('new_records')}`",
            f"- review_records: `{summary.get('review_records')}`",
            f"- low_risk_cleanup_candidates: `{summary.get('low_risk_cleanup_candidates')}`",
            f"- topic_conflicts: `{summary.get('topic_conflicts')}`",
            f"- consistency_ok: `{summary.get('consistency_ok')}`",
            "",
            "## Recent events",
        ]
        for action, count in sorted(report.get("recent_events", {}).items()):
            lines.append(f"- {action}: `{count}`")
        lines.extend(["", "## Alerts"])
        alerts = report.get("alerts", [])
        if alerts:
            for alert in alerts:
                lines.append(f"- `{alert.get('level')}` {alert.get('code')}: {alert.get('message')}")
        else:
            lines.append("- none")
        lines.extend(["", "## Review records"])
        for item in report.get("review_records", []):
            lines.append(f"- #{item.get('id')} `{item.get('topic_key')}` {item.get('content')}")
        lines.extend(["", "## Cleanup candidates"])
        for op in report.get("cleanup", {}).get("operations", []):
            lines.append(f"- #{op.get('id')} `{op.get('reason')}` {op.get('content')}")
        lines.extend(["", "## Topic conflicts"])
        for conflict in report.get("topic_conflicts", []):
            lines.append(f"- `{conflict.get('topic_key')}` count={conflict.get('count')}")
            for r in conflict.get("records", []):
                lines.append(f"  - #{r.get('id')} {r.get('content')}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def _health_report(self, window_hours: int = 24, include_details: bool = True) -> Dict[str, Any]:
        import time
        stats = self._store.stats()
        since_ts = time.time() - max(int(window_hours), 1) * 3600
        db = getattr(self._store, "_db", None)
        new_records = 0
        if db is not None:
            new_records = db.execute(
                "SELECT COUNT(*) FROM memories WHERE created_at >= ?",
                (since_ts,),
            ).fetchone()[0]
        cleanup = self._cleanup_dry_run(reason=f"health report window={window_hours}h")
        review_records = self._store.list_by_status("review", limit=20)
        topic_conflicts = self._topic_conflicts(limit=20)
        consistency = stats.get("consistency", {})
        alerts: List[Dict[str, Any]] = []
        if not consistency.get("ok", False):
            alerts.append({
                "level": "critical",
                "code": "consistency_failed",
                "message": "Vector/FTS consistency check failed.",
            })
        if review_records:
            alerts.append({
                "level": "warning",
                "code": "review_queue_nonempty",
                "message": f"{len(review_records)} records are waiting for review.",
            })
        if cleanup.get("summary", {}).get("low_risk_count", 0):
            alerts.append({
                "level": "warning",
                "code": "cleanup_candidates",
                "message": f"{cleanup['summary']['low_risk_count']} low-risk cleanup candidates found.",
            })
        if topic_conflicts:
            alerts.append({
                "level": "warning",
                "code": "topic_conflicts",
                "message": f"{len(topic_conflicts)} active topic keys have multiple records.",
            })
        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "window_hours": window_hours,
            "status": "attention" if alerts else "ok",
            "summary": {
                "new_records": new_records,
                "review_records": len(review_records),
                "low_risk_cleanup_candidates": cleanup.get("summary", {}).get("low_risk_count", 0),
                "topic_conflicts": len(topic_conflicts),
                "consistency_ok": consistency.get("ok", False),
                "db_size_bytes": stats.get("db_size_bytes", 0),
            },
            "stats": stats,
            "recent_events": self._recent_events_since(since_ts),
            "alerts": alerts,
            "cleanup": cleanup if include_details else {"summary": cleanup.get("summary"), "report_path": cleanup.get("report_path")},
            "review_records": review_records if include_details else [],
            "topic_conflicts": topic_conflicts if include_details else [],
        }
        report["report_path"] = self._write_health_report(report)
        return report

    def _handle_health_report(self, args: dict) -> str:
        import json
        window_hours = int(args.get("window_hours", 24))
        include_details = bool(args.get("include_details", True))
        result = self._health_report(window_hours=window_hours, include_details=include_details)
        return json.dumps(result, ensure_ascii=False)

    def _handle_approve(self, args: dict) -> str:
        import json
        fid = args.get("id")
        if fid is None:
            return json.dumps({"error": "id required"})
        self._store.mark_status(int(fid), "active", args.get("reason"), action="approve", source="tool")
        return json.dumps({"status": "approved", "id": fid}, ensure_ascii=False)

    def _handle_reject(self, args: dict) -> str:
        import json
        fid = args.get("id")
        if fid is None:
            return json.dumps({"error": "id required"})
        self._store.mark_status(int(fid), "archived", args.get("reason"), action="reject", source="tool")
        return json.dumps({"status": "rejected", "id": fid}, ensure_ascii=False)


    def _handle_create_plan(self, args: dict) -> str:
        import json
        operations = args.get("operations") or []
        if not isinstance(operations, list) or not operations:
            return json.dumps({"error": "operations required"}, ensure_ascii=False)
        result = self._create_plan(operations, reason=args.get("reason"))
        return json.dumps(result, ensure_ascii=False)

    def _handle_get_plan(self, args: dict) -> str:
        import json
        plan_id = args.get("plan_id")
        if not plan_id:
            return json.dumps({"error": "plan_id required"}, ensure_ascii=False)
        plan = self._load_plan(str(plan_id))
        return json.dumps(plan, ensure_ascii=False)

    def _handle_list_plans(self, args: dict) -> str:
        import json
        limit = int(args.get("limit", 20))
        plans = []
        for path in sorted(self._plan_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
                plans.append({
                    "plan_id": plan.get("plan_id"),
                    "status": plan.get("status"),
                    "count": plan.get("count"),
                    "applied": plan.get("applied", 0),
                    "path": str(path),
                    "created_at": plan.get("created_at"),
                    "applied_at": plan.get("applied_at"),
                })
            except Exception:
                continue
        return json.dumps({"plans": plans, "count": len(plans)}, ensure_ascii=False)

    def _handle_apply_plan(self, args: dict) -> str:
        import json
        plan_id = args.get("plan_id")
        if not plan_id:
            return json.dumps({"error": "plan_id required"}, ensure_ascii=False)
        if args.get("confirm_text") != "我确认执行计划":
            return json.dumps({"error": "confirm_text must be exactly 我确认执行计划"}, ensure_ascii=False)
        result = self._apply_plan(str(plan_id))
        return json.dumps(result, ensure_ascii=False)

    def _handle_events(self, args: dict) -> str:
        import json
        fid = args.get("id")
        limit = int(args.get("limit", 50))
        events = self._store.list_events(memory_id=int(fid) if fid is not None else None,
                                         limit=limit, action=args.get("event_action"))
        return json.dumps({"events": events, "count": len(events)}, ensure_ascii=False)

    def _handle_explain(self, args: dict) -> str:
        import json
        query = args.get("query", "").strip()
        limit = int(args.get("limit", self._top_k * 3))
        if not query:
            return json.dumps({"error": "query required"})
        embedding = self._embed.embed(query)
        raw_results = self._store.search(embedding, limit=limit, include_inactive=True)

        vector_candidates = []
        for r in raw_results:
            score = r.get("score", 0)
            status = r.get("status", "active")
            included = score >= self._min_score and status == "active"
            reason = "included" if included else (
                f"score {score:.3f} below min_score {self._min_score}" if score < self._min_score else
                f"status={status} not active"
            )
            vector_candidates.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "score": round(score, 4),
                "category": r.get("category"),
                "memory_type": r.get("memory_type"),
                "topic_key": r.get("topic_key"),
                "status": status,
                "search_path": "vector",
                "included": included,
                "reason": reason,
            })

        keyword_results = self._store.keyword_search(query, limit=limit)
        keyword_candidates = []
        for r in keyword_results:
            status = r.get("status", "active")
            keyword_candidates.append({
                "id": r.get("id"),
                "content": r.get("content"),
                "category": r.get("category"),
                "topic_key": r.get("topic_key"),
                "status": status,
                "search_path": "keyword",
                "included": status == "active",
                "reason": "included" if status == "active" else f"status={status} not active",
            })

        return json.dumps({
            "query": query,
            "min_score": self._min_score,
            "top_k": self._top_k,
            "search_method": "ivf" if self._store._ivf_trained else "bruteforce",
            "ivf_trained": self._store._ivf_trained,
            "vector_candidates": vector_candidates,
            "vector_count": len(vector_candidates),
            "vector_included": sum(1 for c in vector_candidates if c["included"]),
            "keyword_candidates": keyword_candidates,
            "keyword_count": len(keyword_candidates),
        }, ensure_ascii=False)

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
