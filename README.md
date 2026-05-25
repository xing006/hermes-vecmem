# Hermes VecMem

<p align="center">
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="https://github.com/xing006/hermes-vecmem/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/xing006/hermes-vecmem"><img src="https://img.shields.io/badge/GitHub-hermes--vecmem-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
</p>

**Vector memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com)** — semantic search over stored facts using sqlite-vec. Zero external processes, pure SQLite extension.

---

## Features

| Capability | Description |
|------------|-------------|
| **Semantic search** | Find related memories by meaning via vector similarity, not just keywords |
| **FTS5 keyword search** | Exact match fallback — `vecmem keyword query="..."` |
| **Auto-extraction** | Extract facts from conversation every N turns (LLM + regex fallback) |
| **Memory mirroring** | Built-in `memory` tool writes auto-sync to vector store |
| **Adaptive dimensions** | Auto-detects embedding dimension (384/1024/1536/...), **auto-rebuilds on model change** |
| **Embedding cache** | Deduplicates API calls for repeated text |
| **Three-tier degradation** | API → local model → TF-IDF feature hashing — never crashes |

---

## Architecture

```
User message → sync_turn() → LLM/regex extract facts → embed → store
                                                  ↓
Next turn → prefetch() → embed query → vec_search → top-k → system prompt
```

---

## Installation

### Option 1: Drop into Hermes plugins

```bash
# 1. Copy plugin
cp -r plugins/memory/vecmem $HERMES_HOME/plugins/memory/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure (see below)
```

> `$HERMES_HOME` defaults to `~/.hermes/` (Linux/macOS) or `~/AppData/Local/hermes/` (Windows).

### Option 2: Install script

```bash
bash install.sh
```

---

## Configuration

In `config.yaml`:

```yaml
memory:
  provider: vecmem
  vecmem:
    embed_mode: api                    # api | local | fallback
    api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: ${DASHSCOPE_API_KEY}
    model: text-embedding-v3           # Qwen embedding (1024d)
    top_k: 5                           # prefetch results per turn
    min_score: 0.3                     # minimum similarity threshold
    governance_mode: light             # light | advanced; light is the personal-use default
    llm_extract: true                  # enable LLM-based fact extraction
    llm_model: deepseek-chat           # model for LLM extraction
    sync_interval: 3                   # auto-extract every N turns
```

> **Note**: DeepSeek deprecated their embedding API in mid-2026 (`/embeddings` returns 404). Use Qwen DashScope (`text-embedding-v3`), any OpenAI-compatible endpoint, or local mode instead.

### Embedding modes

| Mode | Config | Deps | Dimension | Quality |
|------|--------|------|-----------|---------|
| API | `embed_mode: api` | httpx | model-dependent (1024/1536) | ⭐⭐⭐⭐⭐ |
| Local | `embed_mode: local` | sentence-transformers | 384 (all-MiniLM-L6-v2) | ⭐⭐⭐⭐ |
| Fallback | auto when API/local fail | none | 384 (feature hashing) | ⭐⭐⭐ |

### Local embedding (optional)

```yaml
    embed_mode: local
    model_name: all-MiniLM-L6-v2
```

Requires: `pip install sentence-transformers`

### LLM extraction

```yaml
    llm_extract: true
    llm_model: deepseek-chat
```

### Dimension auto-migration

When switching embedding models (e.g., from `text-embedding-v3` 1024d to `all-MiniLM-L6-v2` 384d), vecmem automatically:

1. Detects the dimension change
2. Drops the old vector table and sqlite-vec's 5 internal tables
3. Clears the embedding cache and IVF index
4. Rebuilds the vector table with the new dimension
5. Preserves all text memories (dimension-independent)

---

## Usage

### Automatic (no action needed)

- Memories are auto-prefetched each turn based on conversation context
- Built-in memory writes are mirrored to vector store automatically
- Facts are extracted from conversation every N turns

### Manual tool: `vecmem`

| Action | Description |
|--------|-------------|
| `vecmem add content=...` | Store a fact |
| `vecmem search query=...` | Semantic search |
| `vecmem keyword query=...` | Keyword search |
| `vecmem list limit=N` | Recent facts |
| `vecmem archive id=N` | Hide a fact without deleting it |
| `vecmem restore id=N` | Restore an archived fact |
| `vecmem stats` | Memory statistics |

Light mode is the default visible tool surface. Set `governance_mode: advanced` to expose preview/review/audit/plan/index actions: `preview_add`, `review_list`, `approve`, `reject`, `preview_archive`, `preview_restore`, `preview_approve`, `preview_reject`, `create_plan`, `get_plan`, `list_plans`, `apply_plan`, `events`, `build_index`, and `set_probe`.

---

## Project Structure

```
hermes-vecmem/
├── plugins/memory/vecmem/       ← Plugin source (drop into Hermes)
│   ├── __init__.py              ← MemoryProvider + tool handler
│   ├── store.py                 ← sqlite-vec DB layer
│   ├── embed.py                 ← Embedding engine
│   ├── plugin.yaml              ← Plugin metadata
│   └── README.md                ← Plugin-level docs
├── README.md                    ← This file (English)
├── README.zh-CN.md              ← Chinese version
├── install.sh                   ← Automated install script
├── requirements.txt             ← Python dependencies
└── .gitignore
```

---

## Comparison vs Official Holographic

| Feature | vecmem | holographic (official) |
|---------|--------|----------------------|
| Search | Vector semantics (sqlite-vec) | HRR symbolic algebra |
| Keywords | ✅ FTS5 | ✅ FTS5 |
| Precision | ⭐⭐⭐⭐ (real embeddings) | ⭐⭐ (algebraic) |
| Dimension migration | ✅ auto-rebuild | ❌ |
| Dependencies | sqlite-vec + httpx | None (numpy optional) |
| Embedding | API/local/fallback | None (no embeddings) |

---

## License

MIT
