# Hermes VecMem

**Vector memory provider for Hermes Agent** — semantic search over stored facts using sqlite-vec.

> 中文版见 [README.zh-CN.md](README.zh-CN.md)

## What is this?

A pluggable memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com) that stores memories as vector embeddings and retrieves them by **semantic similarity**, not just keyword matching.

Built on [sqlite-vec](https://github.com/asg017/sqlite-vec) — zero external processes, pure SQLite extension.

## Features

- **Semantic search** — find related memories by meaning via vector similarity
- **FTS5 keyword search** — exact match fallback
- **Auto-extraction** — extract facts from conversation every N turns (LLM + regex fallback)
- **Memory mirroring** — built-in `memory` tool writes auto-sync to vector store
- **Adaptive dimensions** — auto-detects embedding dimension (384/1024/1536/...), **auto-rebuilds on model change**
- **Embedding cache** — deduplicates API calls for repeated text
- **Three-tier degradation** — API → local model → TF-IDF feature hashing, never crashes

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
| Fallback | auto when API/local fail | none | 384 | ⭐⭐⭐ |

### Local embedding (optional)

```yaml
    embed_mode: local
    model_name: all-MiniLM-L6-v2
```

Requires: `pip install sentence-transformers`

### LLM extraction

Facts are extracted from conversation using LLM (more accurate than regex), falling back to regex on failure:

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
| `vecmem delete id=N` | Delete a fact |
| `vecmem list limit=N` | Recent facts |
| `vecmem stats` | Memory statistics |

## Project Structure

```
hermes-vecmem/
├── plugins/memory/vecmem/       ← Plugin source (drop into Hermes)
│   ├── __init__.py              ← MemoryProvider + tool handler
│   ├── store.py                 ← sqlite-vec DB layer
│   ├── embed.py                 ← Embedding engine
│   ├── plugin.yaml              ← Plugin metadata
│   └── README.md                ← Plugin-level docs
├── README.md                    ← This file
├── README.zh-CN.md              ← Chinese docs
├── install.sh                   ← Automated install script
├── requirements.txt             ← Python dependencies
└── .gitignore
```

## How it Works

```
User message → sync_turn() → LLM/regex extract facts → embed → store
                                                    ↓
Next turn → prefetch() → embed query → vec_search → top-k → system prompt
```

## Comparison

| Feature | vecmem | holographic (official) |
|---------|--------|----------------------|
| Search | Vector semantics (sqlite-vec) | HRR symbolic algebra |
| Keywords | ✅ FTS5 | ✅ FTS5 |
| Precision | ⭐⭐⭐⭐ (real embeddings) | ⭐⭐ (algebraic) |
| Dimension migration | ✅ auto-rebuild | ❌ |
| Dependencies | sqlite-vec + httpx | None (numpy optional) |
| Embedding | API/local/fallback | None (no embeddings) |

## License

MIT
