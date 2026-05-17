# Hermes VecMem

**Vector memory provider for Hermes Agent** — semantic search over stored facts using sqlite-vec.

> 中文版见 [README.zh-CN.md](README.zh-CN.md)

## What is this?

A pluggable memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com) that stores memories as vector embeddings and retrieves them by **semantic similarity**, not just keyword matching.

Built on [sqlite-vec](https://github.com/asg017/sqlite-vec) — zero external processes, pure SQLite extension.

## Features

- **Semantic search** — find related memories by meaning via vector similarity
- **FTS5 keyword search** — exact match fallback
- **Auto-extraction** — extract facts from conversation every N turns
- **Memory mirroring** — built-in `memory` tool writes auto-sync to vector store
- **Adaptive dimensions** — auto-detects embedding dimension (384/768/1536/...)
- **Embedding cache** — deduplicates API calls for repeated text
- **Graceful degradation** — API → local model → hash fallback, never crashes

## Requirements

- Python 3.10+
- Hermes Agent (git checkout at `~/.hermes/hermes-agent/`)

## Quick Install

```bash
# 1. Copy plugin into Hermes source tree
cp -r plugins/memory/vecmem ~/.hermes/hermes-agent/plugins/memory/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure
hermes config set memory.provider vecmem
hermes config set memory.vecmem.embed_mode api
hermes config set memory.vecmem.api_base https://api.deepseek.com
# Or use any OpenAI-compatible embedding endpoint
```

## Configuration

In `config.yaml`:

```yaml
memory:
  provider: vecmem
  vecmem:
    embed_mode: api                    # api | local
    api_base: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}       # or hardcode
    model: deepseek-embedding          # or text-embedding-3-small, etc.
    top_k: 5                           # prefetch results per turn
    min_score: 0.3                     # minimum similarity threshold
    sync_interval: 3                   # auto-extract every N turns
```

### Local embedding (optional)

```yaml
    embed_mode: local
    model_name: all-MiniLM-L6-v2        # any sentence-transformers model
```

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
├── plugins/memory/vecmem/       ← Plugin source (drop into Hermes repo)
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
User message → sync_turn() → regex extract facts → embed → store
                                                    ↓
Next turn → prefetch() → embed query → vec_search → top-k → system prompt
```

## Comparison

See discussion in [the original Hermes session](https://github.com/NousResearch/hermes-agent) or compare:

| Feature | vecmem | holographic (official) |
|---------|--------|----------------------|
| Search | Vector semantics (sqlite-vec) | HRR symbolic algebra |
| Keywords | ✅ FTS5 | ✅ FTS5 |
| Precision | ⭐⭐⭐⭐ (real embeddings) | ⭐⭐ (algebraic) |
| Dependencies | sqlite-vec + httpx | None (numpy optional) |
| Embedding | API/local/fallback | None (no embeddings) |

## License

MIT
