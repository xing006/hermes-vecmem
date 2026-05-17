# sqlite-vec 向量记忆插件

基于 sqlite-vec 的 Hermes 记忆插件，用向量相似度替代纯文本关键词匹配。

> 项目主页: [hermes-vecmem](https://github.com/xing006/hermes-vecmem) | [README.zh-CN.md](../../../README.zh-CN.md)

## 工作原理

```
对话文本 → EmbedEngine → 向量 → sqlite-vec → 语义搜索
                                     ↓
                              FTS5 全文搜索 ← 关键词兜底
```

每轮对话前自动做向量召回（prefetch），找到最相关的历史记忆注入到 system prompt。也支持手动操作：`vecmem search/add/delete/list`。

## 配置

在 `config.yaml` 的 `memory` 段：

```yaml
memory:
  provider: vecmem
  vecmem:
    embed_mode: api           # api | local
    api_base: https://api.deepseek.com
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-embedding
    # 可选 API:
    #   api_base: https://api.openai.com
    #   api_key: ${OPENAI_API_KEY}
    #   model: text-embedding-3-small (返回 1536 维)
    #
    # local 模式（需要 pip install sentence-transformers）:
    # embed_mode: local
    # model_name: all-MiniLM-L6-v2
    top_k: 5
    min_score: 0.3
```

## 支持的命令

| 命令 | 说明 |
|------|------|
| `vecmem add content=...` | 存入一段记忆 |
| `vecmem search query=...` | 语义搜索 |
| `vecmem keyword query=...` | 关键词搜索 |
| `vecmem delete id=N` | 删除记忆 |
| `vecmem list limit=N` | 列出最近记忆 |
| `vecmem stats` | 查看统计 |

## 文件

```
plugins/memory/vecmem/
├── __init__.py    # MemoryProvider 实现 + 工具注册
├── plugin.yaml    # 插件元数据
├── store.py       # sqlite-vec 数据库层
├── embed.py       # 嵌入引擎（API / local / fallback）
└── README.md      # 插件文档
```

## 依赖

- `sqlite-vec` — SQLite 向量扩展
- `httpx` — API 嵌入模式的 HTTP 客户端
- （可选）`sentence-transformers` — 本地嵌入模式
