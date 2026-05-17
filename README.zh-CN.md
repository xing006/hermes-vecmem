# Hermes VecMem

**Hermes Agent 向量记忆插件** — 基于 sqlite-vec 的语义搜索记忆库。

## 这是什么？

一个可插拔的 [Hermes Agent](https://hermes-agent.nousresearch.com) 记忆插件。把记忆存为向量嵌入，按**语义相似度**（而非关键词匹配）检索。

基于 [sqlite-vec](https://github.com/asg017/sqlite-vec) — 零额外进程，纯 SQLite 扩展。

## 特性

- **语义搜索** — 通过向量相似度找到含义相近的记忆
- **FTS5 关键词搜索** — 精确匹配兜底
- **自动提取** — 每 N 轮从对话中自动提取事实（支持 LLM 提取 + 正则兜底）
- **记忆镜像** — 内置 `memory` 工具的写入自动同步到向量库
- **维度自适应** — 自动探测嵌入维度（384/1024/1536/...），**模型切换时自动重建向量表**
- **嵌入缓存** — 相同文本重复调用走缓存，省 API 费用
- **三级降级** — API → 本地模型 → TF-IDF 特征哈希，稳健降级

## 安装

### 方法一：直接放入 Hermes 插件目录

```bash
# 1. 复制插件
cp -r plugins/memory/vecmem $HERMES_HOME/plugins/memory/

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 配置（见下方配置示例）
```

> `$HERMES_HOME` 默认为 `~/.hermes/`（Linux/macOS）或 `~/AppData/Local/hermes/`（Windows）。

### 方法二：安装脚本

```bash
bash install.sh
```

## 配置

在 `config.yaml` 中：

```yaml
memory:
  provider: vecmem
  vecmem:
    embed_mode: api                    # api | local | fallback
    api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: ${DASHSCOPE_API_KEY}
    model: text-embedding-v3           # 通义千问嵌入（1024维）
    top_k: 5                           # 每轮 prefetch 条数
    min_score: 0.3                     # 最小相似度阈值
    llm_extract: true                  # 启用 LLM 提取事实
    llm_model: deepseek-chat           # LLM 提取用的模型
    sync_interval: 3                   # 每 N 轮自动提取一次
```

> **注意**：DeepSeek 已于 2026 年中下线 embedding API（`/embeddings` 返回 404），`deepseek-embedding` 模型不可用。推荐使用通义千问 DashScope (`text-embedding-v3`)，也支持任何 OpenAI 兼容的嵌入端点。

### 嵌入模式

| 模式 | 配置 | 依赖 | 维度 | 精度 |
|------|------|------|------|------|
| API | `embed_mode: api` | httpx | 取决于模型（1024/1536） | ⭐⭐⭐⭐⭐ |
| 本地 | `embed_mode: local` | sentence-transformers | 384（all-MiniLM-L6-v2） | ⭐⭐⭐⭐ |
| 降级 | API/本地失败时自动 | 无 | 384 | ⭐⭐⭐ |

### 本地嵌入（可选）

```yaml
    embed_mode: local
    model_name: all-MiniLM-L6-v2
```

需要安装：`pip install sentence-transformers`

### LLM 提取

从对话中提取事实时，默认使用 LLM（效果优于正则），失败时自动降级到正则提取：

```yaml
    llm_extract: true          # 启用
    llm_model: deepseek-chat   # 使用的模型
```

### 维度自动迁移

切换嵌入模型时（如从 `text-embedding-v3` 1024 维换到 `all-MiniLM-L6-v2` 384 维），vecmem 会自动：

1. 检测维度变化
2. 删除旧的向量表和内部表（sqlite-vec 的 5 个隐藏表）
3. 清空嵌入缓存和 IVF 索引
4. 用新维度重建向量表
5. 保留文本记忆不变

## 使用方法

### 自动（无需操作）

- 每轮对话自动根据上下文 prefetch 相关记忆
- 内置 `memory` 工具写入自动镜像到向量库
- 每 N 轮自动从对话提取事实

### 手动工具：`vecmem`

| 动作 | 说明 |
|------|------|
| `vecmem add content=...` | 存入事实 |
| `vecmem search query=...` | 语义搜索 |
| `vecmem keyword query=...` | 关键词搜索 |
| `vecmem delete id=N` | 删除 |
| `vecmem list limit=N` | 列出最近 |
| `vecmem stats` | 统计 |

## 项目结构

```
hermes-vecmem/
├── plugins/memory/vecmem/       ← 插件源码（可直接放入 Hermes）
│   ├── __init__.py              ← MemoryProvider + 工具
│   ├── store.py                 ← sqlite-vec 数据库
│   ├── embed.py                 ← 嵌入引擎
│   ├── plugin.yaml              ← 插件元数据
│   └── README.md                ← 插件级文档
├── README.md                    ← 项目文档（英文）
├── README.zh-CN.md              ← 项目文档（中文）
├── install.sh                   ← 安装脚本
├── requirements.txt             ← Python 依赖
└── .gitignore
```

## 工作原理

```
用户消息 → sync_turn() → LLM/正则提取事实 → 嵌入 → 存储
                                                  ↓
下一轮 → prefetch() → 嵌入查询 → 向量搜索 → top-k → system prompt
```

## 与官方 holographic 对比

| 特性 | vecmem | holographic（官方） |
|------|--------|-------------------|
| 搜索方式 | 向量语义搜索（sqlite-vec） | HRR 符号代数 |
| 关键词 | ✅ FTS5 | ✅ FTS5 |
| 精度 | ⭐⭐⭐⭐ 真实嵌入 | ⭐⭐ 代数近似 |
| 维度迁移 | ✅ 自动重建 | ❌ |
| 依赖 | sqlite-vec + httpx | 无（numpy 可选） |
| 嵌入 | API/本地/降级 | 无嵌入 |

## 协议

MIT
