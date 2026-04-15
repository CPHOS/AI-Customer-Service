# CPHOS AI Customer Service

基于 RAG + 多智能体的 CPHOS 智能客服系统。通过分类路由、双路并行检索和多轮验证，自动回答关于 CPHOS 常见问题。

---

## 项目架构

```
AI-Customer-Service/
├── main.py              # CLI 入口，参数解析，构建 Pipeline
├── server.py            # Uvicorn 入口
├── pipeline.py          # 核心编排：Classifier → Executor → Critic → Verifier
├── config.py            # 全局配置，支持环境变量覆盖
├── agents/
│   ├── base.py          # BaseAgent：封装 LLM 调用、重试逻辑及 tool-call 循环
│   ├── classifier.py    # ClassifierAgent：8 类话题路由（A–H）
│   ├── executor.py      # ExecutorAgent：RAG 回答 + 时效性问题官网抓取
│   ├── critic.py        # CriticAgent：从双路候选中选出更优回答
│   └── verifier.py      # VerifierAgent：验证回答质量，输出最终回复
├── rag/
│   ├── retriever.py     # 基于 OpenAI Embeddings + NumPy 的内存向量检索
│   └── document.py      # 文档加载器，支持 .txt / .md / .pdf / .yml
├── utils/
│   ├── logger.py        # 结构化日志（UTC+8）+ ConversationLogger（JSONL 对话记录）
│   └── web_fetch.py     # CPHOS 官网页面抓取（白名单枚举，非任意 URL）
├── app/                 # FastAPI 应用
│   ├── main.py          # 应用工厂 create_app()
│   ├── config.py        # Pydantic Settings
│   ├── routers/         # /chat, /health 路由
│   └── schemas/         # 请求/响应模型
├── docker-compose.yml   # Docker Compose 部署配置
├── Dockerfile           # 容器镜像构建
├── .env.example         # 环境变量模板
├── pyproject.toml       # uv 项目定义
└── requirements.txt     # pip 依赖列表
```

---

## 快速开始

### 前置条件

- Python ≥ 3.12
- OpenAI 兼容 API Key（支持 OpenRouter）

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OPENROUTER_API_KEY 和 OPENROUTER_BASE_URL
```

### 2. 安装依赖

方式一：pip

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

方式二：uv（推荐）

```bash
# 安装 uv：https://docs.astral.sh/uv/
uv sync
```

### 3. 构建知识库索引

```bash
# 从 references/ 目录构建并保存索引
python main.py --refs-dir references/ --save-index cphos.npz

# 使用 uv
uv run python main.py --refs-dir references/ --save-index cphos.npz
```

### 4. 启动交互对话

```bash
# 加载已有索引
python main.py --load-index cphos.npz

uv run python main.py --load-index cphos.npz

# 启动时检查 references/ 是否有更新，自动重建索引
python main.py --load-index cphos.npz --check-refs

uv run python main.py --load-index cphos.npz --check-refs

# 显示详细 pipeline 日志
python main.py --load-index cphos.npz --verbose

uv run python main.py --load-index cphos.npz --verbose

# 直接启动
uv run cphos-chat
```

### 5. 获取帮助

```bash
python main.py --help

uv run python main.py --help
```

---

## Docker 部署

镜像由 GitHub Actions 自动构建并推送到 GHCR。服务器上无需构建，只需拉取运行。

### 首次部署

```bash
mkdir ~/AI-CS && cd ~/AI-CS

# 准备配置
cp .env.example .env        # 编辑 .env 填入 API Key 等
# 将 references/ 和 cphos.npz 上传到此目录

# 启动（含 Redis）
docker compose pull && docker compose up -d
```

### 更新版本

```bash
docker compose pull app && docker compose up -d app
docker image prune -f       # 清理旧镜像
```

---

## Pipeline

每次用户提问经过以下流程处理：

```
用户提问
   │
   ▼
┌─────────────────────────────────┐
│  1. ClassifierAgent             │  话题路由 → A–F（知识库）/ G（超出范围）/ H（时效性）
│     8 类：成绩/身份/赛季/考试/  │
│     小程序/评卷/无关/时效性     │
└─────────────┬───────────────────┘
              │ G → 礼貌拒绝
              │ H → 跳转官网抓取路径 ────────────────┐
              ▼                                      │
┌─────────────────────────────────┐                  │
│  2. Retriever（双路检索）       │                  │
│     Path A：section-hint 检索   │                  │
│     Path B：通用检索            │                  │
└─────────────┬───────────────────┘                  │
              ▼                                      ▼
┌─────────────────────────────────┐   ┌──────────────────────────────────┐
│  3. ExecutorAgent × 2（并行）   │   │  3H. ExecutorAgent + fetch_page  │
│     基于 RAG 上下文生成候选回答 │   │      工具调用：抓取 CPHOS 官网   │
└─────────────┬───────────────────┘   │      白名单页面获取实时信息      │
              ▼                       └──────────────┬───────────────────┘
┌─────────────────────────────────┐                  │
│  4. CriticAgent                 │                  │
│     LLM 评审，选出更优候选回答  │                  │
└─────────────┬───────────────────┘                  │
              ▼                                      ▼
┌─────────────────────────────────────────────────────┐
│  5. VerifierAgent（验证 + 润色）                    │
│     → summarize 输出最终回复                        │
└─────────────┬───────────────────────────────────────┘
              ▼
┌─────────────────────────────────┐
│  6. ConversationLogger          │  追加写入 JSONL，记录问答/分类/耗时
└─────────────────────────────────┘
              │
              ▼
           最终回复
```

---

## 配置

通过 `.env` 文件或环境变量配置（参考 [.env.example](.env.example)）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | （必填） |
| `OPENROUTER_BASE_URL` | API 端点 | `https://openrouter.ai/api/v1` |
| `OPENROUTER_CLASSIFIER_MODEL` | 话题路由模型 | `openai/gpt-4o-mini` |
| `OPENROUTER_EXECUTOR_MODEL` | 回答生成模型 | `openai/gpt-4o` |
| `OPENROUTER_VERIFIER_MODEL` | 质量验证模型 | `openai/gpt-4o-mini` |
| `OPENROUTER_CRITIC_MODEL` | 双路评审模型 | `openai/gpt-4o-mini` |
| `OPENROUTER_EMBEDDING_MODEL` | 向量嵌入模型 | `openai/text-embedding-3-small` |
| `OPENROUTER_CHUNK_WORD_LENGTH` | 知识库分块长度（字）| `150` |
| `OPENROUTER_TOP_K_CHUNKS` | 每次检索返回的 chunk 数 | `5` |
| `OPENROUTER_MAX_RETRIES` | Executor → Verifier 最大重试次数 | `3` |
| `OPENROUTER_ENABLE_DUAL_PATH` | 启用双路并行执行 | `true` |
| `SESSION_BACKEND` | 会话存储后端 | `memory`（生产用 `redis`）|
| `REDIS_URL` | Redis 连接地址 | — |