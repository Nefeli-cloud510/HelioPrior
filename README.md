# HelioPrior

先羲 HelioPrior：一个由人工 PI、推理规划层、实验执行层与数据分析层共同构成的闭环科学推理系统。系统以真实太阳风数据为对象，默认使用 `StandardScaler + Ridge(solver="svd")` 并按时序折在 alpha 网格 `[0.01, 0.1, 1, 10, 100, 1000]` 内选参，同时保留 ElasticNet 作为对照模型，并调用 Qwen 系列模型生成科学假设、质询、不确定性与候选实验设计。

## 项目结构

- `core/`：中央控制、假设生成与质询、不确定性识别、候选实验生成、实验执行、评价与状态更新。
- `models/`：Ridge（默认，svd + alpha 网格选参）与 ElasticNet（对照）线性模型的数据读取、特征构造、训练、预测与可视化。
- `scripts/`：实时工作流服务与修复脚本。
- `frontend/`：React + TypeScript + Vite 前端。
- `runtime/live_session/current/`：会话状态、上传数据、实验结果与审计记录。
- `runtime/research/`：research_review 模式的项目隔离存储，按项目与修订版本组织。

## research_review 模式

项目同时保留原有 legacy 科研闭环与新的 research_review 科研复核工作流。后者面向“结论与证据导入、同证据包双路独立复核、分歧到可执行验证的映射、结论版本与人工决定”的闭环，真实案例为 LHAASO 日影对未来 IMF By 的增量预测；假设树继续作为竞争解释与区分性实验工具。

默认目标为 `imf_by_gse / by_gse / nT / GSE`，提前量默认 3 天。前端入口为 `/research`，后端接口挂载在 `/api/v2/research`。项目隔离、接口清单、离线 stub、测试记录与 5 分钟验收演示见 [docs/RESEARCH_REVIEW_DEMO.md](docs/RESEARCH_REVIEW_DEMO.md)。

## 本地开发

### 后端

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
python scripts/live_workflow_server.py
```

后端默认监听 `http://127.0.0.1:8765`，健康检查为 `GET /api/health`。

### 前端

```bash
cd frontend
pnpm install
pnpm dev
```

前端默认监听 `http://127.0.0.1:5173`。生产环境请先执行 `pnpm build`，再将 `frontend/dist` 交给 nginx。

## 环境变量

复制 `.env.example` 到项目根目录 `.env`（`.env` 已在 `.gitignore` 中），至少配置 `DASHSCOPE_API_KEY` 或 `BAILIAN_API_KEY`。当前默认使用 `qwen3.8-flash`，模型价格与计费说明见 `core/runtime_config.py`。

## 部署

阿里云 Ubuntu 服务器完整部署步骤见 [DEPLOY.md](DEPLOY.md)。
