# research_review 模式：启动、迁移与 5 分钟验收演示

本文档对应 `docs/HelioPrior_v2_Implementation.md` 中的 research_review 路线，记录 T0-T7 已接线的可运行能力与离线验收方式。

主定位保持不变：大模型多智能体驱动的空间天气科研协作与决策支持；真实案例是 LHAASO 日影对未来 IMF By 的增量预测研究。系统同时保留原有 legacy（先羲科研闭环）模式与真实科研能力。假设树仍然存在，作为竞争解释与区分性实验的工具，不再是唯一界面中心。

## 1. 启动命令

### 后端

项目根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
python scripts/live_workflow_server.py
```

Windows 直接使用已配置的 Python 环境：

```powershell
cd D:\HelioPrior
python scripts/live_workflow_server.py
```

后端默认监听 `http://127.0.0.1:8765`，健康检查为 `GET /api/health`。

### 前端

```bash
cd frontend
pnpm install
pnpm dev
```

前端默认监听 `http://127.0.0.1:5173`。生产环境先执行 `pnpm build`，再将 `frontend/dist` 交给 nginx。

## 2. 环境变量样例

从 `.env.example` 复制 `.env`，以下为最小配置样例（不包含真实 Key）：

```dotenv
# Copy this file to `.env` in the project root: D:\HelioPrior\.env
DASHSCOPE_API_KEY=
BAILIAN_API_KEY=
BAILIAN_MODEL=qwen3.8-flash
DASHSCOPE_BASE_URL=https://llm-jz60biyiqkkwzssm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
HELIOPRIOR_CANDIDATE_DESIGN_WORKERS=2
```

两个 Key 都留空时，后端仍可启动，研究复核、仲裁、候选实验与执行链路会走明确标注的 stub 输出：意见与仲裁记录带 `is_stub: true`，来源记为 `program`，报告中的 stub 统计会单独列出。系统不会把 stub 伪装成真实模型调用或真实科学结果。

## 3. 迁移与数据隔离

- 新增 research_review 项目写入 `runtime/research/<research_id>/`，每个项目拥有独立目录。
- 每次 API 变更必须携带 `expected_revision`，仓库写入新的不可变 revision：`runtime/research/<research_id>/revisions/<0001>/revision.json`，`manifest.json` 记录当前修订号。
- legacy 状态仍位于 `runtime/live_session/current/`，两种存储互不覆盖。
- 结论版本不可覆盖：新版本必须提供 `revision_note`，引用的证据 ID 必须真实存在，状态必须走白名单；人工 PI 决定与程序建议都会落盘。
- `HELIOPRIOR_CANDIDATE_DESIGN_WORKERS=2` 默认最多 2 个并发生成请求；复核编排器首轮最多并发 2 个 LLM 请求，分歧仲裁最多 1 次调用。
- 引用与证据定位必须通过白名单和版本校验；不存在的证据、协议或指标引用会返回明确错误，而不是静默忽略。

## 4. v2 HTTP 接口（已接线）

以下接口均挂载在 `/api/v2` 下。

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| POST / GET | `/api/v2/research` | 创建研究项目 / 列出项目 |
| GET | `/api/v2/research/{research_id}` | 读取项目详情与当前修订 |
| POST | `/api/v2/research/{research_id}/artifacts` | 导入预测 CSV、协议 JSON 等产物 |
| POST | `/api/v2/research/{research_id}/preflight` | 生成证据包并执行证据预检 |
| POST | `/api/v2/research/{research_id}/reviews` | 启动同一证据包上的双路独立复核 |
| POST | `/api/v2/research/{research_id}/reviews/{review_id}/retry` | 单角色失败重试，复用同一 review |
| GET | `/api/v2/research/{research_id}/disputes` | 列出分歧与仲裁记录 |
| POST | `/api/v2/research/{research_id}/plans` | 生成候选验证方案 |
| POST | `/api/v2/research/{research_id}/approvals` | 人工 PI 批准验证协议 |
| POST | `/api/v2/research/{research_id}/executions` | 执行已批准方案并落盘结果 |
| GET / POST | `/api/v2/research/{research_id}/claims/{claim_id}/versions` | 结论版本查询 / 保存新版本 |
| GET | `/api/v2/research/{research_id}/report` | 导出完整审计报告 JSON |

`hypotheses`、`evidence`、`decisions`、`jobs` 以及 `claims` 的非版本操作目前明确返回 `501 not_implemented`，避免把路线图项目伪装成已交付能力。

## 5. 测试记录

后端测试命令：

```powershell
cd D:\HelioPrior
python -m unittest tests.test_research_validation tests.test_research_api tests.test_review_orchestrator tests.test_unified_execution -v
```

2026-10-10 实测：42 个测试全部通过，总耗时约 34.6 秒。覆盖项目创建与幂等、产物导入、证据预检、双路独立意见、分歧仲裁、单角色重试、候选规划、人工批准、门控执行、结论版本、报告导出，以及端到端闭环中主 LLM 调用数量为 4 次的断言。前端 `pnpm build` 通过。

## 6. 5 分钟验收演示

演示前确保后端与 Frontend 均已启动，打开 `http://127.0.0.1:5173/research`。全程可按以下 7 步执行，形成完整闭环。

### 第 1 步：新建研究或导入结论（约 1 分钟）

- 选择“新建问题 / 导入结论”。
- 默认目标为 `imf_by_gse / by_gse / nT / GSE`，提前量默认 3 天。
- 导入结论模式填写一条结论原文，例如“LHAASO 日影南北偏移对 3 天后 IMF By 有独立增量预测信息”。
- 创建后进入项目详情页，确认 `research_id`、`revision` 与协议 `schema_version=research-review/1.0`。

### 第 2 步：导入证据并预检（约 1 分钟）

- 在“证据包”区上传一份预测 CSV / 协议 JSON。
- 执行证据预检，确认生成 `evidence_id`、证据包哈希、协议哈希与数据清单哈希。
- 页面展示证据核查结果（通过 / 未通过 / 未知），失败项必须可见。

### 第 3 步：双路独立复核（约 1 分钟）

- 发起复核，系统在同一证据包上生成两路独立审查意见。
- 界面上确认两位审查角色、各自立场、引用指标与 `is_stub` 标记。
- 若两路立场一致，直接进入候选规划；若冲突，会生成分歧记录。

### 第 4 步：分歧与人工仲裁（约 1 分钟）

- 在“分歧与仲裁”区查看立场冲突和仲裁意见。
- 系统给出结构化分歧说明，人工 PI 确认仲裁结果。
- 不把少数服从多数当作程序事实；数值检查与状态变更由程序执行。

### 第 5 步：选择验证方案并批准执行（约 1 分钟）

- 打开候选验证方案，选择一条可执行的区分性实验。
- 人工 PI 批准协议，系统执行滚动回测（同一共同样本上的基准组 / 增强组）。
- 执行状态必须显示为成功 / 失败 / 部分失败，失败不会被包装成成功。

### 第 6 步：保存带证据的新结论版本

- 在“结论版本与报告”区基于本次执行的 `evidence_id` 保存新版本。
- 填写 `revision_note`，选择白名单状态，确认程序建议与人工决定。
- 查看版本历史，旧版本保持不变，仅新增当前版本。

### 第 7 步：导出审计报告

- 点击“生成报告”并下载 JSON。
- 报告中应包含项目信息、全部结论版本、证据包与定位、协议、复核意见、争议、候选实验、决定、任务、模型调用、未解决问题与 stub 统计。
- 若使用离线 stub，报告的模型调用记录与界面都带有可识别的 stub 标记。

## 7. 当前实现边界

- 前端已接线页面：`/research`、`/research/:researchId`、`/research/:researchId/evidence`。主页与主导航仍以 legacy 闭环为主，`/research` 作为 research_review 工作台入口，未声称完成全站 UI 重构。
- research_review 存储与 legacy 存储完全隔离，互不迁移、互不覆盖。
- 主链路目标 LLM 调用预算约 3-5 次；端到端闭环测试以 4 次主调用通过。
- 未接线功能（如 `hypotheses`、`evidence`、`decisions`、`jobs` 等 v2 子接口）返回明确的 501，不计入“已完成”。
- 科学结论取决于真实数据与运行结果；stub 模式只用于离线验收，不产出可对外引用的科学结论。
