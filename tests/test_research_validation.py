from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.llm_gateway import LLMGateway
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.research_api import ResearchV2Api
from core.research_repository import ResearchRepository
from core.research_schema import Dispute


def create_payload(
    title: str = "日影观测与 IMF By 增量预测验证",
    question: str = "加入日影后是否改善指定条件下的 By 预测",
) -> dict:
    return {
        "workflow_mode": "research_review",
        "entry_mode": "new_question",
        "data_mode": "authorized_data",
        "title": title,
        "question": question,
        "target": {
            "id": "imf_by_gse",
            "column": "BY, nT (GSE)",
            "unit": "nT",
            "coordinate_system": "GSE",
        },
        "horizon_days": 3,
    }


DESIGN_HINTS = {
    "control": ["By", "Bz"],
    "treatment": ["By", "Bz", "DeltaDec"],
    "forecast_horizon_days": 3,
    "past_lag_days": 2,
    "window_size": 4,
    "use_lag_feature": True,
    "max_lag_day": 2,
    "alpha": 1.0,
    "l1_ratio": 0.5,
    "random_state": 7,
    "test_split_ratio": 0.25,
}


class ScriptedPlannerGateway(LLMGateway):
    """Offline gateway for the experiment-planner role only."""

    def __init__(self) -> None:
        super().__init__(client=None, allow_fallback=True)
        self.responses: dict[str, dict] = {}
        self.calls: list[str] = []

    def generate_structured(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        fallback_factory=None,
        payload_fixer=None,
        temperature=None,
        image_paths=None,
    ):
        if "实验规划者" not in system_prompt:
            raise RuntimeError("test gateway only supports the planner role")
        self.calls.append("experiment_planner")
        response = self.responses.get("experiment_planner")
        if response is not None:
            return response_model.model_validate(response)
        return response_model.model_validate(fallback_factory())


def scripted_candidate(
    experiment_id: str,
    *,
    template_id: str = "paired_replay",
    dispute_ids: list[str] | None = None,
    design: dict | None = None,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "dispute_ids": dispute_ids or ["DSP_TEST"],
        "hypothesis_ids": [],
        "template_id": template_id,
        "design": dict(design or DESIGN_HINTS),
        "rationale": "脚本候选：验证争议对应的受控实验。",
    }


class ResearchValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repository = ResearchRepository(self.root / "runtime")
        self.gateway = ScriptedPlannerGateway()
        self.api = ResearchV2Api(self.repository, gateway=self.gateway)

        time_values = pd.date_range("2021-01-01", periods=36, freq="D")
        omni = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "By": [(-1) ** i * (i % 5 + 1) for i in range(36)],
                "Bz": [(i % 7) - 3 for i in range(36)],
                "Vsw": [350 + i * 2 for i in range(36)],
            }
        )
        lhaaso = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "DeltaDec": [0.1 * ((i % 6) - 3) for i in range(36)],
            }
        )
        self.omni_path = self.root / "omni.csv"
        self.lhaaso_path = self.root / "lhaaso.csv"
        omni.to_csv(self.omni_path, index=False)
        lhaaso.to_csv(self.lhaaso_path, index=False)
        self.data_source = ElasticNetDataSourceConfig(
            omni_file_path=self.omni_path,
            lhaaso_file_path=self.lhaaso_path,
            time_column="TIME",
            target_column="Vsw",
            test_split_ratio=0.25,
            random_state=7,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _data_api(self) -> ResearchV2Api:
        return ResearchV2Api(
            self.repository,
            gateway=self.gateway,
            data_source=self.data_source,
            harness_project_root=self.root / "harness",
        )

    def _create_project(self, api: ResearchV2Api, title: str = "验证任务") -> tuple[str, int]:
        created = api.dispatch("POST", "/api/v2/research", create_payload(title=title))
        self.assertEqual(created.status, 201)
        return created.payload["research_id"], created.payload["revision"]

    def _seed_dispute(
        self,
        api: ResearchV2Api,
        research_id: str,
        revision: int,
        dispute_id: str = "DSP_TEST",
    ) -> int:
        next_state = api.service.repository.mutate(
            research_id,
            expected_revision=revision,
            mutation=lambda state: state.disputes.append(
                Dispute(
                    dispute_id=dispute_id,
                    question="两路审读对增量证据判断冲突，需要受控验证。",
                    issue_code="STANCE_CONFLICT",
                    status="open",
                )
            ),
        )
        return next_state.revision

    def _plan(
        self,
        api: ResearchV2Api,
        research_id: str,
        revision: int,
        **overrides,
    ):
        payload = {
            "expected_revision": revision,
            "dispute_ids": ["DSP_TEST"],
            "template_id": "paired_replay",
            "design_hints": dict(DESIGN_HINTS),
        }
        payload.update(overrides)
        return api.dispatch("POST", f"/api/v2/research/{research_id}/plans", payload)

    def _candidate(self, response) -> dict:
        self.assertEqual(response.status, 200)
        return response.payload["data"]["validation_plan"]["candidates"][0]

    def test_plan_requires_open_dispute(self) -> None:
        research_id, revision = self._create_project(self.api)
        response = self._plan(self.api, research_id, revision, dispute_ids=[])
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "dispute_required")

    def test_plan_rejects_unknown_dispute(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        response = self._plan(
            self.api,
            research_id,
            revision,
            dispute_ids=["DSP_MISSING"],
        )
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "invalid_dispute_ids")

    def test_plan_rejects_candidate_limit(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        self.gateway.responses["experiment_planner"] = {
            "batch_id": "BATCH_TOO_MANY",
            "candidates": [
                scripted_candidate(f"EXP_0{i}") for i in range(1, 5)
            ],
        }
        response = self._plan(self.api, research_id, revision)
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "candidate_limit_exceeded")

    def test_plan_rejects_unsupported_template(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        self.gateway.responses["experiment_planner"] = {
            "batch_id": "BATCH_TEMPLATE",
            "candidates": [
                scripted_candidate("EXP_TEMPLATE", template_id="not_in_whitelist")
            ],
        }
        response = self._plan(self.api, research_id, revision)
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "unsupported_template")

    def test_offline_fallback_creates_stub_candidate(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        response = self._plan(self.api, research_id, revision)
        self.assertEqual(response.status, 200)
        plan = response.payload["data"]["validation_plan"]
        self.assertTrue(plan["is_stub"])
        self.assertEqual(len(plan["candidates"]), 1)
        candidate = plan["candidates"][0]
        self.assertEqual(candidate["source"], "program")
        self.assertTrue(candidate["is_stub"])
        self.assertIn("离线模板", candidate["rationale"])
        self.assertEqual(candidate["dispute_ids"], ["DSP_TEST"])
        self.assertEqual(candidate["status"], "pending_approval")
        self.assertTrue(candidate["protocol_hash"])
        self.assertIsNotNone(candidate["utility_score"])
        self.assertEqual(
            response.payload["data"]["project"]["stage"],
            "awaiting_approval",
        )

    def test_new_batch_supersedes_unexecuted_candidate(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        first = self._plan(self.api, research_id, revision)
        first_candidate = self._candidate(first)
        first_hash = first_candidate["protocol_hash"]

        self.gateway.responses["experiment_planner"] = {
            "batch_id": "BATCH_SECOND",
            "candidates": [scripted_candidate("EXP_SECOND")],
        }
        second = self._plan(
            self.api,
            research_id,
            first.payload["revision"],
        )
        self.assertEqual(second.status, 200)

        detail = self.api.dispatch("GET", f"/api/v2/research/{research_id}")
        candidates = detail.payload["data"]["candidates"]
        by_id = {candidate["experiment_id"]: candidate for candidate in candidates}
        self.assertEqual(by_id["VALID_STUB"]["status"], "superseded")
        self.assertEqual(by_id["EXP_SECOND"]["status"], "pending_approval")

        stale_approval = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/approvals",
            {
                "expected_revision": detail.payload["revision"],
                "protocol_hash": first_hash,
            },
        )
        self.assertEqual(stale_approval.status, 409)
        self.assertEqual(stale_approval.payload["code"], "approval_hash_mismatch")

    def test_approval_hash_mismatch(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        plan = self._plan(self.api, research_id, revision)
        self._candidate(plan)
        response = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/approvals",
            {
                "expected_revision": plan.payload["revision"],
                "protocol_hash": "WRONG_HASH",
            },
        )
        self.assertEqual(response.status, 409)
        self.assertEqual(response.payload["code"], "approval_hash_mismatch")

    def test_execution_requires_approval(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        plan = self._plan(self.api, research_id, revision)
        candidate = self._candidate(plan)
        response = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/executions",
            {
                "expected_revision": plan.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
            },
        )
        self.assertEqual(response.status, 409)
        self.assertEqual(response.payload["code"], "approval_required")

    def test_execution_requires_data_source(self) -> None:
        research_id, revision = self._create_project(self.api)
        revision = self._seed_dispute(self.api, research_id, revision)
        plan = self._plan(self.api, research_id, revision)
        candidate = self._candidate(plan)
        approved = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/approvals",
            {
                "expected_revision": plan.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
                "actor_type": "human_pi",
                "rationale": "测试审批",
            },
        )
        self.assertEqual(approved.status, 200)
        response = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/executions",
            {
                "expected_revision": approved.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
            },
        )
        self.assertEqual(response.status, 409)
        self.assertEqual(response.payload["code"], "data_source_required")

    def test_calendar_alignment_builds_real_sample_plan(self) -> None:
        api = self._data_api()
        research_id, revision = self._create_project(api, title="日历样本校准")
        revision = self._seed_dispute(api, research_id, revision)
        plan = self._plan(
            api,
            research_id,
            revision,
            template_id="calendar_alignment",
            design_hints=dict(DESIGN_HINTS),
        )
        candidate = self._candidate(plan)
        plan_protocol = next(
            protocol
            for protocol in plan.payload["data"]["validation_plan"]["protocols"]
            if protocol["protocol_hash"] == candidate["protocol_hash"]
        )
        self.assertEqual(plan_protocol["sample_plan"]["template"], "calendar_alignment")
        self.assertIn(
            "sample_plan",
            plan_protocol["sample_plan"],
            "日历对齐模板必须落盘真实样本计划",
        )
        sample_plan = plan_protocol["sample_plan"]["sample_plan"]
        self.assertGreater(sample_plan["retained_count"], 0)

        approved = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/approvals",
            {
                "expected_revision": plan.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
            },
        )
        self.assertEqual(approved.status, 200)
        executed = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/executions",
            {
                "expected_revision": approved.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
            },
        )
        self.assertEqual(executed.status, 200)
        self.assertEqual(executed.payload["data"]["execution"]["status"], "executed")
        self.assertEqual(
            executed.payload["data"]["execution"]["job"]["status"],
            "completed",
        )
        self.assertEqual(
            executed.payload["data"]["disputes"][0]["validation_ids"],
            [candidate["experiment_id"]],
        )

    def test_paired_replay_executes_real_harness_within_call_budget(self) -> None:
        api = self._data_api()
        research_id, revision = self._create_project(api)
        revision = self._seed_dispute(api, research_id, revision)
        plan = self._plan(api, research_id, revision, template_id="paired_replay")
        candidate = self._candidate(plan)
        approved = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/approvals",
            {
                "expected_revision": plan.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
                "actor_type": "human_pi",
                "rationale": "人工 PI 批准受控验证",
            },
        )
        self.assertEqual(approved.status, 200)

        executed = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/executions",
            {
                "expected_revision": approved.payload["revision"],
                "protocol_hash": candidate["protocol_hash"],
            },
        )
        self.assertEqual(executed.status, 200)
        execution = executed.payload["data"]["execution"]
        self.assertEqual(execution["status"], "executed")
        self.assertEqual(execution["job"]["status"], "completed")
        self.assertTrue(execution["job"]["result_ref"].startswith(candidate["protocol_hash"][:16]))

        detail = executed.payload["data"]
        self.assertEqual(len(self.gateway.calls), 1, "完整验证闭环应只调用一次实验规划 LLM")
        planner_calls = [
            call
            for call in detail["agent_calls"]
            if call["role"] == "experiment_planner"
        ]
        execution_calls = [
            call
            for call in detail["agent_calls"]
            if call["role"] == "validation_execution"
        ]
        self.assertEqual(len(planner_calls), 1)
        self.assertEqual(len(execution_calls), 1)
        self.assertEqual(execution_calls[0]["model_id"], "program")
        self.assertEqual(detail["candidates"][0]["status"], "executed")
        self.assertEqual(
            detail["candidates"][0]["estimates"][-1]["component"],
            "EXECUTION",
        )
        self.assertEqual(detail["candidates"][0]["estimates"][-1]["value"], 1.0)
        decision_actions = [decision["action"] for decision in detail["decisions"]]
        self.assertIn("approve_protocol", decision_actions)
        self.assertIn("execute_protocol", decision_actions)

        harness_files = [
            self.root / "harness" / "results" / "round_01" / "result_unified.json",
            self.root / "harness" / "config" / "latest_protocol.json",
            self.root / "harness" / "outputs" / "latest_metrics.json",
        ]
        for path in harness_files:
            self.assertTrue(path.is_file(), f"harness 产物未落盘: {path}")


if __name__ == "__main__":
    unittest.main()
