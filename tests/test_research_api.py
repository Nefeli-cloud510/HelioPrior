from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.llm_gateway import LLMGateway
from core.research_api import ResearchV2Api
from core.research_repository import ResearchRepository


def create_payload(title: str = "日影观测与 IMF By 增量预测验证", question: str = "加入日影后是否改善指定条件下的 By 预测") -> dict:
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


CSV = """forecast_origin,target_time,horizon_days,arm,y_true,y_pred,split
2024-01-01,2024-01-04,3,base,-1.5,-1.3,train
2024-01-01,2024-01-04,3,aug,-1.5,-1.6,train
2024-01-02,2024-01-05,3,base,2.0,2.2,test
2024-01-02,2024-01-05,3,aug,2.0,1.9,test
"""


class ScriptedReviewGateway(LLMGateway):
    """Offline gateway that never touches the network."""

    def __init__(self) -> None:
        super().__init__(client=None, allow_fallback=True)
        self.responses: dict[str, dict] = {}
        self.fail_roles: set[str] = set()
        self.calls: list[str] = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model, fallback_factory=None, payload_fixer=None, temperature=None, image_paths=None):
        if "中央协调者" in system_prompt:
            role = "central_arbitrator"
        elif "科学质询者" in system_prompt:
            role = "scientific_questioner"
        else:
            role = "scientific_interpreter"
        self.calls.append(role)
        if role in self.fail_roles:
            raise RuntimeError(f"scripted failure: {role}")
        response = self.responses.get(role)
        if response is not None:
            return response_model.model_validate(response)
        if fallback_factory is not None and self.allow_fallback:
            return response_model.model_validate(fallback_factory())
        raise RuntimeError(f"no scripted response for {role}")


def opinion_for_packet(packet: dict, claim_id: str, stance: str) -> dict:
    metric_refs = list(packet.get("metrics") or {})
    evidence_ids = [item["evidence_id"] for item in packet.get("evidence_items") or []]
    evidence_ref = evidence_ids[:1]
    return {
        "claim_id": claim_id,
        "packet_hash": packet["packet_hash"],
        "stance": stance,
        "observations": [
            {
                "observation_id": "OBS_API_1",
                "issue_code": "PAIRED_SAMPLES",
                "text": "共同样本配对检查结果符合要求。",
                "evidence_refs": evidence_ref,
                "metric_refs": metric_refs[:1],
            }
        ],
        "evidence_refs": evidence_ref,
        "metric_refs": metric_refs[:1],
        "missing_evidence": [],
        "validation_proposals": [],
        "scope_notes": ["仅适用于当前证据包。"],
        "is_stub": False,
    }


class ResearchV2ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.api = ResearchV2Api(ResearchRepository(Path(self.tmp.name)))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_create_list_and_get(self) -> None:
        created = self.api.dispatch("POST", "/api/v2/research", create_payload())
        self.assertEqual(created.status, 201)
        research_id = created.payload["research_id"]
        self.assertEqual(created.payload["revision"], 1)
        self.assertIn("request_id", created.payload)

        listed = self.api.dispatch("GET", "/api/v2/research")
        self.assertEqual(listed.status, 200)
        self.assertEqual([item["research_id"] for item in listed.payload["data"]], [research_id])

        detail = self.api.dispatch("GET", f"/api/v2/research/{research_id}")
        self.assertEqual(detail.status, 200)
        self.assertEqual(detail.payload["data"]["project"]["question"], create_payload()["question"])

    def test_idempotency_reuses_project_and_rejects_content_change(self) -> None:
        payload = create_payload()
        first = self.api.dispatch("POST", "/api/v2/research", payload, idempotency_key="IDEM-001")
        self.assertEqual(first.status, 201)
        second = self.api.dispatch("POST", "/api/v2/research", payload, idempotency_key="IDEM-001")
        self.assertEqual(second.status, 201)
        self.assertEqual(second.payload["research_id"], first.payload["research_id"])

        conflict = self.api.dispatch(
            "POST",
            "/api/v2/research",
            {**payload, "title": "另一个标题"},
            idempotency_key="IDEM-001",
        )
        self.assertEqual(conflict.status, 409)
        self.assertEqual(conflict.payload["code"], "idempotency_conflict")

    def test_import_claim_creates_initial_claim_version(self) -> None:
        payload = {
            **create_payload("复核已有结论"),
            "entry_mode": "import_claim",
            "claim_statement": "已有的日影模型报告显示加入日影后 RMSE 改善",
            "claim_scope": {"sample_condition": "TS>=9", "baseline_protocol_id": "PRT_OLD"},
        }
        created = self.api.dispatch("POST", "/api/v2/research", payload)
        self.assertEqual(created.status, 201)
        research_id = created.payload["research_id"]
        detail = self.api.dispatch("GET", f"/api/v2/research/{research_id}")
        self.assertEqual(detail.payload["data"]["claim_count"], 1)

    def test_later_endpoints_return_explicit_not_implemented(self) -> None:
        created = self.api.dispatch("POST", "/api/v2/research", create_payload())
        research_id = created.payload["research_id"]
        response = self.api.dispatch("POST", f"/api/v2/research/{research_id}/plans", {"expected_revision": 1})
        self.assertEqual(response.status, 501)
        self.assertEqual(response.payload["code"], "not_implemented")

    def test_upload_artifact_and_preflight_build_evidence_packet(self) -> None:
        created = self.api.dispatch("POST", "/api/v2/research", create_payload())
        research_id = created.payload["research_id"]
        csv = """forecast_origin,target_time,horizon_days,arm,y_true,y_pred,split
2024-01-01,2024-01-04,3,base,-1.5,-1.3,train
2024-01-01,2024-01-04,3,aug,-1.5,-1.6,train
2024-01-02,2024-01-05,3,base,2.0,2.2,test
2024-01-02,2024-01-05,3,aug,2.0,1.9,test
"""
        uploaded = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/artifacts",
            {
                "expected_revision": 1,
                "file_name": "paired_predictions.csv",
                "kind": "prediction_csv",
                "content": csv,
            },
        )
        self.assertEqual(uploaded.status, 201)
        artifact = uploaded.payload["data"]["artifact"]
        self.assertTrue(artifact["parsed"])
        self.assertTrue(artifact["sha256"])

        preflighted = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/preflight",
            {
                "expected_revision": 2,
                "artifact_id": artifact["artifact_id"],
                "feature_columns": ["shadow_shift"],
            },
        )
        self.assertEqual(preflighted.status, 200)
        preflight = preflighted.payload["data"]["preflight"]
        packet = preflight["packet"]
        self.assertTrue(packet["packet_id"])
        self.assertAlmostEqual(packet["metrics"]["gain_rmse"]["value"], 0.1, places=6)
        self.assertEqual(preflighted.payload["data"]["evidence_packet_count"], 1)

    def test_preflight_without_artifact_returns_actionable_error(self) -> None:
        created = self.api.dispatch("POST", "/api/v2/research", create_payload())
        research_id = created.payload["research_id"]
        response = self.api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/preflight",
            {"expected_revision": 1},
        )
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "artifact_required")

    def _prepare_review_project(self, api: ResearchV2Api) -> tuple[str, str, dict, int]:
        created = api.dispatch(
            "POST",
            "/api/v2/research",
            {
                **create_payload("复核日影增量结论"),
                "entry_mode": "import_claim",
                "claim_statement": "加入日影后 RMSE 相比基准组改善",
            },
        )
        research_id = created.payload["research_id"]
        claim_id = created.payload["data"]["claims"][0]["claim_id"]
        uploaded = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/artifacts",
            {
                "expected_revision": created.payload["revision"],
                "file_name": "paired_predictions.csv",
                "kind": "prediction_csv",
                "content": CSV,
            },
        )
        artifact = uploaded.payload["data"]["artifact"]
        preflighted = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/preflight",
            {
                "expected_revision": uploaded.payload["data"]["revision"],
                "artifact_id": artifact["artifact_id"],
                "feature_columns": ["shadow_shift"],
                "claim_ids": [claim_id],
            },
        )
        packet = preflighted.payload["data"]["preflight"]["packet"]
        return research_id, claim_id, packet, preflighted.payload["data"]["revision"]

    def test_start_review_runs_two_independent_opinions(self) -> None:
        gateway = ScriptedReviewGateway()
        api = ResearchV2Api(self.api.service.repository, gateway=gateway)
        research_id, claim_id, packet, revision = self._prepare_review_project(api)
        gateway.responses["scientific_questioner"] = opinion_for_packet(packet, claim_id, "support")
        gateway.responses["scientific_interpreter"] = opinion_for_packet(packet, claim_id, "support")

        started = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/reviews",
            {"expected_revision": revision},
        )
        self.assertEqual(started.status, 200)
        review = started.payload["data"]["review"]
        self.assertEqual(review["status"], "completed")
        self.assertEqual(len(review["opinions"]), 2)
        self.assertFalse(review["is_stub"])
        self.assertEqual(started.payload["data"]["project"]["stage"], "awaiting_decision")
        self.assertEqual(started.payload["data"]["open_dispute_count"], 0)
        succeeded = [
            call
            for call in review["agent_calls"]
            if call["status"] == "succeeded"
        ]
        self.assertEqual(len(succeeded), 2)

    def test_stance_conflict_creates_dispute_and_arbitration(self) -> None:
        gateway = ScriptedReviewGateway()
        api = ResearchV2Api(self.api.service.repository, gateway=gateway)
        research_id, claim_id, packet, revision = self._prepare_review_project(api)
        gateway.responses["scientific_questioner"] = opinion_for_packet(packet, claim_id, "support")
        gateway.responses["scientific_interpreter"] = opinion_for_packet(packet, claim_id, "challenge")
        gateway.responses["central_arbitrator"] = {
            "issue_summary": "两路审读对增量证据判断直接相反。",
            "recommended_action": "在相同样本上执行一次最小验证。",
            "required_validation": "paired_replay",
            "is_stub": False,
        }

        started = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/reviews",
            {"expected_revision": revision},
        )
        self.assertEqual(started.status, 200)
        review = started.payload["data"]["review"]
        self.assertEqual(len(review["disputes"]), 1)
        self.assertEqual(review["disputes"][0]["issue_code"], "STANCE_CONFLICT")
        self.assertEqual(len(review["arbitrations"]), 1)
        self.assertIn("central_arbitrator", gateway.calls)
        self.assertEqual(
            len([call for call in review["agent_calls"] if call["role"] == "central_arbitrator"]),
            1,
        )

    def test_retry_failed_review_role_reuses_same_review(self) -> None:
        gateway = ScriptedReviewGateway()
        api = ResearchV2Api(self.api.service.repository, gateway=gateway)
        research_id, claim_id, packet, revision = self._prepare_review_project(api)
        gateway.responses["scientific_questioner"] = opinion_for_packet(packet, claim_id, "support")
        gateway.responses["scientific_interpreter"] = opinion_for_packet(packet, claim_id, "support")
        gateway.fail_roles = {"scientific_interpreter"}

        started = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/reviews",
            {"expected_revision": revision},
        )
        self.assertEqual(started.status, 200)
        review = started.payload["data"]["review"]
        self.assertEqual(review["status"], "partial")
        self.assertEqual(review["stage"], "partial_review")
        self.assertEqual(len(review["opinions"]), 1)
        review_id = review["review_id"]

        gateway.fail_roles = set()
        retried = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/reviews/{review_id}/retry",
            {
                "expected_revision": started.payload["data"]["revision"],
                "retry_roles": ["scientific_interpreter"],
            },
        )
        self.assertEqual(retried.status, 200)
        retry_review = retried.payload["data"]["review"]
        self.assertEqual(retry_review["review_id"], review_id)
        self.assertEqual(retry_review["status"], "completed")
        self.assertEqual(len(retry_review["opinions"]), 2)
        interpreter_calls = [
            call
            for call in retry_review["agent_calls"]
            if call["role"] == "scientific_interpreter"
        ]
        self.assertEqual(interpreter_calls[-1]["attempt"], 2)
        self.assertEqual(interpreter_calls[-1]["status"], "succeeded")
        self.assertEqual(retried.payload["data"]["project"]["stage"], "awaiting_decision")

    def test_review_without_evidence_returns_actionable_error(self) -> None:
        gateway = ScriptedReviewGateway()
        api = ResearchV2Api(self.api.service.repository, gateway=gateway)
        created = api.dispatch("POST", "/api/v2/research", create_payload())
        research_id = created.payload["research_id"]
        response = api.dispatch(
            "POST",
            f"/api/v2/research/{research_id}/reviews",
            {"expected_revision": 1},
        )
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "evidence_packet_required")

    def test_invalid_payload_returns_422(self) -> None:
        response = self.api.dispatch("POST", "/api/v2/research", {"workflow_mode": "legacy"})
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "invalid_workflow_mode")


if __name__ == "__main__":
    unittest.main()
