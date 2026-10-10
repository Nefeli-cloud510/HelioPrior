from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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

    def test_invalid_payload_returns_422(self) -> None:
        response = self.api.dispatch("POST", "/api/v2/research", {"workflow_mode": "legacy"})
        self.assertEqual(response.status, 422)
        self.assertEqual(response.payload["code"], "invalid_workflow_mode")


if __name__ == "__main__":
    unittest.main()

