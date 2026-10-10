from __future__ import annotations

import threading
import time
import unittest
from datetime import datetime
from typing import Any, Callable

from core.evidence_builder import build_evidence_packet, normalize_prediction_rows, parse_source_content
from core.llm_gateway import LLMGateway
from core.review_orchestrator import (
    ArbitrationOutput,
    DualReviewResult,
    ReviewOpinionOutput,
    ReviewOrchestrator,
    ReviewObservation,
    make_disputes,
    review_request_hash,
)
from core.research_schema import ClaimVersion, EvidencePacket, ResearchProject, ResearchTarget


CSV = """forecast_origin,target_time,horizon_days,arm,y_true,y_pred,split
2024-01-01,2024-01-04,3,base,-1.5,-1.3,train
2024-01-01,2024-01-04,3,aug,-1.5,-1.6,train
2024-01-02,2024-01-05,3,base,2.0,2.2,test
2024-01-02,2024-01-05,3,aug,2.0,1.9,test
"""


def make_packet() -> EvidencePacket:
    parsed = parse_source_content("prediction_csv", CSV, "paired.csv")
    normalized = normalize_prediction_rows(parsed)
    packet = build_evidence_packet(
        parsed=normalized,
        protocol_hash_value="PRT_HASH",
        data_manifest_hash="DATA_HASH",
        run_id="RUN_T4",
        claim_ids=["CLM_001"],
        target_id="imf_by_gse",
        unit="nT",
        horizon_days=3,
        feature_columns=["shadow_shift"],
        base_arm="base",
        aug_arm="aug",
        primary_artifact_id="ART_TEST",
        primary_artifact_sha256="ab" * 32,
    )
    packet.packet_id = "EVP_PACKET_1"
    return packet


def make_claim() -> ClaimVersion:
    now = datetime.now()
    return ClaimVersion(
        claim_id="CLM_001",
        version=3,
        statement="日影对未来 IMF By 具有增量预测信息",
        created_at=now,
        updated_at=now,
    )


def make_project() -> ResearchProject:
    return ResearchProject(
        research_id="RSR_TEST",
        revision=4,
        title="双路复核测试",
        question="测试问题",
        target=ResearchTarget(id="imf_by_gse", unit="nT", coordinate_system="GSE"),
        horizon_days=3,
        data_mode="synthetic_fixture",
        stage="evidence_ready",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


def opinion_output(*, packet: EvidencePacket, stance: str = "support") -> ReviewOpinionOutput:
    metric_keys = sorted(packet.metrics)
    evidence_ids = [item.evidence_id for item in packet.evidence_items]
    return ReviewOpinionOutput(
        claim_id="CLM_001",
        packet_hash=packet.packet_hash,
        stance=stance,
        observations=[
            ReviewObservation(
                observation_id="OBS_1",
                issue_code="PAIRED_SAMPLES",
                text="共同样本配对检查结果正常。",
                evidence_refs=evidence_ids[:1],
                metric_refs=metric_keys[:1],
                consequence="当前结果可以解释为相同样本上的增量比较。",
            )
        ],
        evidence_refs=evidence_ids[:1],
        metric_refs=metric_keys[:1],
        missing_evidence=[],
        validation_proposals=[],
        scope_notes=["仅适用于当前证据包。"],
        is_stub=False,
    )


class FakeGateway(LLMGateway):
    """Gateway whose calls never touch the network."""

    def __init__(self) -> None:
        super().__init__(client=None, allow_fallback=True)
        self.responses: dict[str, Any] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.hook: Callable[[str, str, str], Any] | None = None
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def generate_structured(self, *, system_prompt, user_prompt, response_model, fallback_factory=None, payload_fixer=None, temperature=None, image_paths=None):
        self._track_enter()
        try:
            role = "central_arbitrator" if "中央协调者" in system_prompt else (
                "scientific_questioner" if "科学质询者" in system_prompt else "scientific_interpreter"
            )
            key = role
            with self._lock:
                self.calls.append((role, user_prompt, system_prompt))
            response = None
            if self.hook is not None:
                response = self.hook(role, user_prompt, system_prompt)
            if response is None:
                response = self.responses.get(key)
            if response is None:
                if fallback_factory is not None and self.allow_fallback:
                    return self._coerce_response(response_model, fallback_factory())
                raise RuntimeError(f"no scripted response for {role}")
            return self._coerce_response(response_model, response)
        finally:
            self._track_exit()

    def _track_enter(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    def _track_exit(self) -> None:
        with self._lock:
            self.in_flight -= 1


class ReviewOrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = make_packet()
        self.claim = make_claim()
        self.project = make_project()

    def scripted_gateway(self) -> FakeGateway:
        gateway = FakeGateway()
        gateway.responses["scientific_questioner"] = opinion_output(packet=self.packet, stance="support")
        gateway.responses["scientific_interpreter"] = opinion_output(packet=self.packet, stance="support")
        return gateway

    def run_review(self, gateway: FakeGateway, *, max_in_flight: int = 2) -> DualReviewResult:
        orchestrator = ReviewOrchestrator(gateway=gateway, max_in_flight=max_in_flight)
        return orchestrator.run_dual_review(
            packet=self.packet,
            claim=self.claim,
            project=self.project,
            frozen_revision=self.project.revision,
        )

    def test_both_roles_succeed_without_arbitration(self) -> None:
        gateway = self.scripted_gateway()
        result = self.run_review(gateway)
        self.assertEqual(result.status, "completed")
        self.assertEqual([opinion.role for opinion in result.opinions], ["scientific_questioner", "scientific_interpreter"])
        self.assertTrue(all(opinion.status == "completed" for opinion in result.opinions))
        self.assertEqual(result.disputes, [])
        self.assertIsNone(result.arbitration)
        self.assertFalse(result.is_stub)
        self.assertEqual(len(result.agent_calls), 2)

    def test_one_role_fails_then_retry_succeeds(self) -> None:
        gateway = self.scripted_gateway()
        state = {"failure_count": 0}

        def hook(role: str, user_prompt: str, system_prompt: str) -> Any:
            if role == "scientific_interpreter" and state["failure_count"] == 0:
                state["failure_count"] += 1
                raise RuntimeError("scripted first failure")
            return gateway.responses.get(role)

        gateway.hook = hook
        orchestrator = ReviewOrchestrator(gateway=gateway, max_in_flight=2)
        result = orchestrator.run_dual_review(
            packet=self.packet,
            claim=self.claim,
            project=self.project,
            frozen_revision=self.project.revision,
            retry_roles=("scientific_interpreter",),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(state["failure_count"], 1)
        interpreter_calls = [call for call in result.agent_calls if call.role == "scientific_interpreter"]
        self.assertEqual(len(interpreter_calls), 2)
        self.assertEqual([call.attempt for call in interpreter_calls], [1, 2])
        self.assertEqual([call.status for call in interpreter_calls], ["failed", "succeeded"])
        self.assertEqual(interpreter_calls[1].status, "succeeded")

    def test_one_role_fails_after_all_retries_gives_partial_with_no_consensus(self) -> None:
        gateway = self.scripted_gateway()

        def hook(role: str, user_prompt: str, system_prompt: str) -> Any:
            if role == "scientific_interpreter":
                raise RuntimeError("always fails")
            return gateway.responses.get(role)

        gateway.hook = hook
        orchestrator = ReviewOrchestrator(gateway=gateway, max_in_flight=2)
        result = orchestrator.run_dual_review(
            packet=self.packet,
            claim=self.claim,
            project=self.project,
            frozen_revision=self.project.revision,
            retry_roles=("scientific_interpreter",),
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.opinions), 1)
        self.assertEqual(len(result.disputes), 0)
        self.assertIsNone(result.arbitration)
        self.assertIn("不生成共识", result.message or "")

    def test_invalid_metric_ref_is_rejected(self) -> None:
        gateway = FakeGateway()
        bad = opinion_output(packet=self.packet, stance="support")
        bad.metric_refs = ["NOT_A_METRIC"]
        gateway.responses["scientific_questioner"] = bad
        gateway.responses["scientific_interpreter"] = opinion_output(packet=self.packet, stance="support")
        result = self.run_review(gateway)
        # The faulty role is retried once and never reaches consensus from an
        # invalid reference set; the other independent route stays intact.
        self.assertEqual(result.status, "partial")
        questioner_calls = [call for call in result.agent_calls if call.role == "scientific_questioner"]
        self.assertEqual(len(questioner_calls), 2)
        self.assertEqual([call.attempt for call in questioner_calls], [1, 2])
        self.assertEqual([call.status for call in questioner_calls], ["failed", "failed"])
        self.assertIn("invalid_review_refs", questioner_calls[-1].error_code or "")
        self.assertIn("NOT_A_METRIC", questioner_calls[-1].error_message or "")
        self.assertEqual([opinion.role for opinion in result.opinions], ["scientific_interpreter"])
        self.assertEqual(
            [
                opinion
                for opinion in result.opinions
                if opinion.role == "scientific_questioner" and "NOT_A_METRIC" in opinion.metric_refs
            ],
            [],
        )

    def test_conflicting_stances_trigger_single_arbitration_call(self) -> None:
        packet = self.packet
        gateway = FakeGateway()
        gateway.responses["scientific_questioner"] = opinion_output(packet=packet, stance="support")
        gateway.responses["scientific_interpreter"] = opinion_output(packet=packet, stance="challenge")
        arbitration = ArbitrationOutput(
            issue_summary="两路判断冲突",
            recommended_action="在相同样本上执行一次最小验证。",
            required_validation="paired_replay",
            is_stub=False,
        )
        gateway.responses["central_arbitrator"] = arbitration
        result = self.run_review(gateway)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.disputes), 1)
        arbitrator_calls = [call for call in result.agent_calls if call.role == "central_arbitrator"]
        self.assertEqual(len(arbitrator_calls), 1)
        self.assertEqual(result.arbitration, arbitration)

    def test_matching_stances_do_not_trigger_arbitration(self) -> None:
        gateway = self.scripted_gateway()
        result = self.run_review(gateway)
        self.assertEqual(result.status, "completed")
        self.assertEqual([call.role for call in result.agent_calls if call.role == "central_arbitrator"], [])

    def test_max_in_flight_is_bounded(self) -> None:
        gateway = self.scripted_gateway()
        original = gateway.hook

        def hook(role: str, user_prompt: str, system_prompt: str) -> Any:
            time.sleep(0.08)
            return gateway.responses.get(role)

        gateway.hook = hook
        result = self.run_review(gateway, max_in_flight=1)
        self.assertEqual(result.status, "completed")
        self.assertEqual(gateway.max_in_flight, 1)

    def test_two_first_round_calls_are_concurrent(self) -> None:
        gateway = self.scripted_gateway()
        barrier = threading.Barrier(2)
        original = gateway.hook

        def hook(role: str, user_prompt: str, system_prompt: str) -> Any:
            barrier.wait(timeout=5)
            return gateway.responses.get(role)

        gateway.hook = hook
        result = self.run_review(gateway, max_in_flight=2)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(gateway.calls), 2)

    def test_stub_output_is_visibly_labeled(self) -> None:
        gateway = FakeGateway()
        orchestrator = ReviewOrchestrator(gateway=gateway, max_in_flight=2)
        result = orchestrator.run_dual_review(
            packet=self.packet,
            claim=self.claim,
            project=self.project,
            frozen_revision=self.project.revision,
        )
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.is_stub)
        self.assertTrue(all(opinion.is_stub for opinion in result.opinions))
        self.assertTrue(any("未调用真实模型" in text for opinion in result.opinions for text in opinion.observations))

    def test_request_hash_changes_when_packet_or_revision_changes(self) -> None:
        base = review_request_hash(
            packet_hash="PACKET_A",
            claim_id="CLM_001",
            claim_version=1,
            project_stage="evidence_ready",
            frozen_revision=4,
        )
        changed_packet = review_request_hash(
            packet_hash="PACKET_B",
            claim_id="CLM_001",
            claim_version=1,
            project_stage="evidence_ready",
            frozen_revision=4,
        )
        changed_revision = review_request_hash(
            packet_hash="PACKET_A",
            claim_id="CLM_001",
            claim_version=1,
            project_stage="evidence_ready",
            frozen_revision=5,
        )
        self.assertNotEqual(base, changed_packet)
        self.assertNotEqual(base, changed_revision)

    def test_make_disputes_uses_program_facts_not_majority(self) -> None:
        gateway = self.scripted_gateway()
        result = self.run_review(gateway)
        self.assertEqual(result.status, "completed")
        disputes = make_disputes(
            result.opinions,
            review_id="RVR_DISPUTE_1",
            claim_id=result.claim_id,
            packet=self.packet,
            now=datetime.now(),
        )
        self.assertEqual(disputes, [])


if __name__ == "__main__":
    unittest.main()
