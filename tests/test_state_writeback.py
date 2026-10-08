import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.evaluation_unified import evaluate_experiment
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.unified_schema import (
    ConclusionNextRoundSuggestion,
    EvaluationSpec,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    HypothesisNode,
    MetricDelta,
    ResearchQuestion,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    VariableBinding,
)


class StateWritebackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.repository = UnifiedStateRepository(self.project_root)

        task = ScientificTask(
            task_id="ST_100",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否改善 Vsw 预测",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(x="DeltaDec", y="Vsw", m_candidates=["By"]),
                ),
                evaluation=EvaluationSpec(primary_metric="RMSE", secondary_metrics=["Pearson_r"]),
                constraints=ScientificConstraints(),
            ),
        )
        initial_hypotheses = [
            HypothesisNode(
                hypothesis_id="H1",
                statement="DeltaDec 带来独立增量",
                level=1,
                status="active",
                support_score=0.5,
                children_ids=["H1a"],
                activation_condition="always",
            ),
            HypothesisNode(
                hypothesis_id="H1a",
                statement="DeltaDec 通过 By 中介增强预测",
                level=2,
                parent_id="H1",
                status="pending",
                support_score=0.36,
                activation_condition="父假设支持度>=0.40",
            )
        ]
        self.repository.initialize_state_skeleton(
            task,
            initial_hypotheses=initial_hypotheses,
            overwrite=True,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_apply_evaluation_writeback(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E100",
            round_id=1,
            task_id="ST_100",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            tested_hypotheses=["H1"],
        )
        result = ExperimentResult(
            experiment_id="E100",
            round_id=1,
            status="completed",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=-1.2, pearson_r=0.06),
            execution=ExecutionSummary(status="completed", duration_seconds=4.0),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {
                    "question": "DeltaDec 的增益是否跨时间段稳定",
                    "priority": "high",
                }
            ],
        )

        updater = UnifiedStateUpdater(self.repository)
        updated = updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )

        tree = updated["hypothesis_tree"]
        uncertainties = updated["uncertainties"]
        experiment_memory = updated["experiment_memory"]
        process_state = updated["process_state"]

        self.assertTrue((self.project_root / "state" / "hypothesis_tree.json").exists())
        self.assertTrue((self.project_root / "state" / "uncertainties.json").exists())
        self.assertTrue((self.project_root / "state" / "uncertainty_priority.json").exists())
        self.assertTrue((self.project_root / "state" / "experiment_memory.json").exists())
        self.assertTrue((self.project_root / "state" / "process.json").exists())
        self.assertTrue((self.project_root / "state" / "task.json").exists())
        self.assertTrue((self.project_root / "state" / "decision_log.json").exists())
        self.assertAlmostEqual(tree.node_index()["H1"].support_score, 0.58)
        # 父假设达标只解锁子假设参与资格，子节点支持度 0.36 仍进入观察期。
        self.assertEqual(tree.node_index()["H1a"].status, "observing")
        self.assertEqual(tree.node_index()["H1a"].activated_at_round, 1)
        self.assertEqual(tree.latest_update.event, "evaluation_writeback")
        self.assertEqual(uncertainties.records[0].priority, "high")
        self.assertEqual(uncertainties.priority_queue.queue[0].priority_score, 0.9)
        self.assertEqual(experiment_memory.entries[0].evaluation_path, "results/round_01/evaluation_unified.json")
        self.assertEqual(process_state.current_stage, "knowledge_memory_updated")
        self.assertEqual(process_state.steps["state_writeback"].status, "completed")

    def test_missing_next_round_advice_marks_round_incomplete(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E101",
            round_id=1,
            task_id="ST_100",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            tested_hypotheses=["H1"],
        )
        result = ExperimentResult(
            experiment_id="E101",
            round_id=1,
            status="completed",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=-1.2, pearson_r=0.06),
            execution=ExecutionSummary(status="completed", duration_seconds=4.0),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {
                    "question": "DeltaDec 的增益是否跨时间段稳定",
                    "priority": "high",
                }
            ],
        )

        updater = UnifiedStateUpdater(self.repository)
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )

        entry = self.repository.load_round_history().entries[-1]
        self.assertEqual(entry.status, "incomplete")
        self.assertFalse(entry.gating_ready)
        advice_item = next(
            (
                item
                for item in entry.closure_checklist
                if item.item_id == "next_round_advice"
            ),
            None,
        )
        self.assertIsNotNone(advice_item)
        self.assertFalse(advice_item.completed)
        self.assertTrue(advice_item.detail)

    def test_complete_next_round_advice_allows_round_completion(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E102",
            round_id=1,
            task_id="ST_100",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            tested_hypotheses=["H1"],
        )
        result = ExperimentResult(
            experiment_id="E102",
            round_id=1,
            status="completed",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=-1.2, pearson_r=0.06),
            execution=ExecutionSummary(status="completed", duration_seconds=4.0),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {
                    "question": "DeltaDec 的增益是否跨时间段稳定",
                    "priority": "high",
                }
            ],
        )
        evaluation.scientific.three_layer_conclusion.next_round_suggestion = (
            ConclusionNextRoundSuggestion(
                evidence_summary="本轮结果已写入证据摘要。",
                resolved_uncertainties_this_round=["U1"],
                unresolved_uncertainties_todo=["U2"],
                remaining_uncertainty_analysis="剩余不确定性分析已写入。",
                recommendation="continue",
                pi_decision_advice="建议进入下一轮。",
                experiment_design_advice="复用对照组，仅调整验证维度。",
                hypothesis_space_advice="保留当前假设并补时间切片。",
            )
        )

        updater = UnifiedStateUpdater(self.repository)
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )

        entry = self.repository.load_round_history().entries[-1]
        self.assertEqual(entry.status, "completed")
        self.assertTrue(entry.gating_ready)


if __name__ == "__main__":
    unittest.main()
