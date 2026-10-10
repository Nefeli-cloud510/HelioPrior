from __future__ import annotations

import unittest

from core.evidence_builder import (
    build_evidence_packet,
    normalize_prediction_rows,
    parse_source_content,
)


PAIRED_CSV = """forecast_origin,target_time,horizon_days,arm,y_true,y_pred,split
2024-01-01,2024-01-04,3,base,-1.5,-1.3,train
2024-01-01,2024-01-04,3,aug,-1.5,-1.6,train
2024-01-02,2024-01-05,3,base,2.0,2.2,test
2024-01-02,2024-01-05,3,aug,2.0,1.9,test
"""


def build_packet(
    *,
    content: str = PAIRED_CSV,
    kind: str = "prediction_csv",
    target_id: str = "imf_by_gse",
    feature_columns: list[str] | None = None,
    base_arm: str | None = None,
    aug_arm: str | None = None,
    imported_metrics: dict | None = None,
):
    source = parse_source_content(kind, content, "artifact.csv")
    parsed = normalize_prediction_rows(source, target_column=None)
    return build_evidence_packet(
        parsed=parsed,
        protocol_hash_value="hash-protocol",
        data_manifest_hash="hash-data",
        run_id="RUN_TEST",
        claim_ids=["CLM_TEST"],
        target_id=target_id,
        unit="nT",
        horizon_days=3,
        feature_columns=feature_columns,
        base_arm=base_arm,
        aug_arm=aug_arm,
        primary_artifact_id="ART_TEST",
        primary_artifact_sha256="artifact-hash",
        imported_metrics=imported_metrics,
    )


class EvidenceBuilderTest(unittest.TestCase):
    def test_single_arm_without_time_marks_external_checks_unknown(self) -> None:
        packet = build_packet(
            content="y_true,y_pred\n1,1.1\n2,2.2\n",
        )
        status_by_code = {check.code: check.status for check in packet.checks}
        self.assertEqual(status_by_code["TIME_TARGET"], "unknown")
        self.assertEqual(status_by_code["SPLIT_BOUNDARY"], "unknown")
        self.assertEqual(status_by_code["FIT_SCOPE"], "unknown")
        self.assertEqual(status_by_code["DATA_VERSION"], "pass")
        self.assertEqual(status_by_code["PAIRED_SAMPLES"], "not_applicable")
        self.assertIn("当前证据包只有一组方案", packet.limitations[0])

    def test_paired_rows_pass_calendar_and_common_sample_checks(self) -> None:
        packet = build_packet(feature_columns=["shadow_shift", "imf_by_gse_history"])
        status_by_code = {check.code: check.status for check in packet.checks}
        self.assertEqual(status_by_code["TIME_TARGET"], "pass")
        self.assertEqual(status_by_code["PAIRED_SAMPLES"], "pass")
        self.assertEqual(status_by_code["TARGET_EQUAL"], "pass")
        self.assertAlmostEqual(packet.metrics["gain_rmse"].value, 0.1)
        self.assertEqual(packet.metrics["common_sample_count"].value, 2)

    def test_target_alias_in_feature_columns_is_blocking_failure(self) -> None:
        packet = build_packet(feature_columns=["by_gse", "shadow_shift"])
        target_alias = next(check for check in packet.checks if check.code == "TARGET_ALIAS")
        self.assertEqual(target_alias.status, "fail")
        self.assertTrue(target_alias.blocking)

    def test_imported_metrics_are_registered_without_recomputation(self) -> None:
        packet = build_packet(
            kind="evidence_json",
            imported_metrics={
                "rmse.base": 1.2,
                "rmse.aug": 1.0,
                "mae.base": 0.9,
                "mae.aug": 0.7,
                "r.base": 0.4,
                "r.aug": 0.5,
            },
        )
        self.assertEqual(packet.metrics["rmse.base"].value, 1.2)
        self.assertEqual(packet.metrics["rmse.aug"].value, 1.0)
        self.assertEqual(packet.metrics["computed_elsewhere"].value, 1)
        self.assertIn("导入模式", packet.limitations[0])

    def test_invalid_json_is_reported_as_parse_error(self) -> None:
        source = parse_source_content("evidence_json", "{not-json", "artifact.json")
        self.assertTrue(source.parse_errors)


if __name__ == "__main__":
    unittest.main()
