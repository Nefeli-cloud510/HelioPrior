from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.research_repository import (
    ResearchProjectNotFoundError,
    ResearchRepository,
    ResearchVersionConflictError,
)
from core.research_schema import ResearchProject, ResearchTarget


def build_project(research_id: str, title: str = "测试项目") -> ResearchProject:
    return ResearchProject(
        research_id=research_id,
        title=title,
        question="日影是否能提供 IMF By 的增量预测信息？",
        target=ResearchTarget(id="imf_by_gse", unit="nT", coordinate_system="GSE"),
        horizon_days=3,
        data_mode="authorized_data",
        workflow_mode="research_review",
        entry_mode="new_question",
        stage="draft",
    )


class ResearchRepositoryTest(unittest.TestCase):
    def test_projects_are_isolated_in_distinct_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = ResearchRepository(Path(tmp))
            first = repo.create_project(build_project("RSR_A"))
            second = repo.create_project(build_project("RSR_B"))
            self.assertEqual(first.project.research_id, "RSR_A")
            self.assertEqual(second.project.research_id, "RSR_B")
            self.assertNotEqual(
                repo.revision_dir("RSR_A", 1),
                repo.revision_dir("RSR_B", 1),
            )
            second_revision = repo.load_revision("RSR_B")
            self.assertEqual(second_revision.project.title, "测试项目")

    def test_expected_revision_conflict_preserves_latest_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = ResearchRepository(Path(tmp))
            repo.create_project(build_project("RSR_A"))
            repo.mutate(
                "RSR_A",
                expected_revision=1,
                mutation=lambda state: setattr(state.project, "stage", "needs_input"),
            )
            repo.mutate(
                "RSR_A",
                expected_revision=2,
                mutation=lambda state: setattr(state.project, "stage", "preflight_running"),
            )
            with self.assertRaises(ResearchVersionConflictError):
                repo.mutate(
                    "RSR_A",
                    expected_revision=1,
                    mutation=lambda state: setattr(state.project, "stage", "draft"),
                )
            latest = repo.load_revision("RSR_A")
            self.assertEqual(latest.revision, 3)
            self.assertEqual(latest.project.stage, "preflight_running")

    def test_project_ids_cannot_escape_directory_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = ResearchRepository(Path(tmp))
            with self.assertRaises(ValueError):
                repo.project_dir("../RSR_EVIL")
            with self.assertRaises(ResearchProjectNotFoundError):
                repo.load_revision("DOES_NOT_EXIST")


if __name__ == "__main__":
    unittest.main()
