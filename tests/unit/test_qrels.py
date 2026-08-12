from __future__ import annotations

from pathlib import Path

from app.evaluation.qrels import load_qrels
from app.ingestion.wands_validator import WandsValidator
from tests.unit.test_wands_validator import write_tiny_wands


def test_qrels_use_max_gain_and_report_conflicts(tmp_path: Path) -> None:
    data_dir = write_tiny_wands(tmp_path / "wands")
    validator = WandsValidator(
        data_dir,
        expected_counts={"product.csv": 2, "query.csv": 2, "label.csv": 3},
        expected_label_counts=None,
        checksum_file=None,
    )

    qrels = load_qrels(validator)

    assert qrels.by_query[0][10] == 2
    assert qrels.raw_count == 3
    assert qrels.unique_count == 2
    assert qrels.duplicate_pair_count == 1
    assert qrels.conflict_pair_count == 1
