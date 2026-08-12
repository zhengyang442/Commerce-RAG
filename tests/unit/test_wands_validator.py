from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import DataValidationError
from app.ingestion.wands_models import PRODUCT_HEADER
from app.ingestion.wands_validator import REQUIRED_SOURCE_FILES, WandsValidator


def write_tiny_wands(
    root: Path,
    *,
    product_rows: list[str] | None = None,
    query_rows: list[str] | None = None,
    label_rows: list[str] | None = None,
) -> Path:
    root.mkdir()
    products = product_rows or [
        "10\tChair\tAccent Chairs\tFurniture > Chairs\tA chair\tColor: Blue\t3.0\t4.5\t2.0",
        "20\tTable\t\tFurniture > Tables\t\tMaterial: Wood\t\t\t",
    ]
    queries = query_rows or ["0\tblue chair\tchair", "2\twood table\t"]
    labels = label_rows or [
        "5\t0\t10\tPartial",
        "6\t0\t10\tExact",
        "8\t2\t20\tIrrelevant",
    ]
    (root / "product.csv").write_text(
        "\t".join(PRODUCT_HEADER) + "\n" + "\n".join(products) + "\n", encoding="utf-8"
    )
    (root / "query.csv").write_text(
        "query_id\tquery\tquery_class\n" + "\n".join(queries) + "\n", encoding="utf-8"
    )
    (root / "label.csv").write_text(
        "id\tquery_id\tproduct_id\tlabel\n" + "\n".join(labels) + "\n", encoding="utf-8"
    )
    for name in REQUIRED_SOURCE_FILES[3:]:
        (root / name).write_text("fixture", encoding="utf-8")
    return root


def fixture_validator(data_dir: Path) -> WandsValidator:
    return WandsValidator(
        data_dir,
        expected_counts={"product.csv": 2, "query.csv": 2, "label.csv": 3},
        expected_label_counts=None,
        checksum_file=None,
    )


def test_valid_tsv_preserves_missing_values_and_reports_conflicts(tmp_path: Path) -> None:
    report = fixture_validator(write_tiny_wands(tmp_path / "wands")).validate()
    products = list(fixture_validator(tmp_path / "wands").iter_products())

    assert report.product_count == 2
    assert report.query_count == 2
    assert report.label_count == 3
    assert report.unique_qrel_count == 2
    assert report.duplicate_pair_count == 1
    assert report.duplicate_annotation_count == 1
    assert report.conflict_pair_count == 1
    assert report.canonical_label_counts == {"Exact": 1, "Partial": 0, "Irrelevant": 1}
    assert products[1].product_class is None
    assert products[1].average_rating is None
    assert products[0].rating_count == 3


def test_comma_delimited_product_file_is_rejected(tmp_path: Path) -> None:
    data_dir = write_tiny_wands(tmp_path / "wands")
    (data_dir / "product.csv").write_text(
        ",".join(PRODUCT_HEADER) + "\n10,Chair,Class,Hierarchy,Description,Features,1,4,1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="TSV 表头"):
        fixture_validator(data_dir).validate()


def test_source_header_requires_category_hierarchy_with_space(tmp_path: Path) -> None:
    data_dir = write_tiny_wands(tmp_path / "wands")
    content = (data_dir / "product.csv").read_text(encoding="utf-8")
    (data_dir / "product.csv").write_text(
        content.replace("category hierarchy", "category_hierarchy", 1), encoding="utf-8"
    )

    with pytest.raises(DataValidationError, match="表头"):
        fixture_validator(data_dir).validate()


def test_non_contiguous_query_and_label_ids_are_allowed(tmp_path: Path) -> None:
    data_dir = write_tiny_wands(tmp_path / "wands")

    report = fixture_validator(data_dir).validate()

    assert report.query_count == 2
    assert report.label_count == 3


@pytest.mark.parametrize(
    ("label_rows", "message"),
    [
        (["5\t99\t10\tExact", "6\t0\t10\tPartial", "8\t2\t20\tIrrelevant"], "query_id 外键"),
        (["5\t0\t999\tExact", "6\t0\t10\tPartial", "8\t2\t20\tIrrelevant"], "product_id 外键"),
        (["5\t0\t10\tUnknown", "6\t0\t10\tPartial", "8\t2\t20\tIrrelevant"], "label 值"),
        (["5\t0\t10\tExact", "5\t0\t10\tPartial", "8\t2\t20\tIrrelevant"], "id 重复"),
    ],
)
def test_invalid_labels_are_rejected(tmp_path: Path, label_rows: list[str], message: str) -> None:
    data_dir = write_tiny_wands(tmp_path / "wands", label_rows=label_rows)

    with pytest.raises(DataValidationError, match=message):
        fixture_validator(data_dir).validate()


def test_duplicate_product_id_is_rejected(tmp_path: Path) -> None:
    rows = [
        "10\tChair\tClass\tHierarchy\tDescription\tFeatures\t1\t4\t1",
        "10\tTable\tClass\tHierarchy\tDescription\tFeatures\t1\t4\t1",
    ]
    data_dir = write_tiny_wands(tmp_path / "wands", product_rows=rows)

    with pytest.raises(DataValidationError, match="product_id 重复"):
        fixture_validator(data_dir).validate()


def test_invalid_numeric_field_is_rejected(tmp_path: Path) -> None:
    rows = [
        "10\tChair\tClass\tHierarchy\tDescription\tFeatures\t1.5\t4\t1",
        "20\tTable\tClass\tHierarchy\tDescription\tFeatures\t1\t4\t1",
    ]
    data_dir = write_tiny_wands(tmp_path / "wands", product_rows=rows)

    with pytest.raises(DataValidationError, match="rating_count"):
        fixture_validator(data_dir).validate()


def test_wrong_field_count_is_rejected_without_exposing_row(tmp_path: Path) -> None:
    rows = [
        "10\tChair\tClass\tHierarchy\tDescription\tFeatures\t1\t4",
        "20\tTable\tClass\tHierarchy\tDescription\tFeatures\t1\t4\t1",
    ]
    data_dir = write_tiny_wands(tmp_path / "wands", product_rows=rows)

    with pytest.raises(DataValidationError, match="字段数量"):
        fixture_validator(data_dir).validate()
