from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.core.config import DEFAULT_RAW_DATA_DIR, PROJECT_ROOT
from app.core.errors import DataValidationError
from app.core.models import Product
from app.ingestion.wands_models import (
    LABEL_GRADES,
    LABEL_HEADER,
    PRODUCT_HEADER,
    QUERY_HEADER,
    LabelRecord,
    QueryRecord,
    parse_label,
    parse_product,
    parse_query,
)

EXPECTED_COUNTS = {"product.csv": 42_994, "query.csv": 480, "label.csv": 233_448}
EXPECTED_LABEL_COUNTS = {"Exact": 25_614, "Partial": 146_633, "Irrelevant": 61_201}
REQUIRED_SOURCE_FILES = (
    "product.csv",
    "query.csv",
    "label.csv",
    "LICENSE",
    "README.md",
    "Product Search Relevance Annotation Guidelines.pdf",
)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    product_count: int
    query_count: int
    label_count: int
    label_counts: dict[str, int]
    missing_product_fields: dict[str, int]
    unique_qrel_count: int
    duplicate_pair_count: int
    duplicate_annotation_count: int
    conflict_pair_count: int
    canonical_label_counts: dict[str, int]
    checksums_verified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "product_count": self.product_count,
            "query_count": self.query_count,
            "label_count": self.label_count,
            "label_counts": self.label_counts,
            "missing_product_fields": self.missing_product_fields,
            "unique_qrel_count": self.unique_qrel_count,
            "duplicate_pair_count": self.duplicate_pair_count,
            "duplicate_annotation_count": self.duplicate_annotation_count,
            "conflict_pair_count": self.conflict_pair_count,
            "canonical_label_counts": self.canonical_label_counts,
            "qrel_aggregation": "max_gain",
            "checksums_verified": self.checksums_verified,
        }


class WandsValidator:
    def __init__(
        self,
        data_dir: Path = DEFAULT_RAW_DATA_DIR,
        *,
        expected_counts: dict[str, int] | None = EXPECTED_COUNTS,
        expected_label_counts: dict[str, int] | None = EXPECTED_LABEL_COUNTS,
        checksum_file: Path | None = PROJECT_ROOT / "data" / "WANDS_SHA256SUMS",
    ) -> None:
        self.data_dir = data_dir
        self.expected_counts = expected_counts
        self.expected_label_counts = expected_label_counts
        self.checksum_file = checksum_file

    def validate(self) -> ValidationReport:
        self._validate_required_files()
        checksums_verified = self._validate_checksums() if self.checksum_file else False

        product_ids: set[int] = set()
        query_ids: set[int] = set()
        label_ids: set[int] = set()
        missing_fields: Counter[str] = Counter()

        product_count = 0
        for line_number, row in self._rows("product.csv", PRODUCT_HEADER):
            product = parse_product(row, location=f"product.csv:{line_number}")
            self._ensure_unique(product.product_id, product_ids, "product_id", line_number)
            product_count += 1
            for field in (
                "product_class",
                "category_hierarchy",
                "product_description",
                "rating_count",
                "average_rating",
                "review_count",
            ):
                if getattr(product, field) is None:
                    missing_fields[field] += 1
        self._assert_count("product.csv", product_count)

        query_count = 0
        for line_number, row in self._rows("query.csv", QUERY_HEADER):
            query = parse_query(row, location=f"query.csv:{line_number}")
            self._ensure_unique(query.query_id, query_ids, "query_id", line_number)
            query_count += 1
        self._assert_count("query.csv", query_count)

        label_counts: Counter[str] = Counter()
        pair_grades: dict[tuple[int, int], set[int]] = {}
        pair_occurrences: Counter[tuple[int, int]] = Counter()
        label_count = 0
        for line_number, row in self._rows("label.csv", LABEL_HEADER):
            label = parse_label(row, location=f"label.csv:{line_number}")
            self._ensure_unique(label.label_id, label_ids, "id", line_number, "label.csv")
            if label.query_id not in query_ids:
                raise DataValidationError(f"label.csv:{line_number}: query_id 外键不存在")
            if label.product_id not in product_ids:
                raise DataValidationError(f"label.csv:{line_number}: product_id 外键不存在")
            pair = (label.query_id, label.product_id)
            pair_occurrences[pair] += 1
            pair_grades.setdefault(pair, set()).add(LABEL_GRADES[label.label])
            label_counts[label.label] += 1
            label_count += 1
        self._assert_count("label.csv", label_count)
        if (
            self.expected_label_counts is not None
            and dict(label_counts) != self.expected_label_counts
        ):
            raise DataValidationError("label.csv: 标签分布与固定 WANDS 版本不一致")

        duplicate_pair_count = sum(count > 1 for count in pair_occurrences.values())
        conflict_pair_count = sum(len(grades) > 1 for grades in pair_grades.values())
        canonical_counts: Counter[str] = Counter()
        grade_labels = {grade: label for label, grade in LABEL_GRADES.items()}
        for grades in pair_grades.values():
            canonical_counts[grade_labels[max(grades)]] += 1

        return ValidationReport(
            product_count=product_count,
            query_count=query_count,
            label_count=label_count,
            label_counts=_ordered_labels(label_counts),
            missing_product_fields=dict(missing_fields),
            unique_qrel_count=len(pair_grades),
            duplicate_pair_count=duplicate_pair_count,
            duplicate_annotation_count=label_count - len(pair_grades),
            conflict_pair_count=conflict_pair_count,
            canonical_label_counts=_ordered_labels(canonical_counts),
            checksums_verified=checksums_verified,
        )

    def iter_products(self) -> Iterator[Product]:
        for line_number, row in self._rows("product.csv", PRODUCT_HEADER):
            yield parse_product(row, location=f"product.csv:{line_number}")

    def iter_queries(self) -> Iterator[QueryRecord]:
        for line_number, row in self._rows("query.csv", QUERY_HEADER):
            yield parse_query(row, location=f"query.csv:{line_number}")

    def iter_labels(self) -> Iterator[LabelRecord]:
        for line_number, row in self._rows("label.csv", LABEL_HEADER):
            yield parse_label(row, location=f"label.csv:{line_number}")

    def _validate_required_files(self) -> None:
        missing = [name for name in REQUIRED_SOURCE_FILES if not (self.data_dir / name).is_file()]
        if missing:
            raise DataValidationError(f"WANDS 必需文件缺失: {', '.join(missing)}")

    def _validate_checksums(self) -> bool:
        assert self.checksum_file is not None
        if not self.checksum_file.is_file():
            raise DataValidationError("WANDS SHA-256 清单不存在")
        expected: dict[str, str] = {}
        for line_number, line in enumerate(
            self.checksum_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            digest, separator, filename = line.partition("  ")
            if not separator or len(digest) != 64 or not filename:
                raise DataValidationError(f"SHA-256 清单:{line_number}: 格式错误")
            expected[filename] = digest
        if set(expected) != set(REQUIRED_SOURCE_FILES):
            raise DataValidationError("SHA-256 清单文件集合不完整")
        for filename, digest in expected.items():
            actual = _sha256(self.data_dir / filename)
            if actual != digest:
                raise DataValidationError(f"{filename}: SHA-256 校验失败")
        return True

    def _rows(
        self, filename: str, expected_header: Sequence[str]
    ) -> Iterator[tuple[int, dict[str, str]]]:
        path = self.data_dir / filename
        try:
            source = path.open(encoding="utf-8", newline="")
        except (OSError, UnicodeError) as error:
            raise DataValidationError(f"{filename}: 无法按 UTF-8 读取") from error
        with source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != tuple(expected_header):
                raise DataValidationError(f"{filename}: TSV 表头不匹配")
            try:
                for line_number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise DataValidationError(f"{filename}:{line_number}: 字段数量不匹配")
                    yield line_number, row  # type: ignore[misc]
            except csv.Error as error:
                raise DataValidationError(f"{filename}:{reader.line_num}: TSV 解析失败") from error

    def _assert_count(self, filename: str, actual: int) -> None:
        if self.expected_counts is None:
            return
        expected = self.expected_counts[filename]
        if actual != expected:
            raise DataValidationError(f"{filename}: 期望 {expected} 条记录，实际 {actual}")

    @staticmethod
    def _ensure_unique(
        value: int,
        seen: set[int],
        field: str,
        line_number: int,
        filename: str = "product.csv",
    ) -> None:
        if value in seen:
            raise DataValidationError(f"{filename}:{line_number}: {field} 重复")
        seen.add(value)


def _ordered_labels(counts: Counter[str]) -> dict[str, int]:
    return {label: counts[label] for label in ("Exact", "Partial", "Irrelevant")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
