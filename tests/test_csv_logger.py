"""Tests for result logging — Protocol + default CSV implementation."""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

from pop_validation.models import ImageValidationResult
from pop_validation.reporting.csv_logger import (
    _CSV_COLUMNS,
    CsvResultLogger,
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _valid_result() -> ImageValidationResult:
    """A result representing a fully valid receipt."""
    return ImageValidationResult(
        message="VALID_RECEIPT_FOUND",
        retailer_name="Foot Locker",
        retailer_location="NYC",
        purchase_date="2025-01-15",
        product_name="Cloud 5",
        price=149.99,
        currency="USD",
        receipt_type="official_receipt_paper",
        product_counts={"Cloud 5": 1},
        product_prices={"Cloud 5": 149.99},
    )


def _empty_result() -> ImageValidationResult:
    """A result with only the message set (failure case)."""
    return ImageValidationResult(message="RECEIPT_NOT_FOUND")


# ──────────────────────────────────────────────
# CsvResultLogger Tests
# ──────────────────────────────────────────────


class TestCsvResultLogger:
    def test_creates_file_with_headers_on_first_log(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(
            result=_valid_result(),
            warranty_id="W-001",
            image_url="https://example.com/receipt.jpg",
        )

        assert csv_path.exists()
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == list(_CSV_COLUMNS)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["warranty_id"] == "W-001"
        assert rows[0]["message"] == "VALID_RECEIPT_FOUND"
        assert rows[0]["retailer_name"] == "Foot Locker"

    def test_appends_rows_without_duplicate_headers(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(result=_valid_result(), warranty_id="W-001", image_url="a.jpg")
        csv_logger.log(
            result=_empty_result(),
            warranty_id="W-002",
            image_url="b.jpg",
            comment="Gemini API timeout",
        )

        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()

        # 1 header line + 2 data lines
        assert len(lines) == 3

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["warranty_id"] == "W-001"
        assert rows[1]["warranty_id"] == "W-002"
        assert rows[1]["comment"] == "Gemini API timeout"

    def test_error_row_has_nulls_and_comment(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(
            result=_empty_result(),
            warranty_id="W-ERR",
            image_url="/path/to/broken.jpg",
            comment="RuntimeError: 503 Service Unavailable",
        )

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        row = rows[0]
        assert row["comment"] == "RuntimeError: 503 Service Unavailable"
        assert row["message"] == "RECEIPT_NOT_FOUND"
        assert row["retailer_name"] == ""  # None → empty in CSV
        assert row["file_name"] == "broken.jpg"
        assert row["file_path"] == "/path/to/broken.jpg"

    def test_timestamp_is_human_readable(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(result=_valid_result(), warranty_id="W-TS", image_url="t.jpg")

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        timestamp = rows[0]["timestamp"]
        parsed = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        assert parsed.year >= 2025

    def test_date_column_is_date_only(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(result=_valid_result(), warranty_id="W-DT", image_url="t.jpg")

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        date_value = rows[0]["date"]
        parsed = datetime.datetime.strptime(date_value, "%Y-%m-%d")
        assert parsed.year >= 2025
        # Confirm it's date-only (no time component in the string)
        assert len(date_value) == 10

    def test_csv_write_failure_does_not_crash(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "nonexistent_dir" / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        # Should NOT raise
        csv_logger.log(result=_valid_result(), warranty_id="W-FAIL", image_url="t.jpg")

    def test_file_name_extracted_from_url(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(
            result=_valid_result(),
            warranty_id="W-URL",
            image_url="https://cdn.example.com/images/receipt_photo.jpg",
        )

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["file_name"] == "receipt_photo.jpg"

    def test_warranty_id_column_present(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(
            result=_valid_result(),
            warranty_id="W-CORR-999",
            image_url="t.jpg",
        )

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["warranty_id"] == "W-CORR-999"

    def test_complex_fields_serialized_as_json(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        csv_logger.log(result=_valid_result(), warranty_id="W-JSON", image_url="t.jpg")

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        product_counts = json.loads(rows[0]["product_counts"])
        assert product_counts == {"Cloud 5": 1}

    def test_newlines_in_fields_are_sanitized(self, tmp_path: Path) -> None:
        """Gemini can return multi-line strings (e.g. address).

        Newlines inside field values must be replaced so each CSV row
        stays on exactly one line.
        """
        csv_path = tmp_path / "results.csv"
        csv_logger = CsvResultLogger(path=csv_path)

        result = ImageValidationResult(
            message="VALID_RECEIPT_FOUND",
            address="123 Main St\nSuite 4\nNew York, NY 10001",
            retailer_location="Floor 1\nBuilding A",
        )
        csv_logger.log(result=result, warranty_id="W-NL", image_url="t.jpg")

        # Each row must be exactly one line (header + 1 data row = 2 lines)
        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["address"] == "123 Main St, Suite 4, New York, NY 10001"
        assert rows[0]["retailer_location"] == "Floor 1, Building A"

    def test_accumulates_across_instances(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "shared.csv"

        logger1 = CsvResultLogger(path=csv_path)
        logger1.log(result=_valid_result(), warranty_id="W-A", image_url="a.jpg")

        logger2 = CsvResultLogger(path=csv_path)
        logger2.log(result=_empty_result(), warranty_id="W-B", image_url="b.jpg")

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["warranty_id"] == "W-A"
        assert rows[1]["warranty_id"] == "W-B"


# ──────────────────────────────────────────────
# Protocol Structural Typing Test
# ──────────────────────────────────────────────


class TestResultLoggerProtocol:
    def test_custom_logger_satisfies_protocol(self) -> None:
        """Any class with log(result, warranty_id, image_url, comment) works."""

        class InMemoryLogger:
            def __init__(self) -> None:
                self.rows: list[dict[str, object]] = []

            def log(
                self,
                *,
                result: ImageValidationResult,
                warranty_id: str,
                image_url: str,
                comment: str | None = None,
            ) -> None:
                self.rows.append({
                    "warranty_id": warranty_id,
                    "message": result.message,
                })

        in_mem = InMemoryLogger()
        in_mem.log(
            result=_valid_result(),
            warranty_id="W-MEM",
            image_url="test.jpg",
        )
        assert in_mem.rows[0]["warranty_id"] == "W-MEM"
