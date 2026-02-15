"""Result logging abstraction — Strategy pattern for injectable audit sinks.

Defines a Protocol for result loggers and a default CSV implementation.
Swap in any logger (BigQuery, Postgres, HTTP sink) with zero pipeline changes.
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Protocol

from loguru import logger

from pop_validation.models import ImageValidationResult

# ── CSV schema ───────────────────────────────────

_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "date",
    "warranty_id",
    "file_name",
    "file_path",
    "comment",
    # All ImageValidationResult fields (model order)
    "message",
    "image_category",
    "product_category",
    "language_category",
    "is_ai_generated",
    "retailer_name",
    "retailer_raw",
    "retailer_location",
    "purchase_date",
    "product_name",
    "transaction_number",
    "price",
    "currency",
    "url",
    "address",
    "country",
    "product_sku",
    "product_counts",
    "product_prices",
    "product_skus",
    "image_in_receipt",
    "confidence",
    "receipt_type",
)

_DEFAULT_CSV_PATH = Path(__file__).resolve().parents[3] / "results" / "validation_results.csv"


# ── Protocol ─────────────────────────────────────


class ResultLogger(Protocol):
    """Interface for persisting per-image validation results.

    Implement this protocol to log results to any backend:
    - CSV file (default, included)
    - BigQuery
    - PostgreSQL
    - HTTP webhook
    """

    def log(
        self,
        *,
        result: ImageValidationResult,
        warranty_id: str,
        image_url: str,
        comment: str | None = None,
    ) -> None:
        """Persist one image validation result row."""
        ...


# ── Default implementation: CSV ──────────────────


class CsvResultLogger:
    """Default implementation — appends one row per image to a local CSV file.

    Creates the file and writes headers on the first call if the file
    does not exist. All subsequent calls append rows.

    Args:
        path: Path to the CSV file. Defaults to validation_results.csv in cwd.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_CSV_PATH

    def log(
        self,
        *,
        result: ImageValidationResult,
        warranty_id: str,
        image_url: str,
        comment: str | None = None,
    ) -> None:
        """Append one result row to the CSV file. Never raises."""
        try:
            self._write_row(
                result=result,
                warranty_id=warranty_id,
                image_url=image_url,
                comment=comment,
            )
        except Exception as e:
            logger.error(
                "CSV logging failed (result NOT persisted): {}: {}",
                type(e).__name__,
                e,
            )

    def _write_row(
        self,
        *,
        result: ImageValidationResult,
        warranty_id: str,
        image_url: str,
        comment: str | None,
    ) -> None:
        """Write the row. May raise on I/O errors."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self._path.exists() and self._path.stat().st_size > 0

        with open(self._path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)

            if not file_exists:
                writer.writeheader()
                logger.info("CSV file created with headers: {}", self._path)

            row = self._build_row(
                result=result,
                warranty_id=warranty_id,
                image_url=image_url,
                comment=comment,
            )
            writer.writerow(row)
            logger.debug(
                "CSV row written | warranty_id={} | file={}",
                warranty_id,
                self._path,
            )

    @staticmethod
    def _build_row(
        *,
        result: ImageValidationResult,
        warranty_id: str,
        image_url: str,
        comment: str | None,
    ) -> dict[str, object]:
        """Build a CSV row dict from result + metadata."""
        now = datetime.datetime.now(tz=datetime.UTC)

        row: dict[str, object] = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "warranty_id": warranty_id,
            "file_name": Path(image_url).name if image_url else "",
            "file_path": image_url,
            "comment": comment or "",
        }

        # Flatten all ImageValidationResult fields
        for field_name, value in result.model_dump().items():
            if isinstance(value, (dict, list)):
                row[field_name] = json.dumps(value)
            elif isinstance(value, str):
                # Replace newlines so each CSV row stays on one line
                row[field_name] = value.replace("\n", ", ").replace("\r", "")
            else:
                row[field_name] = value

        return row
