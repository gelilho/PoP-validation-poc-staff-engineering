#!/usr/bin/env python3
"""Smoke test — validates real receipt images against the PoP pipeline.

Usage:
    python smoke_test.py                              # Run ALL images in samples/
    python smoke_test.py samples/shoe_receipt.webp    # Run specific image(s)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

from pop_validation import PopValidationPipeline
from pop_validation.models import ValidationRequest

# Configure loguru for readable console output
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)

SAMPLES_DIR = Path("samples")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}


def discover_samples() -> list[str]:
    """Auto-discover all image files in the samples/ folder (sorted by name)."""
    if not SAMPLES_DIR.is_dir():
        logger.error("samples/ directory not found — nothing to test")
        return []
    images = sorted(
        str(p)
        for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info("Discovered {} image(s) in samples/", len(images))
    return images


def main() -> None:
    """Run the smoke test."""
    # Use CLI args if provided, otherwise auto-discover from samples/
    image_urls = sys.argv[1:] if len(sys.argv) > 1 else discover_samples()

    if not image_urls:
        logger.error("No images to test. Add images to samples/ or pass paths as args.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("SMOKE TEST | {} image(s) to validate", len(image_urls))
    logger.info("=" * 70)

    pipeline = PopValidationPipeline()

    for idx, image_url in enumerate(image_urls, 1):
        logger.info("")
        logger.info("-" * 50)
        logger.info("[{}/{}] Testing: {}", idx, len(image_urls), image_url)
        logger.info("-" * 50)

        start = time.perf_counter()
        result = pipeline.validate(
            ValidationRequest(
                warranty_id="smoke-test",
                image_urls=[image_url],
            )
        )
        elapsed = (time.perf_counter() - start) * 1000

        r = result.pop_validation_results[0] if result.pop_validation_results else None
        logger.info("")
        logger.info("RESULT for {}", image_url)
        logger.info("  pop_valid        = {}", result.pop_valid)
        logger.info("  uncertain        = {}", result.uncertain)
        if r:
            for field_name, value in r.model_dump().items():
                if value is not None:
                    logger.info("  {:<18} = {}", field_name, value)
        logger.info("  time             = {:.0f}ms", elapsed)
        logger.info("")

    logger.info("=" * 70)
    logger.info("SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
