#!/usr/bin/env python3
"""Run the PoP Validation Pipeline.

Usage:
    python run_pipeline.py                                     # All images in samples/
    python run_pipeline.py path/to/receipt.jpg                 # One specific image
    python run_pipeline.py path/to/folder/                     # All images in a folder
    python run_pipeline.py img1.jpg img2.png img3.webp         # Multiple specific images
    python run_pipeline.py --warranty-id W-123 samples/        # With warranty ID
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
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

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
DEFAULT_SAMPLES_DIR = Path("samples")


def discover_images(path: Path) -> list[str]:
    """Find all supported image files in a directory (sorted by name)."""
    if not path.is_dir():
        logger.error("Directory not found: {}", path)
        return []
    images = sorted(
        str(p)
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info("Discovered {} image(s) in {}/", len(images), path)
    return images


def resolve_inputs(raw_paths: list[str]) -> list[str]:
    """Resolve CLI arguments into a flat list of image file paths.

    Each argument can be:
    - A file path  -> used directly
    - A directory  -> expanded to all images inside it
    """
    images: list[str] = []
    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            images.extend(discover_images(p))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(str(p))
        else:
            logger.warning("Skipping unsupported or missing path: {}", raw)
    return images


def main() -> None:
    """Run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the PoP Validation Pipeline on receipt images.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image file(s) or folder(s) to validate. Defaults to samples/ folder.",
    )
    parser.add_argument(
        "--warranty-id",
        default=None,
        help="Warranty ID to associate with these images (stored in CSV). "
        "Defaults to an auto-generated UUID.",
    )
    args = parser.parse_args()

    # Resolve image paths
    raw_paths: list[str] = args.images if args.images else [str(DEFAULT_SAMPLES_DIR)]
    image_urls = resolve_inputs(raw_paths)

    if not image_urls:
        logger.error(
            "No images found. Add images to samples/ or pass paths as arguments.",
        )
        sys.exit(1)

    # Warranty ID: user-provided or auto-generated
    warranty_id: str = args.warranty_id or f"run-{uuid.uuid4().hex[:8]}"

    logger.info("=" * 70)
    logger.info(
        "PIPELINE RUN | {} image(s) | warranty_id={}",
        len(image_urls),
        warranty_id,
    )
    logger.info("=" * 70)

    pipeline = PopValidationPipeline()

    for idx, image_url in enumerate(image_urls, 1):
        logger.info("")
        logger.info("-" * 50)
        logger.info("[{}/{}] {}", idx, len(image_urls), image_url)
        logger.info("-" * 50)

        start = time.perf_counter()
        result = pipeline.validate(
            ValidationRequest(
                warranty_id=warranty_id,
                image_urls=[image_url],
            ),
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
    logger.info("PIPELINE RUN COMPLETE | warranty_id={}", warranty_id)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
