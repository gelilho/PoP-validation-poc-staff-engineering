"""Technical image quality validation using Pillow.

First line of defense — cheap, fast, no API calls.
Semantic validation (is it a receipt?) happens in the LLM step.
"""

from __future__ import annotations

from loguru import logger
from PIL import Image, ImageFilter

from pop_validation.config import (
    BLUR_THRESHOLD,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    MIN_IMAGE_RESOLUTION,
    SUPPORTED_IMAGE_FORMATS,
)
from pop_validation.image_loader import LoadedImage
from pop_validation.models import ImageQualityReport


def validate_image(image: LoadedImage) -> ImageQualityReport:
    """
    Run all technical quality checks on a loaded image.

    Args:
        image: The loaded image to validate.

    Returns:
        ImageQualityReport with results of each check.
    """
    width, height = image.pil_image.size
    file_size = len(image.data)

    logger.info(
        "Technical validation starting | {}x{} | {} bytes | mime={}",
        width,
        height,
        file_size,
        image.mime_type,
    )

    resolution_ok = check_resolution(image.pil_image)
    if not resolution_ok:
        logger.warning(
            "Resolution FAILED | {}x{} (min={}x{})",
            width,
            height,
            MIN_IMAGE_RESOLUTION,
            MIN_IMAGE_RESOLUTION,
        )

    blur_score = compute_blur_score(image.pil_image)
    is_blurry = blur_score < BLUR_THRESHOLD
    if is_blurry:
        logger.warning(
            "Blur check: image is BLURRY | score={:.1f} (threshold={})",
            blur_score,
            BLUR_THRESHOLD,
        )

    format_ok = check_format(image.mime_type)
    if not format_ok:
        logger.warning(
            "Format UNSUPPORTED | mime={} (accepted: {})",
            image.mime_type,
            ", ".join(sorted(SUPPORTED_IMAGE_FORMATS)),
        )

    file_size_ok = check_file_size(image.data)
    if not file_size_ok:
        logger.warning(
            "File size OUT OF RANGE | {} bytes (min={}, max={})",
            file_size,
            MIN_FILE_SIZE_BYTES,
            MAX_FILE_SIZE_BYTES,
        )

    logger.info(
        "Technical validation complete | resolution_ok={} | blur_score={:.1f} (blurry={}) | "
        "format_ok={} | file_size_ok={}",
        resolution_ok,
        blur_score,
        is_blurry,
        format_ok,
        file_size_ok,
    )

    return ImageQualityReport(
        resolution_ok=resolution_ok,
        blur_score=blur_score,
        is_blurry=is_blurry,
        format_ok=format_ok,
        file_size_ok=file_size_ok,
    )


def check_resolution(pil_image: Image.Image) -> bool:
    """Check that image meets minimum resolution threshold."""
    width, height = pil_image.size
    return width >= MIN_IMAGE_RESOLUTION and height >= MIN_IMAGE_RESOLUTION


def compute_blur_score(pil_image: Image.Image) -> float:
    """
    Compute a blur score using Laplacian variance.

    Higher score = sharper image. Below BLUR_THRESHOLD = blurry.
    """
    grayscale = pil_image.convert("L")
    laplacian = grayscale.filter(
        ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
            scale=1,
            offset=0,
        )
    )

    pixels: list[int] = list(laplacian.getdata(0))
    if not pixels:
        return 0.0

    mean = sum(pixels) / len(pixels)
    variance: float = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return variance


def check_format(mime_type: str) -> bool:
    """Check that the image format is supported."""
    return mime_type.lower() in SUPPORTED_IMAGE_FORMATS


def check_file_size(data: bytes) -> bool:
    """Check that the file size is within acceptable bounds."""
    return MIN_FILE_SIZE_BYTES <= len(data) <= MAX_FILE_SIZE_BYTES
