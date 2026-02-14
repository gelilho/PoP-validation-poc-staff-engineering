"""Tests for technical image quality validation — Pillow-based checks."""

from __future__ import annotations

import io

from PIL import Image

from pop_validation.image_loader import LoadedImage
from pop_validation.image_validator import (
    check_file_size,
    check_format,
    check_resolution,
    compute_blur_score,
    validate_image,
)


def _make_loaded_image(
    width: int = 800,
    height: int = 600,
    mime_type: str = "image/jpeg",
    blur: bool = False,
) -> LoadedImage:
    """Helper to create a LoadedImage for testing."""
    if blur:
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
    else:
        # Create image with high-contrast edges (not blurry)
        img = Image.new("RGB", (width, height), color="white")
        for x in range(0, width, 10):
            for y in range(height):
                img.putpixel((x, y), (0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    data = buf.getvalue()

    return LoadedImage(data=data, mime_type=mime_type, source="test", pil_image=img)


class TestCheckResolution:
    def test_good_resolution(self) -> None:
        img = Image.new("RGB", (800, 600))
        assert check_resolution(img) is True

    def test_minimum_resolution(self) -> None:
        img = Image.new("RGB", (100, 100))
        assert check_resolution(img) is True

    def test_too_small_width(self) -> None:
        img = Image.new("RGB", (50, 600))
        assert check_resolution(img) is False

    def test_too_small_height(self) -> None:
        img = Image.new("RGB", (800, 50))
        assert check_resolution(img) is False

    def test_too_small_both(self) -> None:
        img = Image.new("RGB", (10, 10))
        assert check_resolution(img) is False


class TestComputeBlurScore:
    def test_sharp_image_has_high_score(self) -> None:
        # Image with strong edges
        img = Image.new("RGB", (200, 200), color="white")
        for x in range(0, 200, 5):
            for y in range(200):
                img.putpixel((x, y), (0, 0, 0))
        score = compute_blur_score(img)
        assert score > 50.0

    def test_uniform_image_has_lower_score_than_sharp(self) -> None:
        # Solid color has lower blur score than image with edges
        uniform = Image.new("RGB", (200, 200), color=(128, 128, 128))
        sharp = Image.new("RGB", (200, 200), color="white")
        for x in range(0, 200, 5):
            for y in range(200):
                sharp.putpixel((x, y), (0, 0, 0))
        assert compute_blur_score(uniform) < compute_blur_score(sharp)

    def test_returns_float(self) -> None:
        img = Image.new("RGB", (100, 100))
        score = compute_blur_score(img)
        assert isinstance(score, float)


class TestCheckFormat:
    def test_jpeg_supported(self) -> None:
        assert check_format("image/jpeg") is True

    def test_png_supported(self) -> None:
        assert check_format("image/png") is True

    def test_webp_supported(self) -> None:
        assert check_format("image/webp") is True

    def test_tiff_supported(self) -> None:
        assert check_format("image/tiff") is True

    def test_gif_not_supported(self) -> None:
        assert check_format("image/gif") is False

    def test_pdf_not_supported(self) -> None:
        assert check_format("application/pdf") is False

    def test_case_insensitive(self) -> None:
        assert check_format("IMAGE/JPEG") is True


class TestCheckFileSize:
    def test_normal_size(self) -> None:
        data = b"x" * 50_000  # 50 KB
        assert check_file_size(data) is True

    def test_too_small(self) -> None:
        data = b"x" * 500  # 500 bytes
        assert check_file_size(data) is False

    def test_too_large(self) -> None:
        data = b"x" * (21 * 1024 * 1024)  # 21 MB
        assert check_file_size(data) is False

    def test_boundary_min(self) -> None:
        data = b"x" * 1024  # Exactly 1 KB
        assert check_file_size(data) is True

    def test_boundary_max(self) -> None:
        data = b"x" * (20 * 1024 * 1024)  # Exactly 20 MB
        assert check_file_size(data) is True


class TestValidateImage:
    def test_good_image_passes(self) -> None:
        loaded = _make_loaded_image(800, 600)
        report = validate_image(loaded)
        assert report.resolution_ok is True
        assert report.format_ok is True
        assert report.file_size_ok is True

    def test_small_image_fails_resolution(self) -> None:
        loaded = _make_loaded_image(50, 50)
        report = validate_image(loaded)
        assert report.resolution_ok is False

    def test_unsupported_format(self) -> None:
        loaded = _make_loaded_image(800, 600, mime_type="image/gif")
        report = validate_image(loaded)
        assert report.format_ok is False

    def test_blur_score_computed(self) -> None:
        loaded = _make_loaded_image(800, 600)
        report = validate_image(loaded)
        assert isinstance(report.blur_score, float)
        assert report.blur_score >= 0.0
