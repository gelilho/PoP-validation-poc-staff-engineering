# PoP Validation PoC — Proof of Purchase Validation

Staff-engineered rewrite of the [Insert your brand] Proof of Purchase validation system.
Uses Google Gemini 2.5 Flash for OCR extraction and image quality assessment.

---

## Quick Start

### 1. Install dependencies

```bash
# From the project root (PoP-validation-poc-staff-engineering/)
pip install -e ".[dev]"

# Or using Make:
make install
```

### 2. Set up your Gemini API key

```bash
# One-liner setup — just paste your key:
echo 'GEMINI_API_KEY=YOUR_API_KEY_HERE' > .env.local

# Or copy the example and edit:
cp .env.example .env.local
```

Get a Gemini API key from: https://aistudio.google.com/apikey

### 3. Run it

```python
from pop_validation import PopValidationPipeline
from pop_validation.models import ValidationRequest

pipeline = PopValidationPipeline()

result = pipeline.validate(ValidationRequest(
    warranty_id="W-123",
    image_urls=["samples/shoe_receipt.webp"],
))

print(result.model_dump_json(indent=2))
```

### 4. Run the smoke test

```bash
# Run ALL images in samples/ (auto-discovers every image file)
python3 smoke_test.py

# Run a single specific image
python3 smoke_test.py samples/shoe_receipt.webp

# Run multiple specific images
python3 smoke_test.py samples/skechers.jpg "samples/ticket adidas.jpg"
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | **Yes** | — | Your Google Gemini API key (also accepts `GOOGLE_API_KEY`) |
| `GEMINI_MODEL_NAME` | No | `gemini-2.5-flash` | Gemini model to use |

---

## Development

```bash
# Run tests (with coverage)
make test

# Run linter
make lint

# Auto-format code
make format

# Type checking (mypy strict)
make typecheck

# Full quality pipeline
make all
```

---

## Project Structure

```
src/pop_validation/
  __init__.py            # Facade re-export
  config.py              # Settings + constants
  models.py              # Pydantic data models
  image_loader.py        # Load images (file/http/local)
  image_validator.py     # Technical quality checks (Pillow)
  ocr_extractor.py       # Gemini OCR extraction (Strategy)
  validation_rules.py    # Business rules (Chain of Responsibility)
  pipeline.py            # Orchestrator (Facade)
  prompts/
    quality_prompt.py    # LLM prompt: image quality assessment
    extraction_prompt.py # LLM prompt: receipt field extraction

tests/                   # 135 tests, 94% coverage
samples/                 # Real receipt images for testing
```

---

## Cost Control

The pipeline is designed to **minimize Gemini API calls**:

1. **Technical validation first** (Pillow, zero cost) — rejects bad resolution/format/size before any LLM call
2. **Quality assessment** (1 LLM call) — rejects non-receipts, unreadable, AI-generated images
3. **Field extraction** (1 LLM call) — only runs if quality assessment passes

If an image fails technical validation, **zero Gemini calls** are made.
If an image fails quality assessment, **only 1 call** is made (not 2).

---

## Logging

All execution is traced with structured `loguru` logs. Gemini API calls show:
- Call type (quality vs extraction)
- Timing in milliseconds
- Response preview
- Cost savings when calls are skipped
