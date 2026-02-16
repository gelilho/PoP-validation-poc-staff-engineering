# PoP Validation PoC — Proof of Purchase Validation

Staff-engineered Proof of Purchase validation system using a **multi-agent architecture**.
Google Gemini 2.5 Flash handles OCR extraction and image quality assessment.
Each agent owns a single responsibility — divide and conquer.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      OrchestratorAgent       │
                    │  Receives ValidationRequest  │
                    │  Coordinates all 5 agents    │
                    │  Aggregates results           │
                    └──────────┬──────────────────┘
                               │  For each image:
                               ▼
                    ┌─────────────────────┐
                    │  ImageLoaderAgent    │  Step 1: Load image (free)
                    │  File / URL / HTTP  │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │  QualityAgent       │  Steps 2+3: Tech check (Pillow, free)
                    │  Pillow + Gemini #1 │  + Quality assessment (Gemini call #1)
                    └─────────┬───────────┘
                              ▼ (only if quality passes)
                    ┌─────────────────────┐
                    │  ExtractionAgent    │  Step 4: Field extraction (Gemini call #2)
                    │  Gemini #2 + OCR    │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │  ValidationAgent    │  Steps 5+6: Apply 9 business rules
                    │  Pure logic         │  + Build ImageValidationResult
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │  ReportingAgent     │  Step 7: Audit trail
                    │  CSV / BQ / Postgres│  (CSV today, BigQuery tomorrow)
                    └─────────────────────┘
```

### Why 5 agents?

| Agent | Steps | What it does | Cost |
|-------|-------|-------------|------|
| **ImageLoaderAgent** | 1 | Load image from file path, URL, or HTTP | Free |
| **QualityAgent** | 2 + 3 | Pillow tech checks + Gemini quality assessment | 1 Gemini call |
| **ExtractionAgent** | 4 | Extract structured receipt fields via Gemini OCR | 1 Gemini call |
| **ValidationAgent** | 5 + 6 | Apply 9-rule chain of responsibility + build result | Free |
| **ReportingAgent** | 7 | Persist to CSV audit trail | Free |

### Pipeline per image (7 steps)

```
Step 1: Load image           → ImageLoaderAgent    (free)
Step 2: Technical validation  → QualityAgent        (free — Pillow)
Step 3: Quality assessment    → QualityAgent        (Gemini call #1)
Step 4: Field extraction      → ExtractionAgent     (Gemini call #2)
Step 5: Apply business rules  → ValidationAgent     (free — pure logic)
Step 6: Build result          → ValidationAgent     (free)
Step 7: Log result            → ReportingAgent      (free — CSV append)
```

### Business Rules (Chain of Responsibility)

| # | Rule | Verdict on trigger |
|---|------|--------------------|
| 1 | AI-generated image detected | `FRAUDULENT_IMAGE_DETECTED` |
| 2 | Low image quality | `LOW_IMAGE_QUALITY` |
| 3 | No receipt found | `RECEIPT_NOT_FOUND` |
| 4 | Invalid receipt type | `RECEIPT_FORMAT_NOT_VALID` |
| 5 | Unofficial retailer | `UNOFFICIAL_RETAILER` |
| 6 | Retailer not official on date (stub) | _(deferred)_ |
| 7 | No brand products found | `RECEIPT_DOES_NOT_CONTAIN_BRAND_FOOTWEAR` |
| 8 | Missing required fields | `RECEIPT_MISSING_REQUIRED_INFORMATION` |
| 9 | All pass | `VALID_RECEIPT_FOUND` |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Set up your Gemini API key

```bash
echo 'GEMINI_API_KEY=YOUR_API_KEY_HERE' > .env.local
```

Get a Gemini API key from: https://aistudio.google.com/apikey

### 3. Run the pipeline

```bash
# Default: all images in samples/
python run_pipeline.py

# Specific image
python run_pipeline.py path/to/receipt.jpg

# Specific folder
python run_pipeline.py path/to/receipts/

# With warranty ID (stored in CSV audit trail)
python run_pipeline.py --warranty-id W-123

# Multiple images + warranty ID
python run_pipeline.py --warranty-id W-456 img1.jpg img2.png
```

### 4. Use as a library

```python
from pop_validation import PopValidationPipeline
from pop_validation.models import ValidationRequest

pipeline = PopValidationPipeline()

response = pipeline.validate(ValidationRequest(
    warranty_id="W-123",
    image_urls=["samples/receipt.jpg"],
))

print(response.pop_valid)       # True / False
print(response.uncertain)       # True if missing required fields
for r in response.pop_validation_results:
    print(r.message)            # e.g. "VALID_RECEIPT_FOUND"
    print(r.retailer_name)      # e.g. "Foot Locker"
    print(r.price)              # e.g. 149.99
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | **Yes** | -- | Google Gemini API key (also accepts `GOOGLE_API_KEY`) |
| `GEMINI_MODEL_NAME` | No | `gemini-2.5-flash` | Gemini model to use |

---

## Project Structure

```
src/pop_validation/
    __init__.py                  # Public API exports
    config.py                    # Settings + constants
    models.py                    # Pydantic data models
    pipeline.py                  # Thin facade over OrchestratorAgent

    agents/                      # Multi-agent architecture
        base.py                  # BaseAgent Protocol + AgentResult
        image_loader_agent.py    # Step 1: Load image
        quality_agent.py         # Steps 2+3: Tech + Gemini quality
        extraction_agent.py      # Step 4: Gemini field extraction
        validation_agent.py      # Steps 5+6: Rules + build result
        reporting_agent.py       # Step 7: Audit trail
        orchestrator.py          # Coordinates all 5 agents

    client/                      # Gemini API wrapper
        gemini_client.py

    imaging/                     # Image loading + technical validation
        loader.py
        validator.py

    extraction/                  # Field parsing utilities
        field_utils.py

    validation/                  # Business rules (chain of responsibility)
        rules.py

    catalog/                     # Product catalog (Strategy pattern)
        provider.py
        data/products.json

    prompts/                     # LLM prompts
        quality_prompt.py
        extraction_prompt.py

    reporting/                   # Result logging (Strategy pattern)
        csv_logger.py            # ResultLogger Protocol + CsvResultLogger

tests/                           # 189 tests, 97% coverage
results/                         # CSV audit trail (auto-generated)
samples/                         # Receipt images for testing
```

---

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Multi-Agent** | `agents/` | Divide and conquer — each agent owns one step |
| **Protocol + DI** | `BaseAgent`, `ResultLogger`, `ProductCatalogProvider` | Swappable implementations (CSV today, BigQuery tomorrow) |
| **Chain of Responsibility** | `validation/rules.py` | 9 rules applied in strict order, first trigger short-circuits |
| **Facade** | `pipeline.py` | Backward-compatible entry point over the orchestrator |
| **Strategy** | `catalog/provider.py`, `reporting/csv_logger.py` | Injectable backends via constructor |
| **3-layer cost control** | Pipeline flow | Pillow (free) -> Gemini #1 -> Gemini #2 (only if #1 passes) |

---

## Cost Control

The pipeline minimizes Gemini API calls:

1. **Technical validation first** (Pillow, free) -- rejects bad resolution/format/blur before any LLM call
2. **Quality assessment** (Gemini call #1) -- rejects non-receipts, unreadable, AI-generated images
3. **Field extraction** (Gemini call #2) -- only runs if quality assessment passes

If an image fails technical validation: **0 Gemini calls**.
If an image fails quality assessment: **1 Gemini call** (not 2).
Full extraction path: **2 Gemini calls**.

---

## CSV Audit Trail

Every processed image produces one row in `results/validation_results.csv`:

| Column | Description |
|--------|-------------|
| `timestamp` | Human-readable datetime (e.g. `2025-01-15 14:30:22`) |
| `date` | Date only (e.g. `2025-01-15`) |
| `warranty_id` | Correlates rows to pipeline runs |
| `file_name` | Image filename |
| `file_path` | Full path or URL |
| `comment` | Empty on success, error description on failure |
| `message` | Validation verdict (e.g. `VALID_RECEIPT_FOUND`) |
| `retailer_name`, `price`, `currency`, ... | All extracted fields |

The `ResultLogger` Protocol allows swapping CSV for BigQuery, PostgreSQL, or any other backend with zero pipeline changes.

---

## Development

```bash
# Run tests (189 tests, 97% coverage)
python -m pytest tests/ --tb=short -q

# Linter
python -m ruff check src/ tests/

# Type checking (strict mode)
python -m mypy src/ --strict

# Full quality gate
python -m ruff check src/ tests/ && python -m mypy src/ --strict && python -m pytest tests/ --tb=short -q
```

