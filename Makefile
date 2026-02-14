.PHONY: install test lint format typecheck coverage all clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/ --strict

coverage:
	pytest tests/ -v --cov=pop_validation --cov-report=term-missing --cov-report=html

clean:
	rm -rf .coverage htmlcov/ dist/ build/ .mypy_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: clean install format lint typecheck test coverage
