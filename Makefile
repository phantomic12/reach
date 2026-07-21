.PHONY: install test lint clean serve

install:
	pip install -e "reach-core[all]"

test:
	pytest -x --tb=short

test-cov:
	pytest -x --tb=short --cov=reach --cov-report=term

lint:
	ruff check reach-core/src reach-server tests

typecheck:
	mypy reach-core/src

serve:
	uvicorn reach-server.main:app --host 0.0.0.0 --port 8501 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist build *.egg-info

models:
	python -c "from reach.models.download import download_all; download_all()"
