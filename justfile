# Run the full verification suite: lint, format check, type-check, tests.
check:
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run mypy
    uv run pytest
