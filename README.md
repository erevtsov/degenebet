# degenebet

A quantitative, data-driven NFL betting system built on free data sources.

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # fill in SHARPAPI_KEY
```

## Usage

```bash
uv run degenebet fetch schedules
uv run degenebet fetch player-stats
uv run degenebet fetch team-stats
uv run degenebet fetch rosters
uv run degenebet fetch odds
```

## Development

```bash
uv run pytest
uv run ruff check src/ tests/
uv run mypy
```
