# degenebet

A quantitative, data-driven NFL betting system built on free data sources.

## Setup

```bash
uv sync --extra dev
cp .env.example .env   # fill in SHARPAPI_KEY
```

Install `just` (not a Python dependency): `brew install just` (macOS) or [see installation guide](https://github.com/casey/just#installation)

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
just check   # runs ruff check, ruff format --check, mypy, and pytest — the same gate CI runs
```

Individual steps, if you want to run just one:

```bash
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy
```
