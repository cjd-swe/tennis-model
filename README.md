# tennis-model

ATP + WTA picks model covering three markets: **Moneyline winner**, **Total Games Over/Under**, **Set Spread**.

## Quick start

```bash
uv sync --extra dev
uv run python -m tennis_model.ingest.sackmann      # pull Sackmann data
uv run python -m tennis_model.ingest.tennis_data_uk # pull odds CSVs
uv run python -m tennis_model.ingest.merge          # merge into Parquet
uv run pytest
```

## Project structure

```
tennis_model/
  ingest/      # data pull: Sackmann + tennis-data.co.uk + merge
  features/    # feature families (surface Elo, serve/return stats, fatigue, ...)
  markets/     # market definitions (ML, Games O/U, Set Spread)
  strategies/  # baseline betting strategies
  models/      # trained market-specific models
  backtest/    # walk-forward backtester with CLV tracking

data/raw/       # gitignored — downloaded source files
data/processed/ # gitignored — merged Parquet files

notebooks/     # analysis and feasibility notebooks
reports/       # written outputs (feasibility, data gaps, live-odds viability)
tests/         # pytest suite
```

## Phases

| Phase | Goal |
|---|---|
| 0 | Foundation: data + walk-forward backtester + baseline strategies |
| 1 | Sport-knowledge feature pack (surface Elo, serve/return, fatigue, H2H, ...) |
| 2 | Market-specific models (LightGBM ML + generative Games O/U + set-spread multinomial) |
| 3 | Live odds ingestion + production pipeline |
| 4 | Pick cards UI (FastAPI + Next.js) |
| 5 | Operations: CLV monitoring, recalibration, drift detection |
