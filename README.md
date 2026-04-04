# Godot

![Godot](godot_logo.jpg)

A Python framework for developing and backtesting cycling ETA estimators, with a focus on forward-looking, gradient-aware arrival time prediction.

## Goal

- A framework for implementing, comparing, and backtesting ETA estimators against recorded ride data (GPX/FIT).
- A gradient-aware ETA algorithm that uses upcoming elevation profile to produce more accurate time-to-arrival estimates on hilly routes.

## Project structure

```
eta/            Core library
  gpx.py        GPX parsing and distance computation
  fit.py        FIT file parsing
  ride.py       Ride data pipeline
  estimators.py ETA estimator implementations
  segmentation.py  Route segmentation by gradient
  benchmark.py  Estimator accuracy metrics
  plot.py       Altair-based visualisation helpers
  report.py     HTML report generation
backtest.py     CLI entry point for running backtests
tests/          pytest suite
data/           GPX/FIT ride files
```

## Quickstart

```bash
uv sync
uv run python backtest.py
```

## Requirements

Python >= 3.12, managed with [uv](https://docs.astral.sh/uv/).
