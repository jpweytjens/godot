# Cycling ETA — Project Conventions

## File paths
- Always use `pathlib.Path`, never bare strings for file paths.
- Discover GPX files with `Path("data").glob("*.gpx")` — do not hardcode filenames.

## GPX parsing
- `read_gpx(path)` returns a raw DataFrame: lat, lon, elevation, timestamp, speed_ms, plus any available extensions (hr, watts, cad, atemp).
- Distance is added via `.pipe()` functions: `add_haversine_distance` or `add_integrated_distance`.
- `add_smooth_speed` derives `speed_kmh` via 5s rolling mean of `speed_ms`.
- Keeping distance computation separate lets us compare methods and mirrors the Kotlin target.

## GPX data format
- 1-second recording interval, `<gpxtpx:speed>` (m/s) present in all trackpoints.
- Also available: watts, hr, cad, atemp — ignore for now, useful for future versions.
- Test file: `data/criquielion.gpx` (~18,777 points, 2025-08-23 ride).

## Type hints
- Use type hints for function signatures.
- Do not add `# type: ignore` or contort code just to satisfy type errors from upstream library stubs (e.g. pandas, matplotlib). Those are stub quality issues, not our problem.

## ETA estimation architecture

An ETA estimator is assembled from three independent ingredients:

### 1. Pause strategy (`godot/pause.py`)
How to handle rider stops in the ETA prediction.
- `NoPause` — predict total time (includes stops). Time basis: **T**.
- `WallClockPause` — predict moving time (NaN during pauses, add observed pause time to ETA). Time basis: **M**.

### 2. ETA estimator (`godot/estimators.py`)
The speed/TTG prediction engine. Layers build on each other:

| Level | Class | What it adds |
|-------|-------|-------------|
| 0 | `AvgSpeedEstimator`, `AdaptiveLerpSpeedEstimator` | No gradient awareness — flat speed model |
| 1 | `GradientPriorEstimator`, `PhysicsGradientPriorEstimator` | Gradient prior: speed = `v_flat * ratio(gradient)`. Empirical uses lookup table, physics solves constant-power cubic. |
| 2 | `AdaptivePhysicsEstimator`, `AdaptiveGradientPriorEstimator` | + slow EWMA correction on `actual/predicted` ratio (v_flat calibration) + global fast EWMA for short-term residual |
| 3 | `BinnedAdaptiveEstimator` | + per-gradient-bin fast EWMA (replaces global fast). Each 1% gradient bin learns independently. |
| 4 | `TrustedBinnedAdaptiveEstimator` | + trust ramp (blend toward 1.0 until enough observations) + clamping ([0.5, 1.5]) on per-bin corrections |
| 5 | `IntegralPhysicsEstimator`, `SplitIntegralPhysicsEstimator` | Realistic physics ratios (headwind, climb effort, descent confidence, freewheel cap) + integral correction (cumulative predicted/actual time). Split variant uses separate climb/descent integrals. No EWMA, no per-bin corrections. |

Oracle variants (`OracleTrustedBinnedEstimator`) patch v_flat with the true flat-section average to isolate the correction layers from v_flat estimation error.

### 3. v_flat estimator (`VFlatEstimator` subclasses in `godot/estimators.py`)
How to estimate flat-ground speed during a ride. Used by level 2+ estimators.
- `StaticVFlat` — returns `v_flat_init` unchanged
- `FlatSpeedVFlat` — cumulative mean of flat-section speeds
- `WeightedGainVFlat` — back-derives `v_flat_obs = v_actual / ratio(gradient)` with cos²(θ) weighting
- `EwmaLockVFlat` — flat-only EWMA, locks when stable (converge-and-freeze)
- `MedianLockVFlat` — expanding median of back-derived v_flat, locks when stable
- `PriorFreeVFlat` — expanding mean of `v_actual / ratio(gradient)`, skip period scaled to ride time
- `PriorFreeEwmaVFlat` — like PriorFreeVFlat but EWMA after skip (tracks drift)
- `OracleVFlat` — true whole-ride flat average (cheating baseline)

### Backtests

- `backtest.py` — ETA accuracy. Combines estimator + pause strategy. Metrics: MAE, RMSE, MPE, MAPE (wall-clock and moving-time). Run: `uv run python backtest.py data/gpx/*.gpx --no-plots`
- `vflat_backtest.py` — v_flat convergence speed. Tests `VFlatEstimator` strategies in isolation. Metrics: v_flat at time checkpoints (2/5/10/20/30 min), MAE/bias vs ground truth. Run: `uv run python vflat_backtest.py`

## Estimator naming convention
Backtest estimator keys follow: `{level}_{T|M}_{prior}_{name}`
- **level**: 0-4, encoding complexity / correction layers (see table above)
- **T|M**: Total time (`NoPause`) or Moving time (`WallClockPause`)
- **prior**: `global` (fixed v_flat), `oracle` (true flat avg), or estimator name (e.g. `wgain`)
- **name**: descriptive correction layer (e.g. `physics_trusted`, `empirical_binned`)

## Context
This is a Python prototype for a gradient-aware cycling ETA estimator.
The production target is a Kotlin extension — keep algorithm logic portable.

## Writing style (READMEs, docs, prose)
- Friendly, clear, concise. Not dry academic, but technically precise.
- Limit jargon. When a term is unavoidable, define it the first time in plain language.
- Be very sparse with **bold**. Reserve it for genuine warnings or the one or two key terms in a section.
- Prefer *italic* for emphasis and for introducing a name or concept.
- Short sentences over long ones. Cut filler ("in order to", "it should be noted that").
- Active voice. Address the reader directly when natural.
- Code, file paths, identifiers, and CLI flags in `backticks`.
- Avoid emoji and decorative headings unless explicitly asked.
