# Godot

<img src="godot_logo.jpg" alt="Godot" width="180">

A framework for building and benchmarking gradient-aware cycling ETA estimators, and the reference estimator that gives the project its name: *Godot* — *Gradient Optimized Distance Over Time*.

## Overview

Most cycling computers compute ETA as `remaining_distance / total_average_speed`. It's simple, and it fails in two annoying ways:

1. *Pause blowup.* A long café stop drags the average toward zero, and ETA balloons until you start riding again.
2. *Gradient blindness.* Your average so far tells you very little about a ride that finishes with a 10% climb — or a long descent.

Godot is a small Python codebase for designing estimators that handle both, plus a backtesting harness to compare them on real GPX rides. It also ships *Godot the estimator*: a gradient-aware predictor that gets to a moving-time MAE of about 9.3 minutes on our test corpus.

> A note on names: the repo and the estimator share a name. Below, "Godot" means the estimator unless it clearly means the framework.

## The idea

### Predict moving speed, not total speed

Instead of total average speed, we predict *moving* average speed `v̄_m`. Then:

```
RRT = remaining_distance / v̄_m
ETA = now + RRT + planned_pause_time
```

Past pauses are already baked into `now` (it's just wallclock), and future pauses are an explicit input — your guess at how long you'll stop, not the estimator's. Mathematically it's equivalent to the total-average formulation when there are no pauses, but it sidesteps the blowup completely.

### Make speed gradient-aware

We model speed at a given gradient as a multiplicative deviation from flat-ground speed:

```
v(g) = v_flat · r(g)
```

`r(g)` is a *speed-ratio curve*: at every gradient `g`, what fraction of your flat-ground speed do you ride at? `r(0) = 1` by definition. The shape comes from a force-balance model with a few extras:

- *Mechanical*: gravity, aero drag, headwind, rolling resistance.
- *Behavioural*:
  - `climb_effort` — riders push above their flat-ground power on climbs. We model this as a fraction of the extra power needed to hold constant speed.
  - `descent_confidence` — riders only commit a fraction of the available descent speed before braking. Captures rider comfort, which physics alone won't predict.
  - `freewheel_cap` — on steep descents, terminal velocity (the point where gravity equals drag with zero pedalling) caps the speed regardless of confidence.

Once mass, CdA, Crr, and the behavioural coefficients are fixed, `r(g)` is fully determined by a single parameter: `v_flat`.

> 📈 *TODO: plot of `r(g)` for a representative rider, with and without behavioural terms, over `g ∈ [-15%, +15%]`.*

## The Godot estimator

Given the upcoming route, Godot does four things:

1. *Decimate* the elevation profile into constant-gradient segments using Visvalingam–Whyatt simplification. The whole rest of the route lives as a list of `(length, gradient)` pairs.
2. *Predict TTG* by walking the upcoming segments from the rider's current position and accumulating `length / v(g)`. The current segment is partially consumed; the rest are taken whole. This is essentially a dot product between remaining distances and per-segment speeds.
3. *Maintain two corrections* `c_up` and `c_dn` — the cumulative ratio of predicted to actual elapsed time, computed separately over climb-or-flat samples and descent samples.
4. *Apply the corrections* multiplicatively at prediction time: `v(g) = v_flat · r(g) · c_up` for climbs and flats, `· c_dn` for descents.

The initial `v_flat` is never updated. The corrections do all the online learning, and because `v_flat · c_up` is just a rescaled `v_flat`, you can equivalently think of Godot as maintaining two effective flat speeds — one for going up, one for coming down — sharing a single fixed `r(g)`.

The split is the interesting part. Rider intent on climbs (effort, pacing) drifts from physics in a different direction than rider intent on descents (comfort, braking). One global correction would average those out; two corrections let each side converge to its own bias.

### Algorithm 1 — TTG prediction

```
Input:  current distance d, total distance D,
        upcoming segments {(s_start_j, s_end_j, g_j)},
        v_flat, ratio curve r(·), corrections c_up, c_dn
Output: TTG(d)

1: TTG ← 0
2: j₀ ← index of segment containing d
3: for j = j₀ … N−1:
4:     start  ← max(s_start_j, d) if j = j₀ else s_start_j   ▷ partial first segment
5:     length ← s_end_j − start
6:     if length ≤ 0: continue
7:     corr ← c_up if g_j ≥ 0 else c_dn
8:     v_j  ← v_flat · r(g_j) · corr
9:     TTG  ← TTG + length / v_j
10: return TTG
```

### Algorithm 2 — online correction update

```
State: T_up, T̂_up, T_dn, T̂_dn (initially 0); c_up, c_dn (initially 1)

On each new sample (g_obs, dt, dd) while moving:

1: v̂       ← v_flat · r(g_obs)            ▷ uncorrected predicted speed
2: pred_dt ← dd / v̂
3: if g_obs ≥ 0:
4:     T_up  ← T_up  + dt
5:     T̂_up  ← T̂_up + pred_dt
6:     c_up  ← T̂_up / T_up
7: else:
8:     T_dn  ← T_dn  + dt
9:     T̂_dn  ← T̂_dn + pred_dt
10:    c_dn  ← T̂_dn / T_dn
```

A note on flat ground: `g_obs = 0` falls into the climb bucket (`≥ 0`), so on a flat ride `c_up` converges to the v_flat correction and `c_dn` stays at 1. The grouping assumes flat behaves more like climbing than like coasting — a reasonable default, but not the only one.

## The framework

A Godot estimator is built from three independent pieces:

| Piece              | Module                | Role                                                                   |
| ------------------ | --------------------- | ---------------------------------------------------------------------- |
| Pause strategy     | `godot/pause.py`      | How observed pauses enter the prediction (`NoPause`, `WallClockPause`) |
| Speed estimator    | `godot/estimators.py` | The speed / TTG prediction engine                                      |
| `v_flat` estimator | `godot/estimators.py` | Online flat-ground speed estimation, for estimators that want it       |

Estimators are organised in five complexity levels, from a flat moving average (L0) up through the integral physics estimators at L5, where the Godot estimator lives. See `CLAUDE.md` for the full taxonomy.

## Results

> 🚧 *TODO: results table.* Compare Godot against the baselines on the GPX corpus.
>
> | Estimator | Moving MAE (min) | Moving MAPE | Notes |
> |---|---|---|---|
> | `0_T_avg` (naive total avg) | _tbd_ | _tbd_ | pause blowup + gradient blind |
> | `0_M_avg` (naive moving avg) | _tbd_ | _tbd_ | gradient blind |
> | `1_M_physics` (physics prior, no correction) | _tbd_ | _tbd_ | open-loop |
> | `5_M_godot` (this work) | 9.3 | _tbd_ | |
> | `5_M_godot_oracle` (oracle `v_flat`) | _tbd_ | _tbd_ | upper bound |

> 🚧 *TODO: convergence plot.* ETA error vs. fraction of ride completed, Godot vs. baselines, averaged across the corpus.

> 🚧 *TODO: per-ride breakdown.* Godot's behaviour on rides with (a) a long terminal climb, (b) a long terminal descent, (c) significant pauses.

## Reproducing

```bash
uv sync
uv run python backtest.py data/gpx/*.gpx          # ETA accuracy
uv run python vflat_backtest.py                    # v_flat convergence
```

## Project structure

```
godot/              Core library
  gpx.py            GPX parsing and distance computation
  fit.py            FIT file parsing
  ride.py           Ride data pipeline
  estimators.py     ETA estimator implementations
  segmentation.py   Route segmentation by gradient (Visvalingam–Whyatt)
  pause.py          Pause handling strategies
  benchmark.py      Estimator accuracy metrics
  plot.py           Altair-based visualisation helpers
  report.py         HTML report generation
backtest.py         CLI for ETA backtests
vflat_backtest.py   CLI for v_flat convergence backtests
tests/              pytest suite
data/               GPX/FIT ride corpus
```

## Requirements

Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/).
