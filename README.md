# GTA 6 Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saripkdi01-boop/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saripkdi01-boop/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one — a
[make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty entry
($250 in USDC). One Python file, standard library only, deterministic, no
network, no model calls.

## What it does

`main.py` reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes one `*.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```json
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
```

55 trophies per case, quantiles non-decreasing, each in 0–100, output
≤ 65,536 bytes. Two runs on the same case produce byte-identical output.

## The model

Forecasts live in **logit space** per trophy. The practice cases show why
persistence fails: all 7–8 launch-night reads are perfectly flat
(638 players tracked, identical rates), then the first Exophase refresh
jumps tracked players 638 → 3,868, so almost every rate *falls* because the
denominator grows ~6×. A persistence forecast (the house baseline) scores
~1.5–1.9 pinball loss on practice with ~0 coverage of its 90% band.

1. **Denominator growth `G_N`** — observed log-trend of `players_tracked`
   extrapolated to the horizon, floored by a refresh prior (one refresh
   batch ≈ 5× on a < 6 h horizon; power-law `(t_target/t_checkpoint)^1.2`
   on a long horizon, `t` in hours since 2026-11-19T00:00Z), capped at 10×.
2. **Median** — `shift_logit = -ln(G_N)·(1 − 0.25·fast) + β·ln(1 + horizon_days)`,
   `β = 0.2 + 1.0·fast`. The first term is denominator dilution; the second
   is catch-up earning, small at 90 minutes and large at 4.5 days for fast
   trophies, small for slow ones. `fast` (0–1) is a prior from tier +
   description: early chapters and quick activities (fishing, arcade,
   cocktail, bowling, try-each-activity) 0.8–0.9; story missions 0.65–0.75;
   money trophies 0.35–0.4; grind/collectibles (250 stunts, 100 vehicles,
   50 challenges, 100 locations, card sets, 20 races) 0.3; final mission /
   100% completion / platinum 0.15.
3. **The 100% trophy** ("Just Like Paradise": 100% because every one of
   the first 638 tracked players had it) is *not* forecast from a logit
   of 99.99. A count model is used instead: old players keep it, new
   players earn it at `q = 0.6·days/(days+3)`, giving a practice median
   of ~21% and a much higher final median.
4. **Spread** — deliberately wide: `σ = min(2.0, √(0.55² + (0.75·√days)²
   + refresh²))`, `refresh = 0.9` (< 0.5 day) else `1.3`. Practice σ ≈ 1.07,
   final σ = 2.0. Quantiles use z = (−1.645, −0.674, 0, 0.674, 1.645),
   rounded to 4 decimals half-up, then forced non-decreasing.

## Backtest (honest)

The real practice answers are held by the house, so the backtest below is a
**simulation, not the official practice score**: target tracked players
3,868 (the observed first refresh), earned counts grown by the stated
scenario. Pinball loss on the logit scale, mean over trophies and quantiles:

| scenario | this model | starter (persistence) |
|---|---|---|
| earned +0% | 0.30 | 1.88 |
| fast +20% / slow +5% | 0.28 | 1.76 |
| fast +50% / slow +10% | 0.25 | 1.61 |

("fast" = `fast_score ≥ 0.5`.) The margin is large on practice, but practice
is only 90 minutes while the final horizon is 4.5 days — parameters were
taken from the design above as-is and **not** tuned to exhaustion on the
two practice cases, precisely because overfitting two 90-minute points
would be the wrong way to win a 4.5-day forecast.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Requires Python 3.13 on the grader (tested locally on 3.12, stdlib only).
Runtime is well under 1 s per case.

## License

MIT — see [LICENSE](LICENSE).
