[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saripkdi01-boop/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saripkdi01-boop/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01 — model v2 (count-based, horizon-aware)

Forecasts each of GTA VI's 55 trophies' Exophase earn rate at a case's target
time, as 5 quantiles (5/25/50/75/95%) in percent. Python 3.13, standard
library only, no network, deterministic.

## Model

Work in **counts**, not raw logit shifts. A rate is earned / tracked, and when
Exophase's tracked-player count jumps (a refresh), the denominator grows and
every rate is diluted even though nobody un-earns a trophy.

Per trophy, with `N_last` = tracked players in the last read and
`rate` = its checkpoint rate (percent):

- `E_old = rate * N_last / 100` — players who already earned it.
- `N_target = N_last * G` — tracked players at the target.
- `E_new = (N_target - N_last) * f / 100`, where `f = rate * m` is the new
  cohort's earn rate and `m = 0.02 + 0.35 * fast` is a per-trophy catch-up
  fraction (`fast` in 0..1, from tier + description keywords: story chapters
  and one-off activities are fast; grinds, collectibles and 100% completion
  are slow; money trophies sit in the middle).
- Median `q50 = (E_old + E_new) / N_target * 100`.

Growth factor `G = min(max(g_trend, g_refresh), 10)`:

- `g_trend` — log-trend of `players_tracked` over the case history,
  extrapolated over the horizon (1.0 when history is flat).
- `g_refresh` — refresh prior: 6.0 when the horizon is under 6 hours (the
  first observed refresh multiplied the count ~6.06x, 638 → 3868), otherwise
  `(t_target / t_checkpoint) ** 1.2` with `t` = hours since
  2026-11-19T00:00:00Z (each floored at 1.0).

The 100% trophy (rate ≥ 99.99) is handled in counts, not via a logit from
99.99: `median = (N_last + (N_target - N_last) * q_new) / N_target * 100`
with `q_new = 0.6 * horizon_days / (horizon_days + 3)`.

Quantiles spread around the median on the logit scale:

```
sigma = min(1.5, sqrt(0.30² + (0.40·√horizon_days)² + refresh²))
refresh = 0.40 if horizon_days < 0.5 else 0.80
q_τ = inv_logit(logit(median) + z_τ · sigma),  z = (-1.645, -0.674, 0, 0.674, 1.645)
```

So the 90-minute practice horizon gets sigma ≈ 0.51 (much tighter than v1's
≈ 1.07), while the 4.5-day final horizon gets sigma ≈ 1.20. Quantiles are
rounded to 4 decimals (half up), forced non-decreasing after rounding, and
clamped to [0, 100] (median clamped to [0.01, 99.99] before the logit).

## Usage

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input dir, writes one `<id>.out` per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Validation (honest method)

1. Run over the kit inputs (`p00.in`, `p01.in`); check every `.out` is valid
   JSON with 55 trophies, non-decreasing quantiles, values in [0, 100], size
   ≤ 65,536 bytes. Run twice and diff — outputs must be byte-identical.
2. Score with the coordinator's local harness
   (`~/workspace/makefun/forecast-eval/eval.py`) against synthetic answers at
   several values of `c`, and compare with the v1 numbers measured with the
   same harness. The harness measures loss against *synthetic* answers, not
   the real Exophase answers — so it ranks models only under the harness's
   own assumptions, and a win there does not guarantee a win on the real
   leaderboard (v1's real practice loss is 0.450051). Report the harness
   numbers as harness numbers, nothing more.
