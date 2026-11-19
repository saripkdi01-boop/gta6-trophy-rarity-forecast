"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case's target as five
quantiles (5/25/50/75/95%) in percent, in logit space.

Model (backtested on the two practice cases against simulated answers;
see README for the honest backtest method):
  * The tracked-players denominator grows (G_N): observed log-trend of
    players_tracked extrapolated to the horizon, floored by a refresh
    prior (one Exophase refresh batch ~5x on a short horizon, power-law
    growth on a long horizon), capped at 10x.
  * Median in logit space: dilution from denominator growth, partly
    offset by catch-up earning. Fast trophies (early chapters, quick
    activities) catch up much more than grind/collectible trophies.
  * The 100% trophy (an artefact: every one of the first 638 tracked
    players had it) is handled with a count model, not a logit from 99.99.
  * Spread is deliberately wide: a refresh can move almost every rate,
    so the band widens with horizon and carries an explicit refresh term.

Standard library only, deterministic, no network, no model calls.
Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch epoch used by the refresh prior: 2026-11-19T00:00:00Z.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (same as the starter's JS-style rule)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def fast_score(trophy: dict) -> float:
    """How fast new players earn this trophy, 0 (very slow) to 1 (very fast).

    Prior from tier + description keywords only; the observed history is
    flat (every read identical), so no per-trophy earned-rate signal exists
    to learn from — the prior is all we have, and the wide spread carries
    the uncertainty.
    """
    name = trophy.get("name", "")
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier", "")

    # Slowest: the endgame.
    if tier == "platinum":
        return 0.15
    if "100% completion" in desc or "final mission" in desc:
        return 0.15

    # Grind / collectibles: many repetitions, slow even over 4.5 days.
    grind = (
        "250 stunts" in desc
        or "100 vehicles" in desc
        or "50 challenges" in desc
        or "100 locations" in desc
        or "card sets" in desc
        or "20 different races" in desc
        or "gold medal in all" in desc
        or "every location" in desc
        or "8 mission strands" in desc
        or "meditate at every spot" in desc
        or "explore all of leonida" in desc
        or "rob 5 bank vaults" in desc
        or "register 10 vehicles" in desc
    )
    if grind:
        return 0.3

    # Money trophies: need cash first, mid-speed.
    if "laundering" in desc or "spend $100,000" in desc:
        return 0.4
    if "purchase all garage" in desc:
        return 0.35

    # Story chapters: early chapters are reached in the first hours.
    if "complete chapter 1" in desc:
        return 0.85
    if "complete chapter 2" in desc:
        return 0.75
    if "complete chapter 3" in desc:
        return 0.7
    if "complete chapter 4" in desc:
        return 0.65

    # Quick one-off activities, usually done in the first session.
    quick = (
        "gone fishing" in desc
        or "arcade game" in desc
        or "cocktail" in desc
        or "bowling" in desc
        or "try each type of activity" in desc
        or "fishing challenge" in desc
    )
    if quick:
        return 0.9

    # Named story missions and other single activities: mid-fast.
    # (Check the "final mission for <person>" strands before the generic
    # "Complete ..." rule, so they are not swallowed by it.)
    if "complete the final mission for" in desc:
        return 0.7
    if desc.startswith("complete "):
        return 0.7

    # Fallback by tier.
    if tier == "gold":
        return 0.3
    if tier == "silver":
        return 0.45
    return 0.6


def growth_factor(case: dict) -> float:
    """G_N: players_tracked at target divided by players_tracked at last read."""
    reads = case["history"]
    n_last = reads[-1]["players_tracked"]
    horizon_days = case["horizon_minutes"] / 1440.0

    # Observed trend: log growth per day across the history window.
    g_trend = 1.0
    if len(reads) >= 2:
        n_first = reads[0]["players_tracked"]
        t_first = parse_time(reads[0]["slot"])
        t_last = parse_time(reads[-1]["slot"])
        span_days = max((t_last - t_first).total_seconds(), 0.0) / 86400.0
        if span_days > 0 and n_first > 0 and n_last > n_first:
            rate_per_day = math.log(n_last / n_first) / span_days
            g_trend = math.exp(rate_per_day * horizon_days)

    # Refresh prior: Exophase refreshes in batches, not continuously.
    horizon_hours = case["horizon_minutes"] / 60.0
    if horizon_hours < 6:
        g_refresh = 5.0
    else:
        at = parse_time(case["at"])
        target = parse_time(case["target"])
        t_checkpoint = max((at - EPOCH).total_seconds() / 3600.0, 1.0)
        t_target = max((target - EPOCH).total_seconds() / 3600.0, 1.0)
        g_refresh = (t_target / t_checkpoint) ** 1.2

    g = max(g_trend, g_refresh)
    return min(g, 10.0)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    last = case["history"][-1]
    n_last = last["players_tracked"]
    horizon_days = case["horizon_minutes"] / 1440.0
    g_n = growth_factor(case)
    n_target = n_last * g_n

    # Spread in logit units: base uncertainty + random-walk growth with
    # horizon + an explicit refresh term (a refresh batch moves rates a lot).
    refresh = 0.9 if horizon_days < 0.5 else 1.3
    sigma = min(2.0, math.sqrt(0.55 ** 2 + (0.75 * math.sqrt(horizon_days)) ** 2 + refresh ** 2))

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        if rate is None:
            rate = 0.0
        fast = fast_score(trophy)

        if rate >= 99.5:
            # Artefact trophy: ~100% only because every early tracked player
            # had it. Model counts: old players keep it, new players earn it
            # at a rate that rises slowly with horizon.
            q_new = 0.6 * horizon_days / (horizon_days + 3.0)
            median = (n_last * 1.0 + (n_target - n_last) * q_new) / n_target * 100.0
        else:
            beta = 0.2 + 1.0 * fast
            shift = -math.log(g_n) * (1.0 - 0.25 * fast) + beta * math.log1p(horizon_days)
            median = inv_logit(logit(rate) + shift)

        median = min(99.99, max(0.01, median))
        lg_med = logit(median)
        qs = []
        prev = 0.0
        for z in Z:
            q = inv_logit(lg_med + z * sigma)
            q = round4(min(100.0, max(0.0, q)))
            q = max(q, prev)  # keep quantiles non-decreasing after rounding
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
