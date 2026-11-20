"""Forecast GTA VI trophy earn rates on Exophase (make.fun gta6-trophy-rarity-forecast-01).

Model v2: count-based, horizon-aware.

Idea: a rate is a fraction, earned / tracked. When Exophase's tracked-player
count jumps (a refresh), the denominator grows and every rate is diluted, even
though nobody "un-earns" a trophy. So instead of shifting logits, this model
works in counts:

  * E_old -- players who already earned the trophy at the checkpoint:
    rate_checkpoint * N_last / 100.
  * N_target -- tracked players at the target: N_last * G, where G blends the
    observed log-trend of players_tracked with a refresh prior (the first
    observed refresh multiplied the tracked count ~6x).
  * E_new -- new players who earn the trophy before the target:
    (N_target - N_last) * f / 100, where f is the new cohort's earn rate,
    a per-trophy fraction m of the checkpoint rate. Fast trophies (early story,
    one-off activities) are picked up by newcomers much more than slow ones
    (grinds, collectibles, 100% completion).
  * Median forecast = (E_old + E_new) / N_target * 100.

Quantiles are then spread around that median on the logit scale with a sigma
that grows with the horizon and with whether a refresh is expected, so the
90-minute practice horizon gets a much tighter band than the 4.5-day final.

Standard library only, no network, no randomness: two runs on the same case
are byte-identical.
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
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Reference moment for the refresh prior: launch-day midnight, UTC.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# First observed refresh multiplied the tracked count by ~6.06 (638 -> 3868).
FIRST_REFRESH_GROWTH = 6.0
MAX_GROWTH = 10.0


def parse_time(text: str) -> datetime:
    """Parse an ISO-8601 timestamp ending in Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(moment: datetime) -> float:
    return max(1.0, (moment - EPOCH).total_seconds() / 3600.0)


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def logit(pct: float) -> float:
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round_half_up_4(x: float):
    """Round to 4 decimals, halves away from zero (matches the starter's
    JavaScript-style rounding), returning an int for whole numbers."""
    scaled = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(scaled) if scaled == int(scaled) else scaled


def trend_growth(history, horizon_hours: float) -> float:
    """Growth factor implied by the log-trend of players_tracked over the
    reads in history, extrapolated over the horizon. Flat history -> 1.0."""
    points = []
    if not history:
        return 1.0
    t0 = parse_time(history[0]["at"])
    for read in history:
        n = read.get("players_tracked")
        if n and n > 0:
            hours = (parse_time(read["at"]) - t0).total_seconds() / 3600.0
            points.append((hours, math.log(n)))
    if len(points) < 2:
        return 1.0
    n_pts = len(points)
    mean_t = sum(p[0] for p in points) / n_pts
    mean_y = sum(p[1] for p in points) / n_pts
    denom = sum((p[0] - mean_t) ** 2 for p in points)
    if denom <= 0:
        return 1.0
    slope = sum((p[0] - mean_t) * (p[1] - mean_y) for p in points) / denom
    return clamp(math.exp(slope * horizon_hours), 0.2, 50.0)


def refresh_growth(horizon_minutes: float, t_checkpoint: float, t_target: float) -> float:
    """Prior for the tracked-count jump caused by an Exophase refresh.

    Short horizons (< 6 h) sit inside the first refresh batch, which
    multiplied the count ~6x. Longer horizons grow more slowly, roughly as a
    power of elapsed time since launch."""
    if horizon_minutes < 360.0:
        return FIRST_REFRESH_GROWTH
    return (t_target / t_checkpoint) ** 1.2


def fast_score(description: str, tier: str) -> float:
    """How quickly a newcomer picks up this trophy, 0 (very slow) to 1 (very
    fast). Built from the trophy's tier and the wording of its description:
    story chapters and one-off activities are fast; grinds, collectibles and
    completionist trophies are slow; money trophies sit in the middle."""
    text = (description or "").lower()
    tier = (tier or "").lower()

    if tier == "platinum":
        return 0.05
    if "100% completion" in text:
        return 0.05
    if "final mission for" in text:
        return 0.7
    if "final mission" in text:
        return 0.1

    for chapter, score in (("chapter 1", 0.85), ("chapter 2", 0.75),
                           ("chapter 3", 0.70), ("chapter 4", 0.65)):
        if chapter in text:
            return score

    slow_markers = (
        "stunts", "collectible card", "100 locations", "50 challenges",
        "20 different races", "gold medal in all", "every location",
        "8 mission strands", "meditate", "explore all of",
        "rob 5 bank", "register 10 vehicles", "discover 100 vehicles",
    )
    if any(marker in text for marker in slow_markers):
        return 0.1

    money_markers = ("laundering", "spend $100,000", "purchase all garage")
    if any(marker in text for marker in money_markers):
        return 0.4

    quick_markers = ("fishing", "arcade", "cocktail", "bowl",
                     "try each type of activity")
    if any(marker in text for marker in quick_markers):
        return 0.9

    if text.startswith("complete "):
        return 0.7

    if tier == "gold":
        return 0.3
    if tier == "silver":
        return 0.45
    return 0.6


def spread_sigma(horizon_days: float) -> float:
    """Logit-scale sigma of the forecast band. A refresh inside the horizon
    adds uncertainty; a multi-day horizon is genuinely much less certain
    than a 90-minute one."""
    refresh = 0.40 if horizon_days < 0.5 else 0.80
    variance = 0.30 ** 2 + (0.40 * math.sqrt(horizon_days)) ** 2 + refresh ** 2
    return min(1.5, math.sqrt(variance))


def last_known_rate(history, index: int):
    """The trophy's rate in the latest read that lists it, else None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return rates[index]
    return None


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    history = case["history"]
    n_last = None
    for read in reversed(history):
        if read.get("players_tracked"):
            n_last = read["players_tracked"]
            break
    if n_last is None:
        raise ValueError(f"{name} has no players_tracked in its history")
    horizon_minutes = float(case["horizon_minutes"])
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    t_checkpoint = hours_since_epoch(parse_time(case["at"]))
    t_target = hours_since_epoch(parse_time(case["target"]))

    g_trend = trend_growth(history, horizon_hours)
    g_refresh = refresh_growth(horizon_minutes, t_checkpoint, t_target)
    growth = min(max(g_trend, g_refresh), MAX_GROWTH)
    n_target = n_last * growth
    newcomers = n_target - n_last

    sigma = spread_sigma(horizon_days)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_known_rate(history, i)
        if rate is None:
            rate = 0.0

        if rate >= 99.99:
            # Everybody already has it: only newcomers can still be missing
            # it, and a share of them catch up over the horizon.
            catch_up = 0.6 * horizon_days / (horizon_days + 3.0)
            median = (n_last * 1.0 + newcomers * catch_up) / n_target * 100.0
        else:
            speed = fast_score(trophy.get("description", ""), trophy.get("tier", ""))
            catch_up_fraction = 0.02 + 0.35 * speed
            earned_old = rate * n_last / 100.0
            earned_new = newcomers * (rate * catch_up_fraction) / 100.0
            median = (earned_old + earned_new) / n_target * 100.0

        centre = logit(clamp(median, 0.01, 99.99))
        quantiles = []
        previous = 0.0
        for z in Z_SCORES:
            value = clamp(inv_logit(centre + z * sigma), 0.0, 100.0)
            value = round_half_up_4(value)
            # Rounding must not break the non-decreasing order.
            if value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"id": trophy["id"], "q": quantiles})
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
