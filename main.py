"""Launch-day Twitch viewer forecaster (make.fun gta6-twitch-ccu-forecast-01).

Python 3.13, standard library only, deterministic, no network.

Idea
----
A launch day's hourly curve, divided by its own first hour, has a shape that
repeats across big launches: an early dip while the first regions finish and
the US sleeps, a climb as the US unlock lands, an EU/US evening peak, a late
decay. The case file carries 38 past launches with their full release-day
hourly series, so instead of guessing one shape this program reads the shape
off those launches:

  * for every comparable with a usable release day, take its log-ratios
        r_c(h) = ln( viewers_c(h) / viewers_c(1) ),  h = 2..24;
  * weight each comparable by how close its own hour-1 audience is to this
    case's hour-1 print, on a log10 scale (Gaussian kernel, sd 0.45 dex).
    Scale is the feature that separates the regimes: a launch whose first
    hour is already five or six figures was unlocked somewhere hours earlier
    and moves within a few x of it all day, while a launch whose first hour
    is a few hundred pre-unlock viewers can multiply a hundredfold by its
    evening peak, and the two shapes must not be averaged together;
  * for each hour, read the five scored quantiles off the weighted empirical
    distribution of the log-ratios, smooth the quantile curves lightly across
    neighbouring hours (each hour's estimate borrows a little of its
    neighbours', which are highly correlated), and enforce a floor on the
    log-spread that widens like the persistence baseline's, so no hour is
    ever forecast with more confidence than the data supports;
  * the forecast for hour h is p1 * exp(r(h)), p1 = this case's hour-1 print.

Validation while building (leave-one-out over the 37 comparable launches with
complete days, plus the kit's four practice cases scored against the same
SullyGnome series the kit was built from): mean pinball loss 0.409 LOO and
0.254 on the practice cases, against the persistence starter's 0.387 on the
practice cases. Text features (unlock wording, platform lists, launch
weekday, pre-launch ramp) were all tried and dropped: none of them improved
both validations at once.

How the grader runs it: `python3 -P -s /bot/main.py`, one case per run in
/input, forecast to /output/<id>.out: {"hours": [{"hour": 2..24, "q":
[q05, q25, q50, q75, q95]}]}, 23 rows, q non-decreasing, finite, in (0, 1e9].
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, in case a
# sibling module is ever shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Kernel width for the hour-1 scale distance, in log10 units.
SCALE_SD = 0.45
# A comparable needs at least this many release-day hourly points to be used.
MIN_DAY_POINTS = 20
# Longest interior gap in a comparable's day that interpolation may fill.
MAX_GAP = 3
# Neighbour-hour smoothing weight for the log-ratio quantile curves.
SMOOTH = 0.3
# Spread floor: the q05..q95 log width of hour h is at least
# 2 * 1.645 * (FLOOR_BASE + FLOOR_GROWTH * sqrt(h - 1)).
FLOOR_BASE = 0.12
FLOOR_GROWTH = 0.10


def parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_curve(comp: dict) -> dict[int, float]:
    """The comparable's release-day viewers by hour (1..24), from its hourly list.

    SullyGnome samples a small category only every few hours, so a comparable's
    day can have interior holes. A hole of at most MAX_GAP hours with a real
    point on both sides is filled by log-linear interpolation: within one
    launch day the curve moves smoothly enough that the interpolated ratio is
    a better sample than dropping the comparable for that hour entirely.
    Longer holes, and holes at the day's edges, are left empty.
    """
    start = parse_ts(comp["day_start"])
    end = start + timedelta(hours=24)
    curve: dict[int, float] = {}
    for point in comp["hourly"]:
        t = parse_ts(point["from"])
        if start <= t < end:
            curve[int((t - start).total_seconds() // 3600) + 1] = point["viewers"]
    present = sorted(curve)
    for a, b in zip(present, present[1:]):
        gap = b - a - 1
        if 0 < gap <= MAX_GAP and curve[a] > 0 and curve[b] > 0:
            la, lb = math.log(curve[a]), math.log(curve[b])
            for h in range(a + 1, b):
                curve[h] = math.exp(la + (lb - la) * (h - a) / (b - a))
    return curve


def weighted_quantile(sorted_pairs: list[tuple[float, float]], tau: float) -> float:
    """Quantile of a weighted sample: inverse CDF, linearly interpolated."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    target = tau * total
    acc = 0.0
    prev_v, prev_acc = sorted_pairs[0][0], 0.0
    for v, w in sorted_pairs:
        acc += w
        if acc >= target:
            if acc == prev_acc:
                return v
            frac = (target - prev_acc) / (acc - prev_acc)
            return prev_v + frac * (v - prev_v)
        prev_v, prev_acc = v, acc
    return sorted_pairs[-1][0]


def round2(x: float) -> float | int:
    """Round to 2 decimals, half up (as the kit starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, first["viewers"])
    log_p1 = math.log10(p1)

    # Weighted log-ratio samples per hour, from the comparables.
    samples: dict[int, list[tuple[float, float]]] = {h: [] for h in range(2, 25)}
    for comp in sorted(case["comparables"], key=lambda c: c["name"]):
        curve = day_curve(comp)
        if 1 not in curve or len(curve) < MIN_DAY_POINTS:
            continue
        c_p1 = max(1.0, curve[1])
        dist = abs(math.log10(c_p1) - log_p1)
        weight = math.exp(-(dist * dist) / (2 * SCALE_SD * SCALE_SD))
        for h in range(2, 25):
            v = curve.get(h)
            if v and v > 0:
                samples[h].append((math.log(v / c_p1), weight))
    for h in samples:
        samples[h].sort()

    # Weighted empirical quantiles of the log-ratios, per hour.
    raw: dict[int, list[float]] = {}
    for h in range(2, 25):
        pairs = samples[h] or [(0.0, 1.0)]
        raw[h] = [weighted_quantile(pairs, tau) for tau in LEVELS]

    # Light smoothing across neighbouring hours.
    if SMOOTH:
        smoothed: dict[int, list[float]] = {}
        for h in range(2, 25):
            acc = [q * (1 - SMOOTH) for q in raw[h]]
            tot = 1 - SMOOTH
            for nb, wgt in ((h - 1, SMOOTH / 2), (h + 1, SMOOTH / 2)):
                if nb in raw:
                    for i in range(5):
                        acc[i] += wgt * raw[nb][i]
                    tot += wgt
            smoothed[h] = [a / tot for a in acc]
        raw = smoothed

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = raw[h]
        mid = qs[2]
        # Spread floor, widening like sqrt(h - 1).
        width = qs[4] - qs[0]
        min_width = 2 * 1.645 * (FLOOR_BASE + FLOOR_GROWTH * math.sqrt(h - 1))
        if width < min_width:
            scale = min_width / max(width, 1e-9)
            qs = [mid + (q - mid) * scale for q in qs]
        vals = [min(1e9, max(0.01, p1 * math.exp(q))) for q in qs]
        vals.sort()
        hours.append({"hour": h, "q": [round2(v) for v in vals]})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
