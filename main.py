#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: forecast Roblox's Up-and-Coming top 10.

Reads /input/history.ndjson (chart snapshots, oldest first) and
/input/task.json, writes one static forecast to /output/forecast.json: for
each place 1..10 a probability per named game, plus a top-10 guess. The same
forecast is scored against every hourly chart of the scored window, so the
probabilities are horizon-averaged over the whole window (its length is read
from task.json, never hardcoded).

Model (Python 3.13, standard library only, fully deterministic, no clock, no
network, no randomness):

- Per game, an anchor rank blends its last rank with a time-decayed EWMA of
  its recent ranks, then a momentum adjustment from its rank velocity over a
  ~14 h lookback and a smaller one from its player-count growth.
- At horizon h the anchor mean-reverts exponentially toward a mid-chart
  target rank, and the game's rank is Normal(mu(h), sigma(h)) with sigma
  growing like sqrt(h), mixed with a narrow persistence spike that stays on
  the anchor (the chart barely moves in some hours and jumps in others).
  P(game at place k at h) is the mixture's mass on [k-0.5, k+0.5]; the
  forecast averages this over a grid of horizons spanning the window.
- When the history's detail records carry a game's creation time, a game
  that ages past the chart's 28-day limit inside the window is scaled by the
  share of the window it can still be charted for; with no details (the
  public archive) the factor is 1.
- Per place, the named probabilities are scaled to leave a reserve of
  unnamed mass (growing slightly with the place) for games not named there:
  brand-new games and deep-chart games the forecast does not name. Rows below
  the level that unnamed share would beat are dropped, and no row exceeds a
  cap, so no place is bet confidently on a chart this volatile.
- top10 orders the candidates by their horizon-averaged projected rank.
"""

import json
import math
import os
from datetime import datetime, timedelta

PARAMS = {
    "ewma_half_life_h": 12.0,
    "w_last": 0.86,
    "vel_lookback_h": 14.0,
    "vel_horizon_h": 11.0,
    "vel_clip": 22.0,
    "pg_weight": 4.0,
    "mr_target": 44.0,
    "mr_tau_h": 45.0,
    "sigma0": 1.8,
    "sigma_sqrt": 1.5,
    "spike_weight": 0.12,
    "spike_sigma0": 0.6,
    "spike_sigma_sqrt": 0.25,
    "reserve_base": 0.16,
    "reserve_slope": 0.008,
    "p_cap": 0.42,
    "min_p": 0.0035,
}

SQRT2 = math.sqrt(2.0)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def forecast(rows, task, params=None):
    """Build the forecast dict from history rows (oldest first) and task."""
    p = dict(PARAMS)
    if params:
        p.update(params)
    places = int(task.get("places", 10))
    window = task.get("window") or {}
    slot_minutes = float(window.get("slot_minutes", 60))
    slots = float(window.get("slots", 72))
    horizon_h = slot_minutes * slots / 60.0
    window_from = None
    if window.get("from"):
        window_from = parse_time(window["from"])
    elif task.get("history_until"):
        window_from = parse_time(task["history_until"])

    rows = sorted(rows, key=lambda r: r["at"])
    times = [parse_time(r["at"]) for r in rows]
    t_end = times[-1]

    # Per-game series: list of (time, rank, playing).
    series = {}
    for r, t in zip(rows, times):
        for g in r.get("up_and_coming") or []:
            series.setdefault(g["universe_id"], []).append(
                (t, float(g["rank"]), g.get("playing"))
            )
    last_chart = sorted(
        rows[-1].get("up_and_coming") or [], key=lambda g: g["rank"]
    )
    last_ids = {g["universe_id"] for g in last_chart}

    # Creation times from details, when the history carries them (the house
    # recorder's reads do; the public archive's do not).
    created = {}
    for r in rows:
        det = r.get("details")
        if isinstance(det, dict):
            for key, val in det.items():
                if isinstance(val, dict) and val.get("created"):
                    try:
                        created[int(key)] = parse_time(val["created"])
                    except (ValueError, TypeError):
                        pass

    hl = p["ewma_half_life_h"]

    def lookback_obs(obs, target_h):
        """Observation closest to target_h hours before the last one."""
        if len(obs) < 2:
            return None
        t_last = obs[-1][0]
        best = None
        for o in obs[:-1]:
            dt = (t_last - o[0]).total_seconds() / 3600.0
            if dt < 3.0:
                continue
            if best is None or abs(dt - target_h) < abs(best[1] - target_h):
                best = (o, dt)
        return best

    # Features and horizon-averaged place probabilities per game.
    horizon_fracs = [0.04, 0.15, 0.29, 0.43, 0.57, 0.71, 0.86, 0.99]
    horizons = [f * horizon_h for f in horizon_fracs]
    raw = {}  # gid -> list of P(at place k), k=1..places
    mu_avg = {}  # gid -> horizon-averaged projected rank
    for gid, obs in series.items():
        # EWMA rank and presence with time decay to t_end.
        w_sum = 0.0
        wr_sum = 0.0
        for t, rank, _pl in obs:
            age = (t_end - t).total_seconds() / 3600.0
            w = 0.5 ** (age / hl)
            w_sum += w
            wr_sum += w * rank
        ewma_rank = wr_sum / w_sum if w_sum > 0 else obs[-1][1]
        total_w = sum(
            0.5 ** (((t_end - t).total_seconds() / 3600.0) / hl) for t in times
        )
        presence = w_sum / total_w if total_w > 0 else 0.0

        last_rank = obs[-1][1]
        on_last = gid in last_ids
        if on_last:
            base = p["w_last"] * last_rank + (1.0 - p["w_last"]) * ewma_rank
            mult = 1.0
        else:
            hours_gone = (t_end - obs[-1][0]).total_seconds() / 3600.0
            base = ewma_rank + 30.0 + min(hours_gone, 48.0)
            mult = 0.30 * min(presence * 3.0, 1.0)

        # Momentum: rank velocity and player-count growth over the lookback.
        adj = 0.0
        lb = lookback_obs(obs, p["vel_lookback_h"])
        if lb is not None:
            (t_old, rank_old, play_old), dt_h = lb
            if on_last or obs[-1][0] == t_end:
                vel = (rank_old - last_rank) / dt_h  # >0: climbing
                adj -= max(-p["vel_clip"], min(p["vel_clip"], vel * p["vel_horizon_h"]))
            play_now = obs[-1][2]
            if play_old is not None and play_now is not None:
                pg10 = (
                    math.log((play_now or 0) + 50.0)
                    - math.log((play_old or 0) + 50.0)
                ) / dt_h * 10.0
                pg10 = max(-1.0, min(1.0, pg10))
                adj -= pg10 * p["pg_weight"]
            r_adj = max(1.0, min(115.0, base + adj))

        # Ageing out: the chart only lists games published in the last 28
        # days. If this game crosses that line inside the window, scale by
        # the share of the window before the crossing.
        if gid in created and window_from is not None:
            exit_t = created[gid] + timedelta(days=28)
            share = (exit_t - window_from).total_seconds() / 3600.0 / horizon_h
            mult *= max(0.0, min(1.0, share))

        probs = [0.0] * places
        mu_sum = 0.0
        w_spike = p["spike_weight"]
        for h in horizons:
            mu = p["mr_target"] + (r_adj - p["mr_target"]) * math.exp(
                -h / p["mr_tau_h"]
            )
            mu_sum += mu
            sigma = p["sigma0"] + p["sigma_sqrt"] * math.sqrt(h)
            # Persistence spike: a narrow kernel that stays on the anchor,
            # for the hours where the chart barely moves; the broad kernel
            # above carries the hours where it jumps.
            s_spike = p["spike_sigma0"] + p["spike_sigma_sqrt"] * math.sqrt(h)
            for k in range(1, places + 1):
                broad = phi((k + 0.5 - mu) / sigma) - phi((k - 0.5 - mu) / sigma)
                spike = phi((k + 0.5 - r_adj) / s_spike) - phi(
                    (k - 0.5 - r_adj) / s_spike
                )
                probs[k - 1] += (
                    (1.0 - w_spike) * broad + w_spike * spike
                ) / len(horizons)
        raw[gid] = [mult * x for x in probs]
        mu_avg[gid] = mu_sum / len(horizons)

    # Per place: scale named rows to 1 - reserve, cap, drop rows that the
    # unnamed share would beat anyway.
    probabilities = []
    totals = {gid: 0.0 for gid in raw}
    for k in range(1, places + 1):
        reserve = min(0.35, p["reserve_base"] + p["reserve_slope"] * (k - 1))
        target = 1.0 - reserve
        col = {gid: raw[gid][k - 1] for gid in raw if raw[gid][k - 1] > 0.0}
        s = sum(col.values())
        if s <= 0:
            continue
        scale = target / s
        col = {gid: v * scale for gid, v in col.items()}
        # Two water-filling passes for the cap.
        for _ in range(2):
            over = {g: v for g, v in col.items() if v > p["p_cap"]}
            if not over:
                break
            fixed = sum(p["p_cap"] for _ in over)
            rest = {g: v for g, v in col.items() if g not in over}
            rs = sum(rest.values())
            if rs <= 0:
                col = {g: p["p_cap"] for g in over}
                break
            f = (target - fixed) / rs
            col = {**{g: p["p_cap"] for g in over}, **{g: v * f for g, v in rest.items()}}
        named = sorted(col.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [
            (g, v)
            for g, v in named
            if v >= p["min_p"]
        ][:100]
        for gid, v in kept:
            probabilities.append(
                {"universe_id": int(gid), "place": k, "p": round(v, 6)}
            )
            totals[gid] += v
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # Top 10: by projected rank, then by total top-10 probability.
    candidates = sorted(
        raw.keys(), key=lambda g: (mu_avg[g], -totals[g], g)
    )
    top10 = [int(g) for g in candidates[:places]]
    if len(top10) < places:  # degenerate history: pad from the last chart
        for g in last_chart:
            if g["universe_id"] not in top10:
                top10.append(int(g["universe_id"]))
            if len(top10) >= places:
                break

    return {"probabilities": probabilities, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out = forecast(rows, task)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
