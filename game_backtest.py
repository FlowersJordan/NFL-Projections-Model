#!/usr/bin/env python3
"""
Did our projected game scores beat Vegas this season?

For every finished week of the current season this re-runs weekly_projections.py exactly as it would have run
BEFORE that week's games: only earlier games are visible, the models are refit on that data alone, the injury
reports of that week and the rosters as of the previous week are used. It then compares our projected score,
total and spread with the actual result and with Vegas's line.

    python game_backtest.py                  # all finished weeks of the current season
    python game_backtest.py --weeks 3 5      # just weeks 3 to 5
    python game_backtest.py --season 2025    # another season

Needs weekly_projections.py in the same folder. Takes about a minute per week.
Writes  game_backtest_<season>.csv          one row per game
        game_backtest_<season>_summary.csv  accuracy and betting records
Per-week working files go in ./game_backtest_runs/ so your real weekly projection files are never touched.

Honesty notes
  * The team-score formula is refit on seasons BEFORE the one tested, so it has not seen these games.
  * Lines are nflverse's spread_line / total_line, which are close to the closing numbers.
  * 'Record' = how our side of each line would have done, ignoring the betting vig (break-even at -110 is 52.4%).
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weekly_projections as W   # noqa: E402


def memo(fn):
    box = {}

    def inner(*a, **k):
        key = (a, tuple(sorted(k.items())))
        if key not in box:
            box[key] = fn(*a, **k)
        return box[key]
    return inner


def binom_p(wins, n):
    """Two-sided normal-approximation p-value that the win rate differs from 50%."""
    if n == 0:
        return np.nan
    z = (wins - n / 2) / np.sqrt(n / 4)
    from math import erf, sqrt
    return 1 - erf(abs(z) / sqrt(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse per-week projections already saved in game_backtest_runs/ (skips the slow refit)")
    ap.add_argument("--weeks", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None)
    args = ap.parse_args()
    season = args.season or nfl.get_current_season()
    out_dir = Path.cwd()
    runs = out_dir / "game_backtest_runs"
    runs.mkdir(exist_ok=True)

    sched = nfl.load_schedules(seasons=season).to_pandas()
    sched = sched[sched.game_type == "REG"]
    done = sched[sched.home_score.notna()]
    weeks_done = sorted(w for w in done.week.unique() if (sched[sched.week == w].home_score.notna()).all() or w == done.week.max())
    if args.weeks:
        weeks_done = [w for w in weeks_done if args.weeks[0] <= w <= args.weeks[1]]
    if not weeks_done:
        print("No finished weeks to test.")
        return
    print(f"Testing {season} weeks {weeks_done[0]}-{weeks_done[-1]} ({len(weeks_done)} weeks).")

    orig_load = memo(W.load_data)
    orig_inj = memo(W.load_injuries_all)
    orig_ros = memo(W.load_rosters)

    # team-score formula fit only on seasons BEFORE the one being tested (the saved backtest has 2022+)
    bt_path = out_dir / "backtest_weekly_players.csv"
    coef = dict(W.TEAM_PTS)
    if bt_path.exists():
        st0, g0 = orig_load(season, 0.5, 4.0)
        bt = pd.read_csv(bt_path)
        bt = bt[bt.season < season]
        if len(bt) > 2000:
            with contextlib.redirect_stdout(io.StringIO()):
                coef = W.fit_team_points(bt, g0)
            print("Team-score formula refit on earlier seasons only: " +
                  " ".join(f"{k}={v:.3f}" for k, v in coef.items()))
    else:
        print("(backtest_weekly_players.csv not found: using the built-in team-score formula, which was fit on "
              "2022-2026 and has therefore seen some of these games; run weekly_projections.py --backtest first "
              "for a clean test)")
    W.TEAM_PTS.update(coef)

    rows = []
    for wk in weeks_done:
        fp = runs / f"game_projections_{season}_wk{wk:02d}.csv"
        if args.reuse and fp.exists():
            gp = pd.read_csv(fp)
            gp["week"] = wk
            rows.append(gp)
            print(f"  week {wk}: reused saved projections")
            continue
        print(f"  week {wk} ...", flush=True)

        def cut(s, tp, ptd=4.0, _wk=wk):
            st, g = orig_load(s, tp, ptd)
            st = st[~((st.season == season) & (st.week >= _wk))].copy()
            g = g.copy()
            g.loc[(g.season == season) & (g.week >= _wk), "played"] = False
            return st, g

        W.load_data = cut
        W.load_injuries_all = lambda seasons: orig_inj(tuple(seasons))
        W.load_rosters = lambda s, _wk=wk: (lambda r: None if r is None else r[r.week <= _wk - 1])(orig_ros(s))
        os.chdir(runs)
        # in week 1 every player's last game was in the previous season, so allow a long gap
        sys.argv = ["weekly_projections.py", "--season", str(season), "--week", str(wk)] + (
            ["--max-gap-days", "300"] if wk == 1 else [])
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                W.main()
        except Exception as e:
            os.chdir(out_dir)
            print(f"    week {wk} failed: {str(e)[:100]}")
            continue
        os.chdir(out_dir)
        fp = runs / f"game_projections_{season}_wk{wk:02d}.csv"
        if not fp.exists():
            print(f"    week {wk}: no game projections were produced; last output lines: "
                  + " | ".join(buf.getvalue().strip().splitlines()[-3:])[:200])
            continue
        gp = pd.read_csv(fp)
        gp["week"] = wk
        rows.append(gp)
    if not rows:
        print("Nothing could be projected.")
        return

    g = pd.concat(rows, ignore_index=True)
    res = sched[["week", "home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"home_team": "home", "away_team": "away"})
    for c in ("home", "away"):
        res[c] = res[c].replace(W.TEAM_FIX)
    g = g.merge(res, on=["week", "home", "away"], how="left").dropna(subset=["home_score"])
    g = g.rename(columns={"home_pts": "our_home", "away_pts": "our_away", "vegas_home": "vegas_home_pts",
                          "vegas_away": "vegas_away_pts"})
    g["actual_total"] = g.home_score + g.away_score
    g["actual_margin"] = g.home_score - g.away_score
    g["err_total_ours"] = g.our_total - g.actual_total
    g["err_total_vegas"] = g.vegas_total - g.actual_total
    g["err_margin_ours"] = g.our_spread - g.actual_margin
    g["err_margin_vegas"] = g.vegas_spread - g.actual_margin
    g["closer_total"] = np.where(g.err_total_ours.abs() < g.err_total_vegas.abs(), "ours",
                                 np.where(g.err_total_ours.abs() > g.err_total_vegas.abs(), "vegas", "tie"))
    g["closer_margin"] = np.where(g.err_margin_ours.abs() < g.err_margin_vegas.abs(), "ours",
                                  np.where(g.err_margin_ours.abs() > g.err_margin_vegas.abs(), "vegas", "tie"))
    # our side of each line
    g["pick_total"] = np.where(g.total_diff > 0, "OVER", np.where(g.total_diff < 0, "UNDER", "none"))
    over = np.sign(g.actual_total - g.vegas_total)
    g["total_result"] = np.where(g.total_diff == 0, "no pick", np.where(over == 0, "push",
                                 np.where((over > 0) == (g.total_diff > 0), "win", "loss")))
    g["pick_spread"] = np.where(g.spread_diff > 0, g.home, np.where(g.spread_diff < 0, g.away, "none"))
    cover = np.sign(g.actual_margin - g.vegas_spread)
    g["spread_result"] = np.where(g.spread_diff == 0, "no pick", np.where(cover == 0, "push",
                                  np.where((cover > 0) == (g.spread_diff > 0), "win", "loss")))
    g["big_total_gap"] = g.total_diff.abs() >= 3
    g["big_spread_gap"] = g.spread_diff.abs() >= 3
    cols = ["week", "away", "home", "our_away", "our_home", "away_score", "home_score",
            "our_total", "vegas_total", "actual_total", "err_total_ours", "err_total_vegas", "closer_total",
            "pick_total", "total_result",
            "our_spread", "vegas_spread", "actual_margin", "err_margin_ours", "err_margin_vegas", "closer_margin",
            "pick_spread", "spread_result", "total_diff", "spread_diff", "big_total_gap", "big_spread_gap"]
    g = g[cols].sort_values(["week", "home"]).round(2)
    path = out_dir / f"game_backtest_{season}.csv"
    g.to_csv(path, index=False)

    # ---- summary
    def acc(e_o, e_v, name):
        return {"measure": name, "games": len(e_o), "ours_MAE": e_o.abs().mean(), "vegas_MAE": e_v.abs().mean(),
                "ours_RMSE": float(np.sqrt((e_o ** 2).mean())), "vegas_RMSE": float(np.sqrt((e_v ** 2).mean())),
                "ours_bias": e_o.mean(), "vegas_bias": e_v.mean(),
                "ours_closer_share": float((e_o.abs() < e_v.abs()).mean())}
    s = [acc(g.err_total_ours, g.err_total_vegas, "game total"), acc(g.err_margin_ours, g.err_margin_vegas, "margin (spread)")]
    t = np.concatenate([g.our_home, g.our_away]) - np.concatenate([g.home_score, g.away_score])
    v = np.concatenate([g.vegas_total / 2 + g.vegas_spread / 2, g.vegas_total / 2 - g.vegas_spread / 2]) - np.concatenate([g.home_score, g.away_score])
    s.append(acc(pd.Series(t), pd.Series(v), "team score"))
    summ = pd.DataFrame(s)
    print(f"\n=== OUR PROJECTED SCORES vs VEGAS, {season} weeks {weeks_done[0]}-{weeks_done[-1]} ({len(g)} games) ===")
    print(summ.round(2).to_string(index=False))
    print("(MAE/RMSE = typical miss in points; lower wins. ours_closer_share = share of games where we were nearer the real result.)")

    recs = []
    for label, kind, mask in (("OVER/UNDER, all games", "total", pd.Series(True, index=g.index)),
                              ("OVER/UNDER, we differ by 3+ pts", "total", g.big_total_gap),
                              ("AGAINST THE SPREAD, all games", "spread", pd.Series(True, index=g.index)),
                              ("AGAINST THE SPREAD, we differ by 3+ pts", "spread", g.big_spread_gap)):
        r = g.loc[mask, f"{kind}_result"]
        w, l, p = int((r == "win").sum()), int((r == "loss").sum()), int((r == "push").sum())
        recs.append({"bet": label, "wins": w, "losses": l, "pushes": p, "win_pct": w / (w + l) if w + l else np.nan,
                     "p_value_vs_50pct": binom_p(w, w + l)})
    rec = pd.DataFrame(recs)
    print("\nHow our side of each line would have done (break-even at -110 odds is 52.4%):")
    print(rec.round(3).to_string(index=False))
    print("(p_value_vs_50pct: chance of a record this lopsided from a pure coin flip; small numbers like <0.05 would "
          "suggest a real edge. With this few games, almost any record is within luck.)")

    by_week = g.groupby("week").agg(games=("home", "size"), ours_total_MAE=("err_total_ours", lambda e: e.abs().mean()),
                                    vegas_total_MAE=("err_total_vegas", lambda e: e.abs().mean()),
                                    ours_margin_MAE=("err_margin_ours", lambda e: e.abs().mean()),
                                    vegas_margin_MAE=("err_margin_vegas", lambda e: e.abs().mean()))
    print("\nBy week (MAE, points):")
    print(by_week.round(2).to_string())

    # one summary file with every table stacked
    out = pd.concat([summ.assign(table="accuracy"), rec.assign(table="record"),
                     by_week.reset_index().assign(table="by_week")], ignore_index=True)
    out.round(3).to_csv(out_dir / f"game_backtest_{season}_summary.csv", index=False)
    print(f"\nSaved {path.name} (one row per game) and game_backtest_{season}_summary.csv")


if __name__ == "__main__":
    main()
