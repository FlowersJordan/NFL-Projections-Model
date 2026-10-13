#!/usr/bin/env python3
"""
How accurate were last week's projections? Scores a SAVED projections file against what actually happened.

    python check_accuracy.py projections_2026_wk05.csv

It only scores games that are final AND have player stats posted on nflverse (stats usually lag the final
whistle by some hours), so run it again later and it picks up the games it had to skip.

What you get:
  1. Which games were scored, and which were skipped (and why)
  2. Points accuracy by position (error, bias, rank correlation) vs. what the historical backtest led us to expect
  3. Ranking value: actual points of the players we'd have started vs. a plain recent-average ranking
  4. Calibration: did bigger projections really score more?
  5. Stat-line accuracy (targets, catches, yards, TDs, ...) and where the points came from
  6. Injury checks: did p_play match who played, and did "teammate out" bumps show up?
  7. Biggest misses in both directions, and players we projected who recorded nothing
  8. DraftKings points accuracy (and, if the file has salaries, whether the "value" picks paid off)

Files written next to the script:
  accuracy_<season>_wk<week>.csv   one row per scored player: projection, actual, error, stat lines
  accuracy_log.csv                 one summary row per week (re-scoring a week replaces its row), so you can
                                   watch accuracy build over the season

IMPORTANT: score the file that was saved BEFORE the games. Re-running weekly_projections.py for a week that
has already started overwrites that file, so keep a copy of each week's projections.
Use the same --te-premium / --pass-td you projected with (defaults match the projections script).
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import nflreadpy as nfl

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 60)

POS = ["QB", "RB", "WR", "TE"]
TOPK = {"QB": 6, "RB": 12, "WR": 18, "TE": 8}          # same "start" sizes as the backtest
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# column name in the projections CSV -> nflverse weekly-stats column
STAT = {"pass_att": "attempts", "pass_yds": "passing_yards", "pass_td": "passing_tds",
        "int": "passing_interceptions", "car": "carries", "rush_yds": "rushing_yards",
        "rush_td": "rushing_tds", "tgt": "targets", "rec": "receptions", "rec_yds": "receiving_yards",
        "rec_td": "receiving_tds"}
MODELED = {"QB": ["pass_att", "pass_yds", "pass_td", "int", "car", "rush_yds", "rush_td"],
           "RB": ["car", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"],
           "WR": ["car", "rush_yds", "tgt", "rec", "rec_yds", "rec_td"],
           "TE": ["tgt", "rec", "rec_yds", "rec_td"]}
DK_EXTRA = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost", "passing_2pt_conversions",
            "rushing_2pt_conversions", "receiving_2pt_conversions", "special_teams_tds"]


# ----------------------------------------------------------------------------- helpers
def retry(fn, tries=3):
    for i in range(tries):                 # nflverse downloads occasionally time out
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise


def section(title):
    print(f"\n{'=' * 4} {title} {'=' * max(4, 100 - len(title))}")


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float))
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return a.rank().corr(b.rank())


def acc_table(df, y, p):
    """Error summary of column p (projection) against column y (actual), by position and overall."""
    rows = []
    for pos in POS + ["ALL"]:
        g = df if pos == "ALL" else df[df.pos == pos]
        if len(g) < 3:
            continue
        yy, e = g[y].to_numpy(float), (g[p] - g[y]).to_numpy(float)
        sst = ((yy - yy.mean()) ** 2).sum()
        rows.append({"pos": pos, "n": len(g), "avg_proj": g[p].mean(), "avg_actual": g[y].mean(),
                     "bias": e.mean(), "bias_se": e.std(ddof=1) / np.sqrt(len(e)), "mae": np.abs(e).mean(),
                     "rmse": np.sqrt((e ** 2).mean()), "r2": 1 - (e ** 2).sum() / sst if sst > 0 else np.nan,
                     "rank_corr": spearman(g[p], g[y])})
    return pd.DataFrame(rows)


def stat_line(r, pref=""):
    """Compact one-line stat line from a row, using projection columns (pref='') or actuals (pref='a_')."""
    g = lambda c: 0.0 if pd.isna(r[pref + c]) else float(r[pref + c])
    if pref == "" and all(pd.isna(r[c]) for c in STAT):
        return "n/a"
    if r["pos"] == "QB":
        return (f"{g('pass_att'):.0f} att {g('pass_yds'):.0f}y {g('pass_td'):.1f}TD {g('int'):.1f}int"
                f" | {g('car'):.0f}c {g('rush_yds'):.0f}y {g('rush_td'):.1f}TD")
    return (f"{g('tgt'):.1f}tgt {g('rec'):.1f}rec {g('rec_yds'):.0f}y {g('rec_td'):.1f}TD"
            f" | {g('car'):.0f}c {g('rush_yds'):.0f}y")


# ----------------------------------------------------------------------------- actual results
def load_game_status(season, week):
    sc = retry(lambda: nfl.load_schedules(seasons=season).to_pandas())
    sc = sc[(sc.game_type == "REG") & (sc.week == week)].copy()
    sc["home_team"], sc["away_team"] = sc.home_team.replace(TEAM_FIX), sc.away_team.replace(TEAM_FIX)
    sc["final"] = sc.home_score.notna() & sc.away_score.notna()
    return sc


def load_actuals(season, week, te_premium, pass_td):
    st = retry(lambda: nfl.load_player_stats(seasons=season).to_pandas())
    st = st[(st.season_type == "REG") & (st.week == week) & st.position.isin(POS)].copy().reset_index(drop=True)
    st["team"] = st.team.replace(TEAM_FIX)
    for c in list(STAT.values()) + DK_EXTRA + ["fantasy_points_ppr"]:
        st[c] = st[c].fillna(0).astype(float)
    a = pd.DataFrame({"player": st.player_display_name, "pos": st.position, "team": st.team})
    for short, col in STAT.items():
        a["a_" + short] = st[col]
    # league scoring: nflverse full-PPR points, plus the TE premium and any non-default passing-TD value
    a["a_pts"] = (st.fantasy_points_ppr + (pass_td - 4.0) * st.passing_tds
                  + te_premium * st.receptions * (st.position == "TE"))
    # DraftKings Classic: scored stats + fumbles/2pt/ST TDs + 3-pt bonuses at 300 pass / 100 rush / 100 rec yards
    bonus = 3.0 * ((st.passing_yards >= 300).astype(float) + (st.rushing_yards >= 100) + (st.receiving_yards >= 100))
    a["a_dk"] = (0.04 * st.passing_yards + 4 * st.passing_tds - st.passing_interceptions
                 + 0.1 * (st.rushing_yards + st.receiving_yards) + 6 * (st.rushing_tds + st.receiving_tds)
                 + st.receptions - (st.sack_fumbles_lost + st.rushing_fumbles_lost + st.receiving_fumbles_lost)
                 + 2 * (st.passing_2pt_conversions + st.rushing_2pt_conversions + st.receiving_2pt_conversions)
                 + 6 * st.special_teams_tds + bonus)
    return a


def attach_actuals(proj, act, scored_mask):
    """Left-join actuals onto projections by name+team+position, falling back to name+team if needed."""
    keys = ["player", "team", "pos"]
    m = proj.merge(act, on=keys, how="left")
    miss = scored_mask.to_numpy() & m.a_pts.isna().to_numpy()
    if miss.any():
        act2 = act.drop(columns="pos").drop_duplicates(["player", "team"], keep=False)
        fb = m.loc[miss, ["player", "team"]].reset_index().merge(act2, on=["player", "team"], how="left").set_index("index")
        for c in [c for c in act2.columns if c.startswith("a_")]:
            m.loc[fb.index, c] = fb[c]
    return m


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Score a saved weekly projections file against actual results")
    ap.add_argument("projections", help="the SAVED pre-game projections CSV, e.g. projections_2026_wk05.csv")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--te-premium", type=float, default=0.5, help="extra points per TE reception (default 0.5)")
    ap.add_argument("--pass-td", type=float, default=4.0, help="points per passing TD (default 4)")
    ap.add_argument("--backtest-summary", default="backtest_weekly_summary.csv")
    ap.add_argument("--backtest-players", default="backtest_weekly_players.csv")
    ap.add_argument("--no-log", action="store_true", help="don't update accuracy_log.csv")
    args = ap.parse_args()

    mt = re.search(r"(\d{4})_wk(\d{1,2})", os.path.basename(args.projections))
    season = args.season or (int(mt.group(1)) if mt else None)
    week = args.week or (int(mt.group(2)) if mt else None)
    if season is None or week is None:
        raise SystemExit("couldn't tell the season/week from the file name - pass --season and --week")

    proj = pd.read_csv(args.projections)
    need = {"player", "pos", "team", "proj"}
    if not need <= set(proj.columns):
        raise SystemExit(f"that file is missing columns: {sorted(need - set(proj.columns))}")
    proj["injury"] = proj["injury"].fillna("") if "injury" in proj else ""
    have_components = {"pts_yds", "pts_td"} <= set(proj.columns)      # projected points-by-source columns
    if "opp" not in proj:
        proj["opp"] = ""
    for c in STAT:                                                    # a file without stat lines still gets scored on points
        if c not in proj:
            proj[c] = np.nan
    if "p_play" not in proj:
        proj["p_play"] = 1.0
    if "proj_ev" not in proj:
        proj["proj_ev"] = proj.proj * proj.p_play
    for c in ("recent_avg", "form_only", "inj_team", "inj_own"):
        if c not in proj:
            proj[c] = np.nan

    # ---- which games can be scored?
    sc = load_game_status(season, week)
    act = load_actuals(season, week, args.te_premium, args.pass_td)
    posted = set(act.team)
    section(f"{season} WEEK {week}: WHICH GAMES COULD BE SCORED")
    rows, scored_teams = [], set()
    in_file = set(proj.team)
    for g in sc.sort_values(["gameday", "gametime"]).itertuples():
        has_stats = g.home_team in posted and g.away_team in posted
        status = ("scored" if g.final and has_stats else
                  "final, but nflverse hasn't posted player stats yet - rerun later" if g.final else "not final yet")
        if not ({g.home_team, g.away_team} & in_file):
            status += " (no players from this game in the projections file)"
        elif g.final and has_stats:
            scored_teams |= {g.home_team, g.away_team}
        score = f"{g.away_team} {g.away_score:.0f} - {g.home_team} {g.home_score:.0f}" if g.final else f"{g.away_team} @ {g.home_team}"
        rows.append({"game": score, "date": g.gameday, "status": status})
    print(pd.DataFrame(rows).to_string(index=False))

    scored = proj.team.isin(scored_teams)
    n_pending = int((~scored).sum())
    if not scored.any():
        raise SystemExit("\nNo games with posted stats yet - nothing to score. Try again later.")
    m = attach_actuals(proj, act, scored)
    df = m[scored].copy().reset_index(drop=True)
    df["played"] = df.a_pts.notna()               # False = projected, but no stat line recorded
    for c in [c for c in df.columns if c.startswith("a_")]:
        df[c] = df[c].fillna(0.0)
    df["err"] = df.proj - df.a_pts
    ngames = len(scored_teams) // 2
    print(f"\nScoring {len(df)} projected players from {ngames} games ({int(df.played.sum())} recorded a stat line, "
          f"{int((~df.played).sum())} did not)."
          + (f" Skipped {n_pending} players whose games aren't scored yet." if n_pending else ""))
    pl = df[df.played].copy()

    # ---- benchmark from the historical backtest
    bt_rmse, bt_rho, wk_hist = {}, {}, None
    if os.path.exists(args.backtest_players):
        b = pd.read_csv(args.backtest_players)
        b = b[~((b.season == season) & (b.week == week))]
        for pos in POS:
            g = b[b.position == pos]
            bt_rmse[pos] = float(np.sqrt(((g.proj_final - g.actual_pts) ** 2).mean()))
        bt_rmse["ALL"] = float(np.sqrt(((b.proj_final - b.actual_pts) ** 2).mean()))
    if os.path.exists(args.backtest_summary):
        w = pd.read_csv(args.backtest_summary)
        w = w[(w.n >= 30) & ~((w.season == season) & (w.week == week))]
        wk_hist = w
        for pos in POS:
            bt_rho[pos] = float(w[f"rho_{pos}"].mean())
        bt_rho["ALL"] = float(w.rho_final.mean())

    # ---- 1. headline accuracy
    section("1. HOW FAR OFF WERE THE POINT PROJECTIONS?  (players who recorded a stat line; 'proj' = points if he plays)")
    t = acc_table(pl, "a_pts", "proj")
    if bt_rmse:
        t["backtest_rmse"] = t.pos.map(bt_rmse)
    if bt_rho:
        t["backtest_rank_corr"] = t.pos.map(bt_rho)
    print(t.round(2).to_string(index=False))
    print("bias = projection minus actual (positive = we were too high). bias_se is its standard error: a bias "
          "within about +/-2 x bias_se is just noise.")
    print("rank_corr = Spearman correlation of projected vs actual points (1 = perfect order). backtest_* columns "
          "show the typical value over 2022-2026 for comparison.")
    if wk_hist is not None and len(wk_hist):
        cur_rmse = float(np.sqrt((pl.err ** 2).mean()))
        cur_rho = spearman(pl.proj, pl.a_pts)
        worse = float((wk_hist.rmse >= cur_rmse).mean())          # share of past weeks with a larger error
        below = float((wk_hist.rho_final <= cur_rho).mean())      # share of past weeks with a weaker ranking
        print(f"\nHow unusual is this week?  RMSE {cur_rmse:.2f} vs typical week {wk_hist.rmse.mean():.2f} "
              f"(range {wk_hist.rmse.min():.2f}-{wk_hist.rmse.max():.2f}); {worse:.0%} of past weeks had a larger error.")
        print(f"                           rank corr {cur_rho:.2f} vs typical {wk_hist.rho_final.mean():.2f} "
              f"(range {wk_hist.rho_final.min():.2f}-{wk_hist.rho_final.max():.2f}); {below:.0%} of past weeks ranked worse.")
        print(f"Only {ngames} of the week's games are in this sample, so it is noisier than a full week "
              f"(typical full week: ~200 players).")

    section("1b. THE SAME, COUNTING PLAYERS WHO RECORDED NOTHING AS 0 POINTS  (proj_ev = chance he plays x proj)")
    t2 = acc_table(df, "a_pts", "proj_ev")
    print(t2.round(2).to_string(index=False))
    tt = acc_table(df, "a_pts", "recent_avg") if df.recent_avg.notna().all() else None
    if tt is not None:
        print("\nFor reference, simply using each player's recent average (no matchup, no injuries):")
        print(tt.round(2).to_string(index=False))

    # ---- 2. ranking value
    section("2. START/SIT VALUE: actual points of the players each ranking would have started (this week)")
    ranks = {"ours (proj_ev)": "proj_ev", "recent avg": "recent_avg", "form-only model": "form_only"}
    rows = []
    for pos in POS:
        g = df[df.pos == pos]
        k = min(TOPK[pos], len(g))
        if k < 3:
            continue
        hind = g.nlargest(k, "a_pts")
        r = {"pos": pos, "K": k, "pool": len(g), "pool_avg": g.a_pts.mean()}
        for name, col in ranks.items():
            if g[col].notna().all():
                top = g.nlargest(k, col)
                r[name] = top.a_pts.mean()
        r["hindsight_best"] = hind.a_pts.mean()
        r["our_top_that_were_actual_top"] = len(set(g.nlargest(k, "proj_ev").index) & set(hind.index))
        rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("(avg actual points per player started. 'recent avg' ignores matchup AND injuries, so a player who sat "
          "counts as 0 there. One week is a small sample: treat gaps under ~1 point as noise.)")

    # ---- 3. calibration
    section("3. CALIBRATION: did bigger projections score more?  (all scored players, non-players = 0 points)")
    q = pd.qcut(df.proj_ev, 5, duplicates="drop")
    cal = df.groupby(q, observed=True).agg(n=("a_pts", "size"), avg_proj_ev=("proj_ev", "mean"), avg_actual=("a_pts", "mean"))
    cal["diff"] = cal.avg_proj_ev - cal.avg_actual
    cal.index = [f"{iv.left:.1f} to {iv.right:.1f}" for iv in cal.index]
    print(cal.round(2).to_string())

    # ---- 4. stat lines
    section("4. STAT-LINE ACCURACY  (players who recorded a stat line)")
    rows = []
    for pos in POS:
        g = pl[pl.pos == pos]
        if len(g) < 3:
            continue
        for s in MODELED[pos]:
            p, y = g[s], g["a_" + s]
            if p.isna().all():
                continue
            rows.append({"pos": pos, "stat": s, "n": len(g), "avg_proj": p.mean(), "avg_actual": y.mean(),
                         "mae": (p - y).abs().mean(),
                         "corr": p.corr(y) if p.std() > 0 and y.std() > 0 else np.nan})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    section("4b. WHERE THE POINTS CAME FROM  (average per player: projected vs actual)")
    if not have_components:
        print("(this projections file has no pts_rec / pts_yds / pts_td columns, so there is no breakdown to check)")
    if have_components:
        w_rec = np.where(pl.pos == "TE", 1 + args.te_premium, 1.0)
        comp = pd.DataFrame({"pos": pl.pos,
                             "rec_proj": pl.get("pts_rec", pd.Series(0.0, index=pl.index)).fillna(0.0),
                             "rec_actual": w_rec * pl.a_rec,
                             "yards_proj": pl.get("pts_yds", 0.0), "yards_actual": 0.04 * pl.a_pass_yds + 0.1 * (pl.a_rush_yds + pl.a_rec_yds),
                             "td_proj": pl.get("pts_td", 0.0), "td_actual": args.pass_td * pl.a_pass_td + 6 * (pl.a_rush_td + pl.a_rec_td)})
        comp["other_proj"] = pl.proj - comp.rec_proj - comp.yards_proj - comp.td_proj
        comp["other_actual"] = pl.a_pts - comp.rec_actual - comp.yards_actual - comp.td_actual
        cg = comp.groupby("pos").mean(numeric_only=True)
        cg.loc["ALL"] = comp.drop(columns="pos").mean()
        out = []
        for pos, r in cg.iterrows():
            for k in ("rec", "yards", "td", "other"):
                out.append({"pos": pos, "source": k, "proj": r[f"{k}_proj"], "actual": r[f"{k}_actual"],
                            "actual-proj": r[f"{k}_actual"] - r[f"{k}_proj"]})
        print(pd.DataFrame(out).round(2).to_string(index=False))
        print("('other' = interceptions, fumbles lost, 2-pt conversions. Touchdowns are mostly luck: a big TD gap is "
              "usually variance, not a modeling error.)")


    # ---- 5. injuries
    section("5. INJURY CHECKS")
    df["flag"] = df.injury.replace("", "(no flag)")
    g = df.groupby("flag").agg(n=("a_pts", "size"), avg_p_play=("p_play", "mean"), share_recorded_stats=("played", "mean"),
                               avg_proj_ev=("proj_ev", "mean"), avg_actual_pts=("a_pts", "mean"))
    print("Did p_play match who actually recorded a stat line?  (avg_p_play should be near share_recorded_stats)")
    print(g.sort_values("n", ascending=False).round(2).to_string())
    print(f"\nOverall: average p_play {df.p_play.mean():.2f} vs {df.played.mean():.2f} who recorded a stat line.")
    flagged = df[df.played & (df.flag != "(no flag)")]
    if len(flagged) >= 3:
        e = flagged.proj - flagged.a_pts
        print(f"Hurt-but-played ({len(flagged)} players with an injury flag who played): avg proj {flagged.proj.mean():.2f}, "
              f"avg actual {flagged.a_pts.mean():.2f}, bias {e.mean():+.2f}")
    if df.inj_team.notna().any():
        pb = df[df.played & df.inj_team.notna()].copy()
        pb["teammates_out_bump"] = pd.cut(pb.inj_team, [-100, -0.3, 0.3, 1.0, 100],
                                          labels=["drag (< -0.3)", "none", "small (0.3 to 1)", "bump (1+)"])
        tb = pb.groupby("teammates_out_bump", observed=True).agg(n=("a_pts", "size"), avg_inj_team=("inj_team", "mean"),
                                                                  avg_proj=("proj", "mean"), avg_actual=("a_pts", "mean"))
        tb["bias"] = tb.avg_proj - tb.avg_actual
        print("\nTeammates-out adjustment (inj_team = points added to a player's projection because teammates were out):")
        print(tb.round(2).to_string())
        top = pb.sort_values("inj_team", ascending=False).head(8)
        top = top.assign(actual_vs_proj=top.a_pts - top.proj)
        print("\nBiggest teammates-out bumps and what happened:")
        print(top[["player", "pos", "team", "inj_team", "proj", "a_pts", "actual_vs_proj"]].round(2).to_string(index=False))

    # ---- 6. biggest misses
    section("6. BIGGEST MISSES  (players who recorded a stat line)")
    pl = df[df.played].copy()
    pl["proj_line"] = pl.apply(lambda r: stat_line(r), axis=1)
    pl["actual_line"] = pl.apply(lambda r: stat_line(r, "a_"), axis=1)
    cols = ["player", "pos", "team", "opp", "proj", "a_pts", "err", "proj_line", "actual_line"]
    print("We were too HIGH (projected minus actual):")
    print(pl.sort_values("err", ascending=False).head(10)[cols].round(1).to_string(index=False))
    print("\nWe were too LOW:")
    print(pl.sort_values("err").head(10)[cols].round(1).to_string(index=False))
    ns = df[~df.played].sort_values("proj_ev", ascending=False).head(12)
    if len(ns):
        print("\nProjected but recorded NO stat line (inactive, benched, or a game-day scratch):")
        print(ns[["player", "pos", "team", "opp", "p_play", "proj", "proj_ev", "injury"]].round(2).to_string(index=False))

    # ---- 7. DraftKings
    if "dk_proj" in df.columns and df.dk_proj.notna().any():
        section("7. DRAFTKINGS POINTS  (players who recorded a stat line)")
        pl = df[df.played]
        print(acc_table(pl, "a_dk", "dk_proj").round(2).to_string(index=False))
        if "salary" in df.columns and df.salary.notna().any():
            v = df[df.salary.notna()].copy()
            v["actual_per_k"] = v.a_dk / (v.salary / 1000)
            v["hit_3x"] = (v.a_dk >= 3 * v.salary / 1000).astype(float)
            v["tier"] = pd.qcut(v.value_pts.rank(method="first", ascending=False), 4,
                                labels=["best value (top 25%)", "2nd", "3rd", "worst value"])
            tv = v.groupby("tier", observed=True).agg(n=("a_dk", "size"), avg_salary=("salary", "mean"),
                                                      avg_actual_dk=("a_dk", "mean"), actual_pts_per_k=("actual_per_k", "mean"),
                                                      hit_3x_salary=("hit_3x", "mean"))
            print("\nDid the 'value' ranking pay off? (value_pts quartiles; hit_3x = scored 3x his salary in $1,000s, "
                  "the usual benchmark for a useful DFS player)")
            print(tv.round(2).to_string())
    else:
        print("\n(No DraftKings columns in that file, so no DraftKings check.)")

    # ---- save
    save = df[["player", "pos", "team", "opp", "injury", "p_play", "recent_avg", "form_only", "proj", "proj_ev",
               "played", "a_pts", "err"]].copy()
    save["proj_line"], save["actual_line"] = df.apply(lambda r: stat_line(r), axis=1), df.apply(lambda r: stat_line(r, "a_"), axis=1)
    for c in ("dk_proj", "dk_ev", "salary", "value_pts"):
        if c in df.columns:
            save[c] = df[c]
    save["a_dk"] = df.a_dk
    save = save.rename(columns={"a_pts": "actual_pts", "err": "proj_minus_actual", "a_dk": "actual_dk_pts"})
    path = f"accuracy_{season}_wk{week:02d}.csv"
    save.sort_values("proj_ev", ascending=False).round(3).to_csv(path, index=False)
    print(f"\nSaved {len(save)} scored players to {path}")

    if not args.no_log:
        pl = df[df.played]
        a_all = acc_table(pl, "a_pts", "proj").set_index("pos")
        row = {"season": season, "week": week, "games_scored": ngames, "n_scored": len(df), "n_played": len(pl),
               "rmse": a_all.rmse["ALL"], "mae": a_all.mae["ALL"], "bias": a_all.bias["ALL"],
               "rank_corr": a_all.rank_corr["ALL"]}
        for pos in POS:
            row[f"rank_corr_{pos}"] = a_all.rank_corr.get(pos, np.nan)
        row["avg_p_play"], row["share_recorded_stats"] = df.p_play.mean(), df.played.mean()
        if "dk_proj" in pl and pl.dk_proj.notna().any():
            row["dk_rmse"] = float(np.sqrt(((pl.dk_proj - pl.a_dk) ** 2).mean()))
            row["dk_bias"] = float((pl.dk_proj - pl.a_dk).mean())
        log = pd.read_csv("accuracy_log.csv") if os.path.exists("accuracy_log.csv") else pd.DataFrame()
        if len(log):
            log = log[~((log.season == season) & (log.week == week))]
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True).sort_values(["season", "week"])
        log.round(4).to_csv("accuracy_log.csv", index=False)
        print("Updated accuracy_log.csv (one row per week; re-scoring a week replaces its row).")


if __name__ == "__main__":
    main()