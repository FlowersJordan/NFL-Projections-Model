#!/usr/bin/env python3
"""
Weekly projected STAT LINES and expected fantasy points (TE-premium, full PPR),
using ONLY pre-game information.

Every QB / RB / WR / TE gets a projected stat line, and fantasy points are computed from it:

    receptions, receiving yards, receiving TDs, targets, carries, rushing yards, rushing TDs
    (QBs: pass attempts, passing yards, passing TDs, interceptions, plus their rushing)

    projected points = scoring weights x projected stats  (+ a small average for fumbles/2pt)

Each stat is its own regression on:
    the player's recent form in that stat (EWMA)           <- who the player is
    his recent usage (targets / carries / attempts)        <- role
    how much the opponent over/under-allows THAT stat      <- the matchup
      to the position (last 8 games, shrunk toward 0)
    team implied total, team spread, home/away             <- game environment
Touchdowns and interceptions use Poisson regression (counts); everything else uses OLS.

Usage:
    python weekly_projections.py                    # next unplayed week of the current season
    python weekly_projections.py --week 6           # a specific week (needs posted lines)
    python weekly_projections.py --backtest         # also run the week-by-week backtest (~2-3 min)
    python weekly_projections.py --te-premium 1.0   # 2.0 PPR for TEs, etc.
    python weekly_projections.py --pass-td 6        # 6-pt passing TDs
    python weekly_projections.py --sleeper          # add Sleeper's projections as a column
    python weekly_projections.py --compare-csv espn.csv --compare-name ESPN
                                                    # any other source: CSV with columns
                                                    # player, proj [, pos, team]

Everything uses data available before kickoff, so backtest numbers are honest.
QBs are limited to the probable starter listed in the nflverse schedule; players listed
"Out" on the injury report are dropped, and other designations are shown in the output.
Known gaps: no depth-chart/snap-count info, no weather, and players with <4 prior games
(rookies, new starters) get no projection. The backtest scores players who actually played,
so it does not test the "will he play?" question (use the injury report for that).
"""
import argparse
import re
import warnings

import numpy as np
import pandas as pd
import nflreadpy as nfl
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

POS = ["QB", "RB", "WR", "TE"]
FIRST_SEASON = 2018
MIN_PRIOR_GAMES = 4                                   # need this much history to project
MIN_BASELINE = {"QB": 10.0, "RB": 5.0, "WR": 5.0, "TE": 4.0}   # fantasy-relevant players only
HALFLIFE = 4                                          # games; weight of recent form
DEF_WINDOW = 8                                        # games used for defense rating
DEF_SHRINK = 4                                        # shrink defense rating toward 0
TOPK = {"QB": 6, "RB": 12, "WR": 18, "TE": 8}         # backtest "start" sizes
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
CTX = ["def_rel", "team_implied", "team_spread", "is_home"]   # features of the direct points model

CFG = {"te_premium": 0.5, "pass_td": 4.0}             # set from the command line in main()

# ---- which stats are modeled for each position (anything else falls into the small "other" bucket)
_RECV = ["targets", "receptions", "receiving_yards", "receiving_tds"]
MODEL_STATS = {
    "QB": ["attempts", "passing_yards", "passing_tds", "passing_interceptions",
           "carries", "rushing_yards", "rushing_tds"],
    "RB": ["carries", "rushing_yards", "rushing_tds"] + _RECV,
    "WR": ["carries", "rushing_yards"] + _RECV,
    "TE": list(_RECV),
}
STAT_COLS = ["attempts", "passing_yards", "passing_tds", "passing_interceptions",
             "carries", "rushing_yards", "rushing_tds",
             "targets", "receptions", "receiving_yards", "receiving_tds"]
USAGE_DRIVER = {"passing_yards": "attempts", "passing_tds": "attempts", "passing_interceptions": "attempts",
                "rushing_yards": "carries", "rushing_tds": "carries",
                "receptions": "targets", "receiving_yards": "targets", "receiving_tds": "targets"}
COUNT_STATS = {"passing_tds", "passing_interceptions", "rushing_tds", "receiving_tds"}   # Poisson
SHORT = {"attempts": "pass_att", "passing_yards": "pass_yds", "passing_tds": "pass_td",
         "passing_interceptions": "int", "carries": "car", "rushing_yards": "rush_yds",
         "rushing_tds": "rush_td", "targets": "tgt", "receptions": "rec",
         "receiving_yards": "rec_yds", "receiving_tds": "rec_td"}


def score_weights(pos):
    """Fantasy points per unit of each stat (full PPR, TE premium on TE receptions)."""
    return {"passing_yards": 0.04, "passing_tds": CFG["pass_td"], "passing_interceptions": -2.0,
            "rushing_yards": 0.1, "rushing_tds": 6.0,
            "receptions": 1.0 + (CFG["te_premium"] if pos == "TE" else 0.0),
            "receiving_yards": 0.1, "receiving_tds": 6.0}


# ----------------------------------------------------------------------------- data
def load_data(season, te_premium, pass_td=4.0):
    seasons = list(range(FIRST_SEASON, season + 1))

    stats = nfl.load_player_stats(seasons=seasons).to_pandas()
    stats = stats[(stats.season_type == "REG") & stats.position.isin(POS)].copy()
    stats["team"] = stats.team.replace(TEAM_FIX)
    for c in STAT_COLS:
        stats[c] = stats[c].fillna(0)
    # nflverse fantasy_points_ppr = scored stats (1 pt/rec, 4-pt pass TD) + fumbles/2pt/ST TDs.
    # Add the TE premium and any non-default passing-TD value on top.
    stats["pts"] = (stats.fantasy_points_ppr
                    + (pass_td - 4.0) * stats.passing_tds
                    + te_premium * stats.receptions * (stats.position == "TE"))
    stats = stats[["player_id", "player_display_name", "position", "season", "week",
                   "team", "pts"] + STAT_COLS]

    sched = nfl.load_schedules(seasons=seasons).to_pandas()
    sched = sched[sched.game_type == "REG"].copy()
    sched["home_team"] = sched.home_team.replace(TEAM_FIX)
    sched["away_team"] = sched.away_team.replace(TEAM_FIX)
    sched["gameday"] = pd.to_datetime(sched.gameday)
    sched["played"] = sched.home_score.notna()

    # one row per team per game, from that team's point of view
    # nflverse spread_line: positive = home team favored
    home = sched.assign(team=sched.home_team, opp=sched.away_team, is_home=1,
                        team_spread=sched.spread_line, team_qb_id=sched.home_qb_id)
    away = sched.assign(team=sched.away_team, opp=sched.home_team, is_home=0,
                        team_spread=-sched.spread_line, team_qb_id=sched.away_qb_id)
    games = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "week", "gameday", "played", "team", "opp", "is_home",
         "team_spread", "total_line", "team_qb_id"]]
    games["team_implied"] = games.total_line / 2 + games.team_spread / 2
    return stats, games


def build_frame(stats, games, season, week, max_gap_days):
    """History rows (played games) + one 'upcoming' row per active player for the target week."""
    hist = stats.merge(games, on=["season", "week", "team"], how="inner")
    hist = hist.dropna(subset=["team_spread", "total_line"])
    hist["upcoming"] = False

    last = (hist.sort_values(["season", "week"]).groupby("player_id").tail(1)
            [["player_id", "player_display_name", "position", "team", "gameday"]]
            .rename(columns={"gameday": "last_gameday"}))
    ug = games[(games.season == season) & (games.week == week) & (~games.played)]
    up = last.merge(ug, on="team")
    up = up[(up.gameday - up.last_gameday).dt.days <= max_gap_days].dropna(
        subset=["team_spread", "total_line"])
    # QBs: only the probable starter listed in the schedule (when one is listed)
    keep = (up.position != "QB") | up.team_qb_id.isna() | (up.player_id == up.team_qb_id)
    up = up[keep]
    up["season"], up["week"] = season, week
    for c in ["pts"] + STAT_COLS:
        up[c] = np.nan
    up["upcoming"] = True

    d = pd.concat([hist, up[hist.columns]], ignore_index=True)
    return d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def add_features(d, games, season, week):
    cols = ["pts"] + STAT_COLS

    # ---- recent form: EWMA of each stat over PRIOR games only (shift(1) first)
    sh = d.groupby("player_id")[cols].shift(1)
    ew = (sh.groupby(d.player_id)
            .ewm(halflife=HALFLIFE, min_periods=MIN_PRIOR_GAMES).mean()
            .droplevel(0).reindex(d.index))
    for c in cols:
        d["e_" + c] = ew[c]
    d["pts_ewma"] = d.e_pts
    d["tgt_ewma"] = d.e_targets
    d["n_prior"] = d.groupby("player_id").cumcount()
    d["elig"] = (d.n_prior >= MIN_PRIOR_GAMES) & (d.pts_ewma >= d.position.map(MIN_BASELINE))

    # ---- defense-vs-position ratings: prior-8-game average of (stat above player's own norm) allowed
    for c in cols:
        d["r_" + c] = d[c] - d["e_" + c]
    d["resid"] = d.r_pts
    rc = ["r_" + c for c in cols]

    played_rows = d[d.elig & ~d.upcoming]
    allowed = (played_rows.groupby(["season", "week", "opp", "position"])[rc].sum()
               .reset_index().rename(columns={"opp": "def_team"}))

    dg = games[(games.season < season) | ((games.season == season) & (games.week <= week))]
    dg = dg[dg.played | ((dg.season == season) & (dg.week == week))]
    dg = dg[["season", "week", "opp", "played"]].drop_duplicates().rename(columns={"opp": "def_team"})
    dg = dg.merge(pd.DataFrame({"position": POS}), how="cross")
    dg = dg.merge(allowed, on=["season", "week", "def_team", "position"], how="left")
    for c in rc:
        dg[c] = np.where(dg.played, dg[c].fillna(0.0), np.nan)
    dg = dg.sort_values(["def_team", "position", "season", "week"])

    grp = dg.groupby(["def_team", "position"])
    for c in cols:
        g = grp["r_" + c]
        roll_sum = g.transform(lambda s: s.shift(1).rolling(DEF_WINDOW, min_periods=1).sum())
        roll_cnt = g.transform(lambda s: s.shift(1).rolling(DEF_WINDOW, min_periods=1).count())
        dg["def_" + c] = (roll_sum / (roll_cnt + DEF_SHRINK)).fillna(0.0)

    dcols = ["def_" + c for c in cols]
    d = d.merge(dg[["season", "week", "def_team", "position"] + dcols],
                left_on=["season", "week", "opp", "position"],
                right_on=["season", "week", "def_team", "position"], how="left")
    d[dcols] = d[dcols].fillna(0.0)
    d["def_rel"] = d.def_pts
    return d.drop(columns="def_team")


# ----------------------------------------------------------------------------- direct points model
def feature_list(pos, ctx=True):
    f = ["pts_ewma"] + ([] if pos == "QB" else ["tgt_ewma"])
    return f + (CTX if ctx else [])


def fit_models(train, ctx=True, robust=False):
    fits = {}
    for pos in POS:
        t = train[train.position == pos].dropna(subset=feature_list(pos, ctx) + ["pts"])
        m = smf.ols("pts ~ " + " + ".join(feature_list(pos, ctx)), data=t)
        if robust:   # players in the same game share a matchup -> cluster by game
            fits[pos] = m.fit(cov_type="cluster",
                              cov_kwds={"groups": t.game_id.astype("category").cat.codes.values})
        else:
            fits[pos] = m.fit()
    return fits


def predict(fits, df, ctx=True):
    out = pd.Series(np.nan, index=df.index)
    for pos in POS:
        m = df.position == pos
        if m.any():
            x = df.loc[m, feature_list(pos, ctx)]
            ok = x.notna().all(axis=1)
            out.loc[x.index[ok]] = fits[pos].predict(x[ok])
    return out


# ----------------------------------------------------------------------------- stat-line models
def stat_features(s, ctx=True):
    f = ["e_" + s]
    if s in USAGE_DRIVER:
        f.append("e_" + USAGE_DRIVER[s])
    if ctx:
        f += ["def_" + s, "team_implied", "team_spread", "is_home"]
    return f


def fit_stat_models(train, ctx=True):
    """One regression per (position, stat). Counts (TDs, INTs) are Poisson, the rest OLS."""
    fits = {}
    for pos in POS:
        t0 = train[train.position == pos]
        for s in MODEL_STATS[pos]:
            feats = stat_features(s, ctx)
            t = t0.dropna(subset=feats + [s])
            X = sm.add_constant(t[feats], has_constant="add")
            if s in COUNT_STATS:
                res = sm.GLM(t[s], X, family=sm.families.Poisson()).fit(maxiter=100)
            else:
                res = sm.OLS(t[s], X).fit()
            fits[(pos, s)] = (res, feats, s in COUNT_STATS)
    return fits


def predict_stats(fits, df, prefix):
    """Projected stat columns '<prefix><stat>' (NaN where that stat isn't modeled for the position)."""
    cols = {prefix + s: pd.Series(np.nan, index=df.index) for s in STAT_COLS}
    for pos in POS:
        sub = df[df.position == pos]
        if sub.empty:
            continue
        for s in MODEL_STATS[pos]:
            res, feats, is_count = fits[(pos, s)]
            x = sub[feats]
            ok = x.notna().all(axis=1)
            if not ok.any():
                continue
            p = np.asarray(res.predict(sm.add_constant(x[ok], has_constant="add")), dtype=float)
            if not is_count:
                p = np.clip(p, 0, None)
            cols[prefix + s].loc[x.index[ok]] = p
    return pd.DataFrame(cols)


def stat_points(df, pos, prefix=""):
    """Points implied by the stat columns (prefix '' = actual stats, 'p_' = projected), by source."""
    ms, w = MODEL_STATS[pos], score_weights(pos)
    g = lambda s: df[prefix + s] if s in ms else 0.0
    out = pd.DataFrame(index=df.index)
    out["pts_rec"] = w["receptions"] * g("receptions")
    out["pts_yds"] = (w["passing_yards"] * g("passing_yards") + w["rushing_yards"] * g("rushing_yards")
                      + w["receiving_yards"] * g("receiving_yards"))
    out["pts_td"] = (w["passing_tds"] * g("passing_tds") + w["rushing_tds"] * g("rushing_tds")
                     + w["receiving_tds"] * g("receiving_tds"))
    out["pts_int"] = w["passing_interceptions"] * g("passing_interceptions")
    return out


def other_means(train):
    """Average points per game NOT explained by the modeled stats (fumbles lost, 2pt, ST TDs, ...)."""
    out = {}
    for pos in POS:
        t = train[train.position == pos]
        out[pos] = float((t.pts - stat_points(t, pos).sum(axis=1)).mean())
    return out


def project_points(df, prefix, other_mean):
    """Projected fantasy points = scoring weights x projected stats (+ average 'other')."""
    parts = []
    for pos in POS:
        sub = df[df.position == pos]
        sp = stat_points(sub, pos, prefix)
        sp["pts_other"] = other_mean[pos]
        sp["proj"] = sp[["pts_rec", "pts_yds", "pts_td", "pts_int", "pts_other"]].sum(axis=1, skipna=False)
        parts.append(sp)
    return pd.concat(parts).reindex(df.index)


# ----------------------------------------------------------------------------- weekly backtest
def backtest_weekly(d, first_test=2022):
    """Walk forward one week at a time: fit on every earlier week, predict that week."""
    pool = d[d.elig & ~d.upcoming].copy()
    pool["tkey"] = pool.season * 100 + pool.week
    keys = [k for k in sorted(pool.tkey.unique()) if k // 100 >= first_test]
    parts, seen = [], set()
    for k in keys:
        tr, te = pool[pool.tkey < k], pool[pool.tkey == k].copy()
        full, base = fit_stat_models(tr, True), fit_stat_models(tr, False)
        om = other_means(tr)
        te = te.join(predict_stats(full, te, "p_")).join(predict_stats(base, te, "b_"))
        te["proj_stats"] = project_points(te, "p_", om).proj
        te["base_stats"] = project_points(te, "b_", om).proj
        te["proj_direct"] = predict(fit_models(tr, True), te, True)
        te["base_direct"] = predict(fit_models(tr, False), te, False)
        parts.append(te)
        if k // 100 not in seen:
            seen.add(k // 100)
            print(f"  backtesting {k // 100} ...", flush=True)
    bt = pd.concat(parts)
    return bt.dropna(subset=["proj_stats", "base_stats", "proj_direct", "base_direct"])


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def weekly_table(bt):
    rows = []
    for (s, w), g in bt.groupby(["season", "week"]):
        r = {"season": int(s), "week": int(w), "n": len(g), "rmse": _rmse(g.pts, g.proj_stats),
             "bias": float((g.proj_stats - g.pts).mean())}
        for k, col in (("stats", "proj_stats"), ("base", "base_stats"), ("direct", "proj_direct")):
            r[f"rho_{k}"] = g[col].corr(g.pts, method="spearman") if len(g) >= 30 else np.nan
        for pos in POS:
            gp = g[g.position == pos]
            r[f"rho_{pos}"] = gp.proj_stats.corr(gp.pts, method="spearman") if len(gp) >= 10 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def report_weekly_backtest(bt, season):
    tag = f"{int(bt.season.min())}-{int(bt.season.max())}"
    wk_all = weekly_table(bt)
    wk = wk_all[wk_all.n >= 30]            # skip partial weeks (e.g. only a Thursday game played so far)
    print(f"\n=== WEEK-BY-WEEK BACKTEST ({tag}: {len(wk)} complete weeks, {len(bt):,} player-games) ===")
    print("Each week is predicted using ONLY earlier weeks (models refit weekly).")
    print("stats = stat-line projection | base = form+usage only | direct = one regression on points\n")

    print("--- 1. Fantasy-points accuracy ---")
    rows = []
    for pos in POS + ["ALL"]:
        b = bt if pos == "ALL" else bt[bt.position == pos]
        r = {"pos": pos, "n": len(b), "avg_actual": b.pts.mean(), "bias_stats": (b.proj_stats - b.pts).mean()}
        for k, col in (("stats", "proj_stats"), ("base", "base_stats"), ("direct", "proj_direct")):
            r[f"rmse_{k}"], r[f"r2_{k}"] = _rmse(b.pts, b[col]), _r2(b.pts, b[col])
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- 2. Accuracy of each projected stat (full model vs form+usage only) ---")
    rows = []
    for pos in POS:
        b = bt[bt.position == pos]
        for s in MODEL_STATS[pos]:
            y, p, q = b[s], b["p_" + s], b["b_" + s]
            rows.append({"pos": pos, "stat": SHORT[s], "avg_actual": y.mean(), "avg_pred": p.mean(),
                         "rmse_base": _rmse(y, q), "rmse_full": _rmse(y, p),
                         "r2_base": _r2(y, q), "r2_full": _r2(y, p)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- 3. Ranking skill by season: average weekly Spearman rank correlation with actual points ---")
    by = wk.groupby("season").agg(weeks=("week", "size"), rmse=("rmse", "mean"), bias=("bias", "mean"),
                                  rho_stats=("rho_stats", "mean"), rho_base=("rho_base", "mean"),
                                  rho_direct=("rho_direct", "mean"), rho_QB=("rho_QB", "mean"),
                                  rho_RB=("rho_RB", "mean"), rho_WR=("rho_WR", "mean"),
                                  rho_TE=("rho_TE", "mean"))
    print(by.round(3).to_string())

    print(f"\n--- 4. Week by week, {season} (a week with n < 30 is only partly played, so it has no rank score) ---")
    cur = wk_all[wk_all.season == season].drop(columns="season")
    print(cur.round(3).to_string(index=False))

    print("\n--- 5. Start/sit value: actual points of the top-K players each model would start (per week) ---")
    out = []
    for (s, w, pos), g in bt.groupby(["season", "week", "position"]):
        k = TOPK[pos]
        if len(g) >= 2 * k:
            out.append((pos, g.nlargest(k, "proj_stats").pts.mean(), g.nlargest(k, "base_stats").pts.mean(),
                        g.nlargest(k, "pts").pts.mean()))
    o = pd.DataFrame(out, columns=["pos", "stats", "base", "hindsight_best"])
    o["gain"] = o.stats - o.base
    t = o.groupby("pos").agg(weeks=("gain", "size"), stats=("stats", "mean"), base=("base", "mean"),
                             gain=("gain", "mean"), se=("gain", lambda x: x.std() / np.sqrt(len(x))),
                             hindsight_best=("hindsight_best", "mean"))
    t["t"] = t.gain / t.se
    t["K"] = t.index.map(TOPK)
    print(t.round(3).to_string())

    print("\n--- 6. Calibration: projected vs actual points by projection decile (all positions) ---")
    c = bt.assign(decile=pd.qcut(bt.proj_stats, 10, labels=False, duplicates="drop") + 1)
    cal = c.groupby("decile").agg(n=("pts", "size"), avg_proj=("proj_stats", "mean"), avg_actual=("pts", "mean"))
    cal["diff"] = cal.avg_proj - cal.avg_actual
    print(cal.round(2).to_string())
    return wk_all


def save_backtest(bt, wk, season):
    keep = (["season", "week", "player_display_name", "position", "team", "opp", "pts", "proj_stats",
             "base_stats", "proj_direct"] + STAT_COLS + ["p_" + s for s in STAT_COLS])
    keep = [c for c in keep if c in bt.columns]
    bt[keep].rename(columns={"player_display_name": "player", "pts": "actual_pts",
                             **{s: "actual_" + SHORT[s] for s in STAT_COLS},
                             **{"p_" + s: "proj_" + SHORT[s] for s in STAT_COLS}}
                    ).round(3).to_csv("backtest_weekly_players.csv", index=False)
    wk.round(4).to_csv("backtest_weekly_summary.csv", index=False)
    print("\nSaved backtest_weekly_players.csv (every player-week: projected vs actual stat line) "
          "and backtest_weekly_summary.csv (one row per week)")


def report_coefficients(fits):
    print("\n=== DIRECT POINTS MODEL: what it learned (fit on all history; game-clustered p-values) ===")
    print("Coefficients are fantasy points per 1-unit change, holding everything else fixed.\n")
    rows = []
    for pos in POS:
        f = fits[pos]
        for v in CTX + ["pts_ewma"] + ([] if pos == "QB" else ["tgt_ewma"]):
            rows.append({"pos": pos, "feature": v, "coef": f.params[v], "p": f.pvalues[v]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def report_te_stat_coefficients(full):
    print("\n=== TIGHT END STAT MODELS: what drives each stat (receiving TDs are on a log scale) ===")
    rows = []
    for s in MODEL_STATS["TE"]:
        res, feats, _ = full[("TE", s)]
        for v in feats:
            rows.append({"stat": SHORT[s], "feature": v, "coef": res.params[v], "p": res.pvalues[v]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# ----------------------------------------------------------------------------- other sources
SLEEPER_URL = ("https://api.sleeper.com/projections/nfl/{season}/{week}"
               "?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE")
SLEEPER_TEAM_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV"}


def name_key(s):
    """'Marvin Harrison Jr.' / 'A.J. Brown' / "Ja'Marr Chase" -> comparable lowercase keys."""
    s = str(s).lower().replace("-", " ").replace(".", "")
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_sleeper(items, te_premium):
    """Sleeper projection JSON -> DataFrame[sleeper_id, name, pos, team, other].
    Uses Sleeper's own PPR points, then adds the TE premium on projected receptions."""
    rows = []
    for it in items:
        st, pl = it.get("stats") or {}, it.get("player") or {}
        pts, pos = st.get("pts_ppr"), pl.get("position")
        if pts is None or pos not in POS:
            continue
        team = it.get("team") or pl.get("team")
        rows.append({"sleeper_id": str(it.get("player_id")),
                     "name": f"{pl.get('first_name', '')} {pl.get('last_name', '')}".strip(),
                     "pos": pos, "team": SLEEPER_TEAM_FIX.get(team, team),
                     "other": pts + te_premium * (st.get("rec") or 0) * (pos == "TE")})
    return pd.DataFrame(rows)


def fetch_sleeper(season, week, te_premium):
    import requests
    r = requests.get(SLEEPER_URL.format(season=season, week=week), timeout=30)
    r.raise_for_status()
    return parse_sleeper(r.json(), te_premium)


def load_csv_source(path):
    """CSV with columns: player, proj  (optional: pos, team). 'proj' must already be in YOUR scoring."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"name": "player", "position": "pos", "projection": "proj"})
    for c in ("pos", "team"):
        if c not in df.columns:
            df[c] = np.nan
    df["team"] = df.team.replace({**SLEEPER_TEAM_FIX, **TEAM_FIX})
    return df.rename(columns={"player": "name", "proj": "other"})[["name", "pos", "team", "other"]]


def attach_other(up, other, label, use_ids=True):
    """Add column '<label>_proj' to `up` by matching players (ID crosswalk first, then name)."""
    up = up.copy()
    col = f"{label}_proj"
    up[col] = np.nan
    if use_ids and "sleeper_id" in other.columns:
        try:
            ids = nfl.load_ff_playerids().to_pandas()[["gsis_id", "sleeper_id"]].dropna()
            ids["sleeper_id"] = ids.sleeper_id.astype(str).str.replace(r"\.0$", "", regex=True)
            by_id = other.drop_duplicates("sleeper_id").set_index("sleeper_id").other
            up[col] = up.player_id.map(dict(zip(ids.gsis_id, ids.sleeper_id))).map(by_id)
        except Exception as e:
            print(f"(ID crosswalk unavailable - matching {label} by name instead: {str(e)[:80]})")
    idx = {}
    for r in other.itertuples():
        k = name_key(r.name)
        idx.setdefault((k, r.pos), []).append(r)
        idx.setdefault((k, None), []).append(r)
    for i in up.index[up[col].isna()]:
        k, pos, team = name_key(up.at[i, "player_display_name"]), up.at[i, "position"], up.at[i, "team"]
        cands = idx.get((k, pos)) or idx.get((k, None)) or []
        if len(cands) > 1:
            cands = [c for c in cands if c.team == team]
        if len(cands) == 1:
            up.at[i, col] = cands[0].other
    up[f"vs_{label}"] = up.proj - up[col]
    return up


def report_comparison(out, label):
    c, v = f"{label}_proj", f"vs_{label}"
    m = out.dropna(subset=[c])
    print(f"\n=== OURS vs {label.upper()} - matched {len(m)} of {len(out)} players "
          f"({(out.head(60)[c].notna().mean()):.0%} of our top 60) ===")
    rows = []
    for pos, g in m.groupby("pos"):
        rows.append({"pos": pos, "n": len(g), "ours_avg": g.proj.mean(), f"{label}_avg": g[c].mean(),
                     "corr": g.proj.corr(g[c]), "mean_abs_diff": g[v].abs().mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    show = ["player", "pos", "team", "opp", "recent_avg", "proj", c, v, "injury"]
    print(f"\nWe are HIGHER than {label} (ours - theirs):")
    print(m.sort_values(v, ascending=False).head(10)[show].to_string())
    print(f"\nWe are LOWER than {label}:")
    print(m.sort_values(v).head(10)[show].to_string())
    print("\n(big gaps usually = injury/role news they know and we don't; check those first)")


# ----------------------------------------------------------------------------- injuries
def attach_injuries(df, season, week):
    try:
        for attempt in range(3):          # the nflverse download occasionally times out
            try:
                inj = nfl.load_injuries(seasons=season).to_pandas()
                break
            except Exception:
                if attempt == 2:
                    raise
        inj = inj[inj.week == week][["gsis_id", "report_status", "practice_status"]]
        inj = inj.drop_duplicates("gsis_id")
        df = df.merge(inj, left_on="player_id", right_on="gsis_id", how="left").drop(columns="gsis_id")
        df["injury"] = (df.report_status.fillna("") + " " +
                        df.practice_status.fillna("").str.replace("Did Not Participate In Practice", "DNP")
                        ).str.strip()
        n_out = int((df.report_status == "Out").sum())
        df = df[df.report_status != "Out"]
        print(f"Injury report: dropped {n_out} players listed Out for week {week}.")
    except Exception as e:  # report not posted / not available
        print(f"(injury report unavailable: {e}) - check inactives yourself before trusting a rank.")
        df["injury"] = ""
    return df


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--te-premium", type=float, default=0.5,
                    help="EXTRA points per TE reception on top of full PPR (0.5 => 1.5 PPR for TEs)")
    ap.add_argument("--pass-td", type=float, default=4.0, help="points per passing TD (4 or 6)")
    ap.add_argument("--max-gap-days", type=int, default=45,
                    help="skip players whose last game was longer ago than this")
    ap.add_argument("--backtest", action="store_true",
                    help="run the week-by-week walk-forward backtest (2-3 minutes)")
    ap.add_argument("--first-test", type=int, default=2022, help="first season to backtest")
    ap.add_argument("--no-backtest", action="store_true", help=argparse.SUPPRESS)   # old flag, now the default
    ap.add_argument("--sleeper", action="store_true", help="add Sleeper's projections as a column")
    ap.add_argument("--compare-csv", default=None, help="CSV (player, proj[, pos, team]) from ESPN etc.")
    ap.add_argument("--compare-name", default="other", help="label for --compare-csv, e.g. ESPN")
    args = ap.parse_args()

    CFG.update(te_premium=args.te_premium, pass_td=args.pass_td)
    season = args.season or nfl.get_current_season()
    stats, games = load_data(season, args.te_premium, args.pass_td)

    cur = games[games.season == season]
    week = args.week or int(cur[~cur.played].week.min())
    print(f"Season {season}, projecting week {week}  |  TE premium: +{args.te_premium}/rec "
          f"(TE = {1 + args.te_premium:g} PPR), passing TD = {args.pass_td:g}")

    d = build_frame(stats, games, season, week, args.max_gap_days)
    d = add_features(d, games, season, week)

    if args.backtest:
        bt = backtest_weekly(d, args.first_test)
        wk = report_weekly_backtest(bt, season)
        save_backtest(bt, wk, season)

    pool = d[d.elig & ~d.upcoming]
    full, base = fit_stat_models(pool, True), fit_stat_models(pool, False)
    om = other_means(pool)
    direct = fit_models(pool, True, robust=True)
    report_coefficients(direct)
    report_te_stat_coefficients(full)

    up = d[d.upcoming & d.elig].copy()
    up = up.join(predict_stats(full, up, "p_"))
    up = up.join(predict_stats(base, up, "b_"))
    pp = project_points(up, "p_", om)
    up["proj"] = pp.proj
    up["pts_rec"], up["pts_yds"], up["pts_td"] = pp.pts_rec, pp.pts_yds, pp.pts_td
    up["pts_other"] = pp.pts_int + pp.pts_other           # interceptions + fumbles/2pt average
    up["form_only"] = project_points(up, "b_", om).proj
    up["proj_direct"] = predict(direct, up, True)         # one-regression-on-points model, for reference
    up["matchup_boost"] = up.proj - up.form_only
    up = up.dropna(subset=["proj"])
    up = attach_injuries(up, season, week)

    sources = []
    if args.sleeper:
        try:
            sources.append(("sleeper", fetch_sleeper(season, week, args.te_premium)))
        except Exception as e:
            print(f"(Sleeper fetch failed: {str(e)[:100]} - skipping. "
                  "Use --compare-csv with an export instead.)")
    if args.compare_csv:
        sources.append((args.compare_name.lower(), load_csv_source(args.compare_csv)))
    for label, other in sources:
        up = attach_other(up, other, label)
    extra = [c for label, _ in sources for c in (f"{label}_proj", f"vs_{label}")]

    up["side"] = np.where(up.is_home == 1, "vs", "@")
    up["opp_str"] = up.side + " " + up.opp
    up["line"] = up.team_spread.map(lambda x: f"{x:+.1f}")
    up = up.rename(columns={"p_" + s: SHORT[s] for s in STAT_COLS})
    stat_cols = [SHORT[s] for s in STAT_COLS]
    cols = (["player_display_name", "position", "team", "opp_str", "line", "total_line", "def_rel",
             "pts_ewma", "form_only", "proj", "matchup_boost"] + stat_cols +
            ["pts_rec", "pts_yds", "pts_td", "pts_other", "proj_direct"] + extra + ["injury"])
    out = up.sort_values("proj", ascending=False)[cols].rename(columns={
        "player_display_name": "player", "position": "pos", "opp_str": "opp",
        "total_line": "total", "pts_ewma": "recent_avg"})
    out = out.round(2).reset_index(drop=True)
    out.index += 1

    path = f"projections_{season}_wk{week:02d}.csv"
    out.to_csv(path, index_label="rank")

    head = ["player", "pos", "team", "opp", "line", "total", "recent_avg", "proj", "matchup_boost"] + extra + ["injury"]
    print(f"\n=== TOP 30 OVERALL - week {week} (TE-premium PPR) ===")
    print(out.head(30)[head].to_string())

    recv = ["player", "team", "opp", "line", "total", "proj", "tgt", "rec", "rec_yds", "rec_td",
            "car", "rush_yds", "rush_td", "pts_rec", "pts_yds", "pts_td", "pts_other"] + extra + ["injury"]
    qbc = ["player", "team", "opp", "line", "total", "proj", "pass_att", "pass_yds", "pass_td", "int",
           "car", "rush_yds", "rush_td", "pts_yds", "pts_td", "pts_other"] + extra + ["injury"]
    for pos, n in (("QB", 10), ("RB", 20), ("WR", 25), ("TE", 15)):
        print(f"\n--- {pos}: projected stat line and where the points come from ---")
        t = out[out.pos == pos].head(n)
        c = qbc if pos == "QB" else recv
        if pos == "TE":
            c = [x for x in c if x not in ("car", "rush_yds", "rush_td")]
        elif pos == "WR":
            c = [x for x in c if x != "rush_td"]
        print(t[c].to_string())
    print("\n=== BIGGEST MATCHUP BOOSTS / DRAGS (vs. the same player's form-only projection) ===")
    mb = ["player", "pos", "team", "opp", "line", "total", "recent_avg", "proj", "matchup_boost", "injury"]
    print(out.sort_values("matchup_boost", ascending=False).head(10)[mb].to_string())
    print(out.sort_values("matchup_boost").head(10)[mb].to_string())
    for label, _ in sources:
        report_comparison(out, label)
    print(f"\nSaved {len(out)} projections to {path}  (keep each week's file: it's your snapshot "
          "for scoring us vs. them after the games)")


if __name__ == "__main__":
    main()