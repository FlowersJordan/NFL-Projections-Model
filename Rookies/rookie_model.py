#!/usr/bin/env python3
"""
Rookie-draft model, part 1: draft position + combine -> NFL career outcomes (QB / RB / WR / TE).

What it does
  1. Links every drafted QB/RB/WR/TE (2000+) to his combine results and to his NFL fantasy production
     (TE-premium full PPR, same scoring as weekly_projections.py).
  2. Measures how each combine number and draft slot relates to later success, by position, and whether
     athletic testing adds anything AFTER you account for draft slot.
  3. Fits a model on past draft classes and tests it honestly: for each class from 2012 on it trains only on
     classes whose first five NFL seasons were already finished.
  4. Scores the 2026 class (or any prospect CSV) with expected best season, chance of a top-12 / top-24 finish.

Outcomes (first five NFL seasons, positional finish by full-season fantasy points)
    best_pts   best single-season fantasy points
    top12/top24  finished top 12 / 24 at his position in at least one of those seasons
    A drafted player who never recorded a stat counts as 0.

Usage
    python rookie_model.py                           # report + score the 2026 class
    python rookie_model.py --class 2025              # score a different class (uses its actual picks and combine)
    python rookie_model.py --prospects prospects.csv # score your own list (see below)
    python rookie_model.py --te-premium 0.0          # different TE scoring

prospects.csv columns (header names exactly; leave blank what you don't know):
    player, pos, pick [, age, ht, wt, forty, vertical, broad_jump, cone, shuttle, bench]
    pick = overall draft slot (or your guess); ht in inches, wt in pounds, times in seconds.

Data come from nflverse (free, no key). First run downloads and caches them in ./rookie_cache/.
"""
import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

POS = ["QB", "RB", "WR", "TE"]
FIRST_CLASS = 2000
LAST_COMPLETE_SEASON = 2025          # 2026 is in progress; its stats are not used for outcomes
WINDOW = 5                           # first five NFL seasons
ATHLETIC = ["age", "ht", "wt", "bmi", "speed_score", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]
CACHE = Path("rookie_cache")
CFG = {"te_premium": 0.5}


def retry(f, tries=4):
    for i in range(tries):
        try:
            return f()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  (download problem, retrying: {str(e)[:60]})")
            time.sleep(3)


def cached(name, fn):
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    df = fn()
    df.to_pickle(p)
    return df


def name_key(s):
    s = str(s).lower().replace(".", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


# ----------------------------------------------------------------------------- data
def load_raw():
    c = cached("combine", lambda: retry(lambda: nfl.load_combine().to_pandas()))
    d = cached("draft", lambda: retry(lambda: nfl.load_draft_picks().to_pandas()))
    st = cached("stats", lambda: retry(lambda: nfl.load_player_stats(seasons=list(range(1999, 2026))).to_pandas()))
    return c, d, st


def season_table(st):
    """One row per player-season: fantasy points, games, and finish among players at his position that season."""
    s = st[st.season_type == "REG"].copy()
    s["pts"] = s.fantasy_points_ppr + CFG["te_premium"] * s.receptions.fillna(0) * (s.position == "TE")
    ps = s[s.position.isin(POS)].groupby(["player_id", "season", "position"], as_index=False).agg(
        pts=("pts", "sum"), games=("week", "nunique"))
    ps["rank"] = ps.groupby(["season", "position"]).pts.rank(ascending=False, method="first")
    return ps


def build_dataset(c, d, st):
    d = d[(d.season >= FIRST_CLASS) & d.position.isin(POS)].copy()
    d = d.rename(columns={"season": "class", "position": "pos", "pfr_player_name": "player"})
    d["nk"] = d.player.map(name_key)
    c = c[c.pos.isin(POS)].copy()
    c["nk"] = c.player_name.map(name_key)
    c["ht"] = c.ht.map(_inches)
    keep = ["pfr_id", "ht", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]

    # combine link 1: pfr id (most reliable); link 2: name + draft year for those left over
    m = d.merge(c.drop_duplicates("pfr_id")[keep], left_on="pfr_player_id", right_on="pfr_id", how="left")
    c2 = c.drop_duplicates(["draft_year", "nk"])[["draft_year", "nk"] + keep[1:]].rename(columns={"draft_year": "class"})
    miss = m.pfr_id.isna()
    fill = m.loc[miss, ["class", "nk"]].merge(c2, on=["class", "nk"], how="left")
    for col in keep[1:]:
        m.loc[miss, col] = fill[col].to_numpy()
    m["has_combine"] = m[["forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]].notna().any(axis=1).astype(int)
    for col in ["wt", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    m["age"] = pd.to_numeric(m.age, errors="coerce")
    m["bmi"] = 703.0 * m.wt / (m.ht ** 2)
    m["speed_score"] = m.wt * 200.0 / (m.forty ** 4)       # Barnwell speed score
    m["log_pick"] = np.log(m.pick.astype(float))
    m["log_pick2"] = m.log_pick ** 2

    # ---- career outcomes over the first WINDOW NFL seasons
    ps = season_table(st)
    j = m[["gsis_id", "class"]].dropna().merge(ps, left_on="gsis_id", right_on="player_id")
    j = j[(j.season >= j["class"]) & (j.season < j["class"] + WINDOW)]
    agg = j.groupby("gsis_id").agg(best_pts=("pts", "max"), best_rank=("rank", "min"),
                                   total_pts=("pts", "sum"), seasons=("season", "nunique")).reset_index()
    m = m.merge(agg, on="gsis_id", how="left")
    m["recorded"] = m.best_pts.notna()
    for col in ["best_pts", "total_pts", "seasons"]:
        m[col] = m[col].fillna(0.0)
    m["top12"] = (m.best_rank <= 12).astype(int)
    m["top24"] = (m.best_rank <= 24).astype(int)
    m["complete"] = (m["class"] + WINDOW - 1) <= LAST_COMPLETE_SEASON
    return m


def _inches(v):
    if pd.isna(v):
        return np.nan
    v = str(v)
    mt = re.match(r"^\s*(\d)\s*-\s*(\d{1,2})\s*$", v)
    if mt:
        return int(mt.group(1)) * 12 + int(mt.group(2))
    try:
        x = float(v)
        return x if x > 40 else np.nan
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------- descriptive tables
def diagnostics(m):
    comp = m[m.complete]
    print(f"\n=== DATA ===\nDrafted QB/RB/WR/TE, classes {FIRST_CLASS}-{int(m['class'].max())}: {len(m)}  "
          f"(outcomes complete for classes {FIRST_CLASS}-{LAST_COMPLETE_SEASON - WINDOW + 1}: {len(comp)})")
    t = comp.groupby("pos").agg(n=("player", "size"), with_combine=("has_combine", "mean"),
                                ever_recorded_stat=("recorded", "mean"), top24_rate=("top24", "mean"),
                                top12_rate=("top12", "mean"), avg_best_pts=("best_pts", "mean"))
    print(t.round(2).to_string())
    nog = comp.gsis_id.isna().sum()
    if nog:
        print(f"({nog} drafted players have no NFL player id; they never appeared in a game, so they count as 0.)")


def hit_rates_by_round(m):
    comp = m[m.complete]
    print("\n=== HOW OFTEN EACH DRAFT ROUND PRODUCES A STARTER (share finishing top 24 / top 12 at the position in years 1-5) ===")
    rows = []
    for pos, g in comp.groupby("pos"):
        for rd, h in g.groupby(g["round"].clip(upper=5)):
            rows.append({"pos": pos, "round": f"{int(rd)}" if rd < 5 else "5-7", "n": len(h),
                         "top24": h.top24.mean(), "top12": h.top12.mean(), "avg_best_pts": h.best_pts.mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def correlations(m):
    comp = m[m.complete].copy()
    rows = []
    for pos, g in comp.groupby("pos"):
        gr = g.copy()
        for f in ["log_pick"] + ATHLETIC:
            h = gr[list(dict.fromkeys([f, "best_pts", "log_pick"]))].dropna()
            if len(h) < 30:
                continue
            rho = spearmanr(h[f], h.best_pts)[0]
            # partial: remove what draft slot already explains from both (on ranks)
            r = h.rank()
            res = lambda y: y - np.polyval(np.polyfit(r.log_pick, y, 2), r.log_pick)
            part = np.corrcoef(res(r[f]), res(r.best_pts))[0, 1] if f != "log_pick" else np.nan
            rows.append({"pos": pos, "feature": "draft pick (lower=better)" if f == "log_pick" else f, "n": len(h),
                         "rho_vs_best_pts": rho, "rho_after_draft_slot": part})
    t = pd.DataFrame(rows)
    print("\n=== WHAT CORRELATES WITH A BETTER CAREER (classes with finished 5-year outcomes) ===")
    print("rho_vs_best_pts: Spearman with best single-season fantasy points. For 'draft pick' a NEGATIVE number "
          "means earlier picks do better.\nrho_after_draft_slot: the same, after removing what the draft slot "
          "already explains: this is what testing adds beyond draft capital.\nFor times (forty/cone/shuttle) lower is faster, "
          "so a negative number means faster players did better.")
    for pos in POS:
        g = t[t.pos == pos].drop(columns="pos")
        n = len(g)
        # |rho| of about 2/sqrt(n) is the noise band
        print(f"\n{pos}")
        g = g.assign(noise_band=(2.0 / np.sqrt(g.n)))
        g["signal"] = np.where(g.rho_after_draft_slot.abs() > g.noise_band, "*", "")
        print(g.round(3).to_string(index=False))
    print("\n('*' = beyond-draft-slot relationship larger than about 2 standard errors. With ~10 tests per position, "
          "expect a stray '*' or two by luck alone.)")
    return t


# ----------------------------------------------------------------------------- model
def design(tr, te, feats):
    """Standardize, fill missing testing numbers with the training median (plus a 'no testing' flag)."""
    X, Y = tr[feats].copy(), te[feats].copy()
    med = X.median()
    for df in (X, Y):
        df.fillna(med, inplace=True)
        df.fillna(0.0, inplace=True)
    X["no_combine"], Y["no_combine"] = (1 - tr.has_combine).to_numpy(), (1 - te.has_combine).to_numpy()
    sc = StandardScaler().fit(X)
    return sc.transform(X), sc.transform(Y)


BASE = ["log_pick", "log_pick2"]
WITH_AGE = BASE + ["age"]
MODELS = {"pick only": BASE, "pick + age": WITH_AGE, "pick + age + testing": WITH_AGE + [a for a in ATHLETIC if a != "age"]}


def fit_predict(tr, te, feats, alpha=30.0):
    """Returns expected best_pts, P(top24), P(top12) for the rows of `te`."""
    Xtr, Xte = design(tr, te, feats)
    out = {}
    out["best_pts"] = np.clip(Ridge(alpha=alpha).fit(Xtr, tr.best_pts).predict(Xte), 0, None)
    for tgt in ("top24", "top12"):
        if tr[tgt].sum() < 5:
            out[tgt] = np.full(len(te), tr[tgt].mean())
        else:
            out[tgt] = LogisticRegression(C=0.3, max_iter=2000).fit(Xtr, tr[tgt]).predict_proba(Xte)[:, 1]
    return out


def walk_forward(m, first_test=2012):
    comp = m[m.complete]
    print("\n=== HONEST TEST: predicting each class from earlier, finished classes only ===")
    print(f"For class Y we train only on classes <= Y-{WINDOW} (their first {WINDOW} seasons were already played).")
    rows, best = [], {}
    for pos in POS:
        g = comp[comp.pos == pos]
        res = {"y": [], "top24": [], "naive": []}
        for k in MODELS:
            res[k], res[k + "_24"] = [], []
        for Y in range(first_test, LAST_COMPLETE_SEASON - WINDOW + 2):
            tr, te = g[g["class"] <= Y - WINDOW], g[g["class"] == Y]
            if len(tr) < 60 or te.empty:
                continue
            res["y"] += list(te.best_pts); res["top24"] += list(te.top24); res["naive"] += [tr.best_pts.mean()] * len(te)
            for k, feats in MODELS.items():
                p = fit_predict(tr, te, feats)
                res[k] += list(p["best_pts"]); res[k + "_24"] += list(p["top24"])
        y = np.array(res["y"])
        if len(y) < 50:
            continue
        r = lambda p: float(np.sqrt(np.mean((y - np.array(p)) ** 2)))
        row = {"pos": pos, "n": len(y), "naive": r(res["naive"])}
        for k in MODELS:
            row[f"rmse {k}"] = r(res[k])
        for k in MODELS:
            row[f"rank_corr {k}"] = spearmanr(res[k], y)[0]
        for k in MODELS:
            row[f"AUC top24 {k}"] = roc_auc_score(res["top24"], res[k + "_24"])
        rows.append(row)
        best[pos] = min(MODELS, key=lambda k: row[f"rmse {k}"])
    t = pd.DataFrame(rows).set_index("pos")
    print("\nTypical miss in best-season fantasy points (rmse; lower is better; 'naive' = always guess the position average):")
    print(t[["n", "naive"] + [f"rmse {k}" for k in MODELS]].round(1).to_string())
    print("\nRank correlation of predicted vs actual best season (higher is better):")
    print(t[[f"rank_corr {k}" for k in MODELS]].round(3).to_string())
    print("\nAUC for 'finishes top 24 at some point' (0.5 = coin flip, 1 = perfect):")
    print(t[[f"AUC top24 {k}" for k in MODELS]].round(3).to_string())
    print("\nModel used for scoring, by position (lowest honest-test miss): " + ", ".join(f"{p}: {k}" for p, k in best.items()))
    print("If 'pick + age + testing' is not clearly better than 'pick only', combine numbers are not adding information beyond "
          "where the player was drafted.")
    return t.reset_index(), best


def score(m, targets, best, label):
    """Fit on all finished classes, score `targets` (rows with the same columns as m)."""
    comp = m[m.complete]
    rows = []
    for pos in POS:
        te = targets[targets.pos == pos]
        if te.empty:
            continue
        tr = comp[comp.pos == pos]
        feats = MODELS[best.get(pos, "pick + age")]
        p = fit_predict(tr, te, feats)
        rows.append(te.assign(exp_best_pts=p["best_pts"], p_top24=p["top24"], p_top12=p["top12"]))
    if not rows:
        return None
    o = pd.concat(rows)
    o["pos_rank"] = o.groupby("pos").exp_best_pts.rank(ascending=False, method="first").astype(int)
    return o


def prepare_prospects(path):
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]
    p["pos"] = p.pos.str.upper().str.strip()
    p["player"] = p.player.astype(str)
    for c in ["pick", "age", "ht", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]:
        p[c] = pd.to_numeric(p[c], errors="coerce") if c in p else np.nan
    p["pick"] = p.pick.fillna(p.pick.median() if p.pick.notna().any() else 100)
    p["bmi"] = 703.0 * p.wt / (p.ht ** 2)
    p["speed_score"] = p.wt * 200.0 / (p.forty ** 4)
    p["log_pick"] = np.log(p.pick)
    p["log_pick2"] = p.log_pick ** 2
    p["has_combine"] = p[["forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]].notna().any(axis=1).astype(int)
    p["round"] = np.ceil(p.pick / 32).clip(upper=7)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", type=int, default=2026, help="draft class to score (default 2026)")
    ap.add_argument("--prospects", default=None, help="CSV of prospects to score (see the docstring)")
    ap.add_argument("--te-premium", type=float, default=0.5)
    ap.add_argument("--refresh", action="store_true", help="re-download the data instead of using the cache")
    args = ap.parse_args()
    CFG["te_premium"] = args.te_premium
    if args.refresh and CACHE.exists():
        for f in CACHE.glob("*.pkl"):
            f.unlink()

    c, d, st = load_raw()
    m = build_dataset(c, d, st)
    diagnostics(m)
    hit_rates_by_round(m)
    corr = correlations(m)
    wf, best = walk_forward(m)
    corr.to_csv("rookie_correlations.csv", index=False)
    wf.to_csv("rookie_model_test.csv", index=False)

    if args.prospects:
        tg = prepare_prospects(args.prospects)
        name = Path(args.prospects).stem
    else:
        tg = m[m["class"] == args.cls].copy()
        name = f"class_{args.cls}"
        if tg.empty:
            print(f"(no drafted players found for class {args.cls})")
            return
    o = score(m, tg, best, name)
    if o is None:
        print("(nothing to score)")
        return
    cols = ["player", "pos", "pick", "round", "age", "forty", "exp_best_pts", "p_top24", "p_top12", "pos_rank"]
    if args.prospects is None and args.cls <= LAST_COMPLETE_SEASON - 1:
        cols += ["best_pts", "best_rank"]
    o = o.sort_values(["pos", "exp_best_pts"], ascending=[True, False])
    path = f"rookie_scores_{name}.csv"
    o[[c for c in cols if c in o]].round(2).to_csv(path, index=False)
    print(f"\n=== {name.upper().replace('_', ' ')}: expected career by position (model per position: {', '.join(f'{p}: {k}' for p, k in best.items())}) ===")
    for pos in POS:
        g = o[o.pos == pos]
        if g.empty:
            continue
        print(f"\n{pos}")
        print(g[[c for c in cols if c in g]].head(15).round(2).to_string(index=False))
    print(f"\n(saved to {path}; exp_best_pts = expected best single-season fantasy points in years 1-5; "
          "p_top24 / p_top12 = chance of a top-24 / top-12 finish at the position)")


if __name__ == "__main__":
    main()
