#!/usr/bin/env python3
"""
Rookie comps: for each drafted rookie (QB / RB / WR / TE), find the former and current NFL players who looked most
like him physically and athletically -- same position, size, combine numbers -- and show how those players' careers went.

Usage (same folder as rookie_model.py; first run downloads nflverse data into ./rookie_cache/):
    python rookie_comps.py                          # the 2026 class, every position
    python rookie_comps.py --player "Jeremiyah Love"
    python rookie_comps.py --class 2025 --pos WR
    python rookie_comps.py --prospects my_list.csv  # same columns as rookie_model.py (player, pos, pick, ht, wt, forty, ...)
    python rookie_comps.py --k 8 --pick-weight 0.5  # 8 comps; also reward similar draft slots (default 0 = athletic only)
    python rookie_comps.py --validate               # does "what happened to similar athletes" actually predict careers?

How similarity works
  * Compared only within the same position (the comp pool is every drafted QB/RB/WR/TE, 2000 on, other than the player himself).
  * Each measurement is turned into a z-score within the position (height, weight, 40, vertical, broad jump, cone, shuttle,
    bench, age). Missing numbers are skipped, never guessed: a comp must cover at least 70% of the weight of the numbers
    the rookie actually has. Height/weight come from the combine, falling back to NFL roster listings.
  * distance = weighted RMS gap in z-scores, nudged toward 1 when only a few numbers are available; similarity = 100 * exp(-distance^2 / 2)  (100 = identical, ~60 = one standard deviation off on average).
  * Draft slot is NOT part of the match unless you set --pick-weight; each comp's draft pick is shown so you can judge.

Outputs
  rookie_comps_<name>.csv      one row per rookie/comp pair (rank, similarity, comp's numbers and career)
  rookie_profiles_<name>.csv   each rookie's size/testing percentiles within his position (100 = best; fast times count as high)
  Career tiers (of comps drafted 2022 or earlier): Star = 2+ top-12 seasons; Starter = a top-12 or two top-24 seasons;
  Fringe = a top-36 season; Bust = none. Comps drafted 2023+ show "too early".
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.stats import rankdata, spearmanr

import rookie_model as rm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

POS = rm.POS
WEIGHTS = {"ht": 1.0, "wt": 1.5, "forty": 2.0, "vertical": 1.0, "broad_jump": 1.0, "cone": 0.75, "shuttle": 0.75,
           "bench": 0.5, "age": 0.5}
LOWER_BETTER = {"forty", "cone", "shuttle", "age"}
MIN_COVERAGE = 0.70           # a comp must cover this share of the rookie's measured weight
PRIOR_W = 2.0                # weight of the 'one standard deviation apart' prior (penalises matches built on few numbers)
MIN_FEATURE_SHARE = 0.25      # skip a measurement nobody at the position has (e.g. QB bench press)
EARLY_CLASS = 2023            # comps drafted this year or later have too little career to label
LABEL = {"ht": "height", "wt": "weight", "forty": "40", "vertical": "vertical", "broad_jump": "broad", "cone": "3-cone",
         "shuttle": "shuttle", "bench": "bench", "age": "age"}


# ----------------------------------------------------------------------------------------------------- data
def load_all():
    c, d, st = rm.load_raw()
    m = rm.build_dataset(c, d, st)
    ro = rm.cached("rosters_all", lambda: rm.retry(lambda: nfl.load_rosters(seasons=list(range(2000, 2027))).to_pandas()))
    ro = ro[["gsis_id", "season", "height", "weight"]].dropna(subset=["gsis_id"])
    ro = ro.sort_values("season").drop_duplicates("gsis_id", keep="last")      # latest listing per player
    m = m.merge(ro[["gsis_id", "height", "weight"]], on="gsis_id", how="left")
    m["ht"] = m.ht.fillna(m.height)
    m["wt"] = m.wt.fillna(m.weight)
    m = m.drop(columns=["height", "weight"])

    # career-wide summary of every drafted skill player (not just the first five seasons)
    ps = rm.season_table(st)
    g = ps.groupby("player_id")
    car = pd.DataFrame({
        "career_seasons": g.season.nunique(), "last_season": g.season.max(), "career_pts": g.pts.sum(),
        "career_best_pts": g.pts.max(), "career_best_rank": g["rank"].min(),
        "top12_seasons": g["rank"].apply(lambda s: int((s <= 12).sum())),
        "top24_seasons": g["rank"].apply(lambda s: int((s <= 24).sum())),
        "top36_seasons": g["rank"].apply(lambda s: int((s <= 36).sum())),
    }).reset_index().rename(columns={"player_id": "gsis_id"})
    m = m.merge(car, on="gsis_id", how="left")
    for col in ["career_seasons", "career_pts", "career_best_pts", "top12_seasons", "top24_seasons", "top36_seasons"]:
        m[col] = m[col].fillna(0)
    m["tier"] = [tier(r) for r in m.to_dict("records")]
    return m


def tier(r):
    if r["class"] >= EARLY_CLASS:
        return "too early"
    if r["top12_seasons"] >= 2:
        return "Star"
    if r["top12_seasons"] >= 1 or r["top24_seasons"] >= 2:
        return "Starter"
    if r["top36_seasons"] >= 1:
        return "Fringe"
    return "Bust"


def feature_matrix(df, feats, mu, sd):
    Z = np.column_stack([(pd.to_numeric(df[f], errors="coerce").to_numpy() - mu[f]) / sd[f] for f in feats])
    return Z


def position_stats(pool, pos):
    p = pool[pool.pos == pos]
    feats = [f for f in WEIGHTS if p[f].notna().mean() >= MIN_FEATURE_SHARE]
    mu = {f: p[f].mean() for f in feats}
    sd = {f: p[f].std() or 1.0 for f in feats}
    return feats, mu, sd


def comps_for(targets, pool, k, pick_weight):
    rows, prof = [], []
    for pos in POS:
        t = targets[targets.pos == pos]
        if t.empty:
            continue
        pl = pool[pool.pos == pos].copy()
        feats, mu, sd = position_stats(pool, pos)
        w = np.array([WEIGHTS[f] for f in feats])
        if pick_weight > 0:
            feats_all = feats + ["log_pick"]
            mu["log_pick"], sd["log_pick"] = pl.log_pick.mean(), pl.log_pick.std()
            w = np.append(w, pick_weight)
        else:
            feats_all = feats
        Zp = feature_matrix(pl, feats_all, mu, sd)
        Zt = feature_matrix(t, feats_all, mu, sd)
        Mp = ~np.isnan(Zp)
        for i, tr in enumerate(t.to_dict("records")):
            zt, mt = Zt[i], ~np.isnan(Zt[i])
            tw = float((w * mt).sum())
            if tw <= 0:
                continue
            shared = Mp & mt[None, :]
            cov = (shared * w).sum(1) / tw
            sw = (shared * w).sum(1)
            # a match on few measurements is weak evidence: shrink toward "one standard deviation apart" by a small prior weight
            d2 = (np.where(shared, (np.nan_to_num(Zp) - np.nan_to_num(zt)[None, :]) ** 2 * w, 0).sum(1) + PRIOR_W) / (np.maximum(sw, 1e-9) + PRIOR_W)
            ok = (cov >= MIN_COVERAGE) & (pl.player.to_numpy() != tr["player"])
            if isinstance(tr.get("gsis_id"), str):
                ok &= pl.gsis_id.to_numpy() != tr["gsis_id"]
            if not ok.any():
                continue
            dist = np.sqrt(d2)
            order = np.argsort(np.where(ok, dist, np.inf))[:k]
            order = [j for j in order if ok[j]]
            for rank, j in enumerate(order, 1):
                c = pl.iloc[j]
                rows.append({
                    "player": tr["player"], "pos": pos, "class": tr.get("_class", np.nan), "pick": tr.get("pick", np.nan),
                    "rank": rank, "comp": c.player, "comp_class": int(c["class"]), "comp_pick": int(c.pick),
                    "comp_school": c.get("college", ""),
                    "similarity": round(100 * float(np.exp(-dist[j] ** 2 / 2)), 1), "distance": round(float(dist[j]), 2),
                    "overlap_%": round(100 * float(cov[j]), 0),
                    **{f"comp_{f}": c[f] for f in ["ht", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]},
                    "comp_tier": c.tier, "comp_best_pts": round(float(c.career_best_pts), 1),
                    "comp_best_rank": c.career_best_rank, "comp_top12_seasons": int(c.top12_seasons),
                    "comp_top24_seasons": int(c.top24_seasons), "comp_seasons": int(c.career_seasons),
                    "comp_last_season": c.last_season, "comp_first5_best_pts": round(float(c.best_pts), 1),
                    "comp_first5_top24": int(c.top24), "comp_first5_top12": int(c.top12),
                })
            # percentile profile versus the position (100 = best)
            pr = {"player": tr["player"], "pos": pos, "pick": tr.get("pick", np.nan)}
            for f in feats:
                v = tr.get(f, np.nan)
                col = pl[f].dropna().to_numpy()
                if pd.notna(v) and len(col):
                    pct = 100 * (col < v).mean() + 50 * (col == v).mean()
                    pr[f"pct_{f}"] = round(100 - pct if f in LOWER_BETTER else pct, 0)
                pr[f] = v
            prof.append(pr)
    return pd.DataFrame(rows), pd.DataFrame(prof)


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def summarize(comps):
    """Per-rookie view of what his comps went on to do."""
    out = []
    for (p, pos), g in comps.groupby(["player", "pos"], sort=False):
        lab = g[g.comp_tier != "too early"]
        out.append({"player": p, "pos": pos, "n_comps": len(g), "avg_similarity": round(g.similarity.mean(), 1),
                    "comp_avg_pick": round(g.comp_pick.mean(), 0),
                    "comps_star_%": round(100 * (lab.comp_tier == "Star").mean(), 0) if len(lab) else np.nan,
                    "comps_starter_or_better_%": round(100 * lab.comp_tier.isin(["Star", "Starter"]).mean(), 0) if len(lab) else np.nan,
                    "comps_bust_%": round(100 * (lab.comp_tier == "Bust").mean(), 0) if len(lab) else np.nan,
                    "comp_avg_best_pts": round(lab.comp_best_pts.mean(), 0) if len(lab) else np.nan})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------------- validation
def validate(m, k, pick_weight):
    """For past classes: predict each player's first-5-year best season from athletic comps drawn only from classes whose
    five seasons were already finished, and compare with the draft-slot-only model from rookie_model.py."""
    print("\n=== DO ATHLETIC COMPS PREDICT CAREERS? (walk-forward, no peeking) ===")
    print(f"Each rookie in classes 2010-{rm.LAST_COMPLETE_SEASON - rm.WINDOW + 1} is matched only to players from classes at least "
          f"{rm.WINDOW} years earlier. Prediction = the average first-{rm.WINDOW}-year best season of his {k} comps.")
    comp = m[m.complete].copy()
    rows = []
    for pos in POS:
        y_true, y_comp, y_pick, y_naive = [], [], [], []
        for Y in range(2010, rm.LAST_COMPLETE_SEASON - rm.WINDOW + 2):
            tg = comp[(comp.pos == pos) & (comp["class"] == Y)]
            pool = comp[comp["class"] <= Y - rm.WINDOW]
            if tg.empty or len(pool[pool.pos == pos]) < 60:
                continue
            cm, _ = comps_for(tg.assign(_class=Y), pool, k, pick_weight)
            if cm.empty:
                continue
            pred = cm.groupby("player").comp_first5_best_pts.mean()
            tr = pool[pool.pos == pos]
            pk = rm.fit_predict(tr, tg, rm.BASE)["best_pts"]
            for i, r in enumerate(tg.to_dict("records")):
                if r["player"] in pred.index:
                    y_true.append(r["best_pts"]); y_comp.append(pred[r["player"]]); y_pick.append(pk[i]); y_naive.append(tr.best_pts.mean())
        if len(y_true) < 50:
            continue
        y = np.array(y_true)
        rm_ = lambda p: float(np.sqrt(np.mean((y - np.array(p)) ** 2)))
        rows.append({"pos": pos, "n": len(y), "rmse_naive": rm_(y_naive), "rmse_comps": rm_(y_comp), "rmse_draft_slot": rm_(y_pick),
                     "rank_corr_comps": spearmanr(y_comp, y)[0], "rank_corr_draft_slot": spearmanr(y_pick, y)[0]})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    print("\nrmse: typical miss in best-season fantasy points (lower is better; naive = always guess the position average).")
    print("rank_corr: how well the ordering matches reality (higher is better). If comps trail the draft-slot model, treat comps as a")
    print("description of what kind of athlete he is, not a forecast -- the draft slot already carries most of the information.")
    return t


# ----------------------------------------------------------------------------------------------------- output
def show(comps, summary, profiles, limit):
    names = list(dict.fromkeys(comps.player))[:limit]
    for p in names:
        g = comps[comps.player == p]
        s = summary[summary.player == p].iloc[0]
        pr = profiles[profiles.player == p]
        head = f"\n{p} ({g.pos.iloc[0]}, pick {int(g.pick.iloc[0]) if pd.notna(g.pick.iloc[0]) else '?'})"
        if len(pr):
            pr = pr.iloc[0]
            bits = [f"{LABEL[f]} {pr[f]:g} ({ordinal(int(pr['pct_' + f]))})" for f in WEIGHTS if f in pr and pd.notna(pr[f]) and f"pct_{f}" in pr and pd.notna(pr[f"pct_{f}"])]
            head += "   " + ", ".join(bits)
        print(head)
        t = g[["rank", "comp", "comp_class", "comp_pick", "similarity", "comp_tier", "comp_best_pts", "comp_top12_seasons", "comp_top24_seasons"]]
        print(t.rename(columns={"comp_class": "class", "comp_pick": "pick", "comp_tier": "tier", "comp_best_pts": "best_pts",
                                "comp_top12_seasons": "top12", "comp_top24_seasons": "top24"}).to_string(index=False))
        print(f"   comps: {s['comps_starter_or_better_%']:.0f}% became starters or better, {s['comps_bust_%']:.0f}% busts "
              f"(avg comp drafted at pick {s.comp_avg_pick:.0f}; avg similarity {s.avg_similarity:.0f})" if pd.notna(s["comps_starter_or_better_%"]) else
              "   comps: mostly recent players, careers too early to label")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", type=int, default=2026)
    ap.add_argument("--player", help="one player by name (any class)")
    ap.add_argument("--pos", help="QB, RB, WR or TE")
    ap.add_argument("--prospects", help="CSV of prospects (see rookie_model.py)")
    ap.add_argument("--k", type=int, default=5, help="comps per player")
    ap.add_argument("--pick-weight", type=float, default=0.0, help="weight on draft slot in the match (0 = athletic only)")
    ap.add_argument("--print", dest="limit", type=int, default=12, help="how many players to print to the screen (CSV has all)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.refresh and rm.CACHE.exists():
        for f in rm.CACHE.glob("*.pkl"):
            f.unlink()

    m = load_all()
    if a.validate:
        validate(m, a.k, a.pick_weight)
        return

    if a.prospects:
        tg = rm.prepare_prospects(a.prospects)
        name = Path(a.prospects).stem
        tg["_class"] = np.nan
    elif a.player:
        key = rm.name_key(a.player)
        tg = m[m.player.map(rm.name_key) == key].copy()
        if tg.empty:
            raise SystemExit(f"No drafted QB/RB/WR/TE named '{a.player}' (2000+).")
        name = key
        tg["_class"] = tg["class"]
    else:
        tg = m[m["class"] == a.cls].copy()
        if tg.empty:
            raise SystemExit(f"No drafted players found for class {a.cls}.")
        name = f"class_{a.cls}"
        tg["_class"] = a.cls
    if a.pos:
        tg = tg[tg.pos == a.pos.upper()]
    pool = m[m["class"] <= rm.LAST_COMPLETE_SEASON + 1]          # every drafted player 2000 on, including the current rookies
    if not a.prospects and not a.player:
        pool = pool[pool["class"] < tg["class"].max()] if a.cls == pool["class"].max() else pool

    comps, prof = comps_for(tg, pool, a.k, a.pick_weight)
    if comps.empty:
        raise SystemExit("No comps found (the player has too few measurements).")
    summary = summarize(comps)
    comps = comps.merge(summary[["player", "pos", "avg_similarity"]], on=["player", "pos"], how="left")
    comps.to_csv(f"rookie_comps_{name}.csv", index=False)
    prof.to_csv(f"rookie_profiles_{name}.csv", index=False)
    summary.to_csv(f"rookie_comp_summary_{name}.csv", index=False)
    print(f"{len(tg)} players -> {comps.player.nunique()} with comps; wrote rookie_comps_{name}.csv, "
          f"rookie_profiles_{name}.csv, rookie_comp_summary_{name}.csv")
    show(comps, summary, prof, a.limit)
    print("\nSimilarity is about size and athletic testing only. Run --validate to see how much it says about careers.")


if __name__ == "__main__":
    main()
