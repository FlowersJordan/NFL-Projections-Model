#!/usr/bin/env python3
"""
Rookie comps: for each drafted rookie (QB / RB / WR / TE), find the former and current NFL players who looked most like
him -- same position, size and combine testing AND how he played and was used in college -- and show how their careers went.

Usage (same folder as rookie_model.py and college_data.py; first run downloads data into ./rookie_cache/):
    python rookie_comps.py                          # the 2026 class, every position (athletic + college)
    python rookie_comps.py --player "Jeremiyah Love"
    python rookie_comps.py --class 2025 --pos WR
    python rookie_comps.py --mode athletic          # size and testing only (the original comps)
    python rookie_comps.py --mode college           # college production and usage only
    python rookie_comps.py --college-weight 2       # lean harder on college (default 1; athletic weight 1)
    python rookie_comps.py --k 8 --pick-weight 0.5  # 8 comps; also reward similar draft slots (default 0)
    python rookie_comps.py --validate               # do athletics / college production add anything beyond draft slot?

How similarity works
  * Same position only; the pool is every drafted QB/RB/WR/TE since 2000, other than the player himself.
  * Two groups of numbers, each turned into z-scores within the position:
      athletic   height, weight, 40, vertical, broad jump, 3-cone, shuttle, bench, age
      college    his final college season: production per game (yards, catches, rush yards, yards per carry/catch/attempt),
                 usage (target share, touch share, share of team yards, red-zone work), efficiency (EPA per opportunity,
                 explosive rate), and "dominator" (share of his team's receiving yards and TDs) plus the age he broke out
  * Missing numbers are skipped, never guessed: a comp must cover at least 70% of the weight of the numbers the rookie has.
    College data starts with the 2004 season, so with college included, comps come from draft classes 2005 on.
  * distance = weighted RMS gap in z-scores (nudged toward 1 when few numbers are available);
    similarity = 100 * exp(-distance^2 / 2). sim_athletic and sim_college show each group on its own.
  * Draft slot is not part of the match unless you set --pick-weight; each comp's draft pick is shown so you can judge.

Outputs
  rookie_comps_<name>.csv      one row per rookie/comp pair (rank, similarity, comp's numbers and career)
  rookie_profiles_<name>.csv   each rookie's athletic and college percentiles within his position (100 = best)
  rookie_comp_summary_<name>.csv
  Career tiers (of comps drafted 2022 or earlier): Star = 2+ top-12 seasons; Starter = a top-12 or two top-24 seasons;
  Fringe = a top-36 season; Bust = none. Comps drafted 2023+ show "too early".
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy.stats import spearmanr

import rookie_model as rm
import college_data as cd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

POS = rm.POS
ATH_W = {"ht": 1.0, "wt": 1.5, "forty": 2.0, "vertical": 1.0, "broad_jump": 1.0, "cone": 0.75, "shuttle": 0.75,
         "bench": 0.5, "age": 0.5}
_REC = {"c_rec_ypg": 1.5, "c_tgt_share": 1.5, "c_dom": 1.5, "c_ypr": 1.0, "c_epa_opp": 1.0, "c_explosive": 0.75,
        "c_td_pg": 0.75, "c_breakout_age": 1.0}
COL_W = {
    "WR": _REC, "TE": _REC,
    "RB": {"c_rush_ypg": 1.5, "c_ypc": 1.0, "c_touch_share": 1.5, "c_scr_yd_share": 1.5, "c_rec_pg": 1.0, "c_tgt_share": 0.75,
           "c_epa_opp": 1.0, "c_explosive": 0.75, "c_td_pg": 0.75},
    "QB": {"c_ypa": 1.5, "c_cmp_pct": 1.0, "c_pass_td_pg": 1.0, "c_int_pct": 0.75, "c_pass_ypg": 1.0, "c_rush_att_pg": 1.0,
           "c_rush_ypg": 0.75},
}
COLLEGE_ALL = sorted({f for d in COL_W.values() for f in d})
LOWER_BETTER = {"forty", "cone", "shuttle", "age", "c_int_pct", "c_breakout_age"}
MIN_COVERAGE = 0.70           # a comp must cover this share of the rookie's measured weight
PRIOR_W = 2.0                 # weight of the 'one standard deviation apart' prior (penalises matches built on few numbers)
MIN_FEATURE_SHARE = 0.25      # skip a measurement nobody at the position has (e.g. QB bench press)
CLIP = 3.5                    # z-scores are capped so one wild number cannot dominate a match
EARLY_CLASS = 2023            # comps drafted this year or later have too little career to label
LABEL = {"ht": "height", "wt": "weight", "forty": "40", "vertical": "vertical", "broad_jump": "broad", "cone": "3-cone",
         "shuttle": "shuttle", "bench": "bench", "age": "age",
         "c_rec_ypg": "rec yds/g", "c_tgt_share": "target share", "c_dom": "dominator", "c_ypr": "yds/catch",
         "c_epa_opp": "EPA/opp", "c_explosive": "explosive %", "c_td_pg": "TD/g", "c_breakout_age": "breakout age",
         "c_rush_ypg": "rush yds/g", "c_ypc": "yds/carry", "c_touch_share": "touch share", "c_scr_yd_share": "yd share",
         "c_rec_pg": "catches/g", "c_ypa": "yds/att", "c_cmp_pct": "comp %", "c_pass_td_pg": "pass TD/g",
         "c_int_pct": "INT %", "c_pass_ypg": "pass yds/g", "c_rush_att_pg": "rush att/g"}


# ----------------------------------------------------------------------------------------------------- data
def load_all():
    c, d, st = rm.load_raw()
    m = rm.build_dataset(c, d, st).reset_index(drop=True)
    cf = cd.college_features(m)
    m = pd.concat([m, cf.drop(columns=["_idx"])], axis=1)
    ro = rm.cached("rosters_all", lambda: rm.retry(lambda: nfl.load_rosters(seasons=list(range(2000, 2027))).to_pandas()))
    ro = ro[["gsis_id", "season", "height", "weight"]].dropna(subset=["gsis_id"])
    ro = ro.sort_values("season").drop_duplicates("gsis_id", keep="last")      # latest listing per player
    m = m.merge(ro[["gsis_id", "height", "weight"]], on="gsis_id", how="left")
    m["ht"] = m.ht.fillna(m.height)
    m["wt"] = m.wt.fillna(m.weight)
    m = m.drop(columns=["height", "weight"])
    for col in COLLEGE_ALL:
        if col not in m:
            m[col] = np.nan

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


def feature_weights(pool, pos, mode, a_w, c_w):
    """Ordered dict feature -> weight for this position, only features enough of the position's players have."""
    p = pool[pool.pos == pos]
    out = {}
    if mode in ("both", "athletic"):
        for f, w in ATH_W.items():
            if p[f].notna().mean() >= MIN_FEATURE_SHARE:
                out[f] = w * a_w
    if mode in ("both", "college"):
        for f, w in COL_W[pos].items():
            if p[f].notna().mean() >= MIN_FEATURE_SHARE * 0.5:
                out[f] = w * c_w
    return out


def zmat(df, feats, mu, sd):
    Z = np.column_stack([(pd.to_numeric(df[f], errors="coerce").to_numpy() - mu[f]) / sd[f] for f in feats])
    return np.clip(Z, -CLIP, CLIP)


def _dist(Zp, zt, w):
    """Weighted RMS z gap with the evidence prior; returns (distance, shared weight)."""
    mt = ~np.isnan(zt)
    shared = ~np.isnan(Zp) & mt[None, :]
    sw = (shared * w).sum(1)
    d2 = (np.where(shared, (np.nan_to_num(Zp) - np.nan_to_num(zt)[None, :]) ** 2 * w, 0).sum(1) + PRIOR_W) / (np.maximum(sw, 1e-9) + PRIOR_W)
    return np.sqrt(d2), sw


def comps_for(targets, pool, k, pick_weight, mode="both", a_w=1.0, c_w=1.0):
    rows, prof = [], []
    for pos in POS:
        t = targets[targets.pos == pos]
        if t.empty:
            continue
        pl = pool[pool.pos == pos].copy()
        fw = feature_weights(pool, pos, mode, a_w, c_w)
        feats = list(fw)
        if not feats:
            continue
        mu = {f: pl[f].mean() for f in feats}
        sd = {f: (pl[f].std() or 1.0) for f in feats}
        w = np.array([fw[f] for f in feats])
        is_col = np.array([f.startswith("c_") for f in feats])
        feats_all, w_all = feats, w
        if pick_weight > 0:
            mu["log_pick"], sd["log_pick"] = pl.log_pick.mean(), pl.log_pick.std()
            feats_all = feats + ["log_pick"]
            w_all = np.append(w, pick_weight)
            is_col = np.append(is_col, False)
        Zp, Zt = zmat(pl, feats_all, mu, sd), zmat(t, feats_all, mu, sd)
        w_ath = np.where(is_col, 0.0, w_all)
        w_col = np.where(is_col, w_all, 0.0)
        for i, tr in enumerate(t.to_dict("records")):
            zt = Zt[i]
            mt = ~np.isnan(zt)
            tw = float((w_all * mt).sum())
            if tw <= 0:
                continue
            dist, sw = _dist(Zp, zt, w_all)
            cov = sw / tw
            ok = (cov >= MIN_COVERAGE) & (pl.player.to_numpy() != tr["player"])
            if isinstance(tr.get("gsis_id"), str):
                ok &= pl.gsis_id.to_numpy() != tr["gsis_id"]
            if not ok.any():
                continue
            da, swa = _dist(Zp, zt, w_ath)
            dc, swc = _dist(Zp, zt, w_col)
            order = [j for j in np.argsort(np.where(ok, dist, np.inf))[:k] if ok[j]]
            for rank, j in enumerate(order, 1):
                c = pl.iloc[j]
                sim = lambda dd, ss: round(100 * float(np.exp(-dd ** 2 / 2)), 1) if ss > 0 else np.nan
                rows.append({
                    "player": tr["player"], "pos": pos, "class": tr.get("_class", np.nan), "pick": tr.get("pick", np.nan),
                    "rank": rank, "comp": c.player, "comp_class": int(c["class"]), "comp_pick": int(c.pick),
                    "comp_school": c.get("college", ""),
                    "similarity": round(100 * float(np.exp(-dist[j] ** 2 / 2)), 1), "distance": round(float(dist[j]), 2),
                    "sim_athletic": sim(da[j], swa[j]) if (w_ath > 0).any() else np.nan,
                    "sim_college": sim(dc[j], swc[j]) if (w_col > 0).any() else np.nan,
                    "overlap_%": round(100 * float(cov[j]), 0),
                    **{f"comp_{f}": c[f] for f in ["ht", "wt", "forty", "vertical", "broad_jump", "cone", "shuttle", "bench"]},
                    **{f"comp_{f}": c[f] for f in COLLEGE_ALL},
                    "comp_tier": c.tier, "comp_best_pts": round(float(c.career_best_pts), 1),
                    "comp_best_rank": c.career_best_rank, "comp_top12_seasons": int(c.top12_seasons),
                    "comp_top24_seasons": int(c.top24_seasons), "comp_seasons": int(c.career_seasons),
                    "comp_last_season": c.last_season, "comp_first5_best_pts": round(float(c.best_pts), 1),
                    "comp_first5_top24": int(c.top24), "comp_first5_top12": int(c.top12),
                })
            # percentile profile versus the position (100 = best), athletic and college measures
            pr = {"player": tr["player"], "pos": pos, "pick": tr.get("pick", np.nan)}
            for f in list(ATH_W) + list(COL_W[pos]):
                v = tr.get(f, np.nan)
                col = pl[f].dropna().to_numpy()
                if pd.notna(v) and len(col) >= 30:
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
                    "avg_sim_athletic": round(g.sim_athletic.mean(), 1) if g.sim_athletic.notna().any() else np.nan,
                    "avg_sim_college": round(g.sim_college.mean(), 1) if g.sim_college.notna().any() else np.nan,
                    "comp_avg_pick": round(g.comp_pick.mean(), 0),
                    "comps_star_%": round(100 * (lab.comp_tier == "Star").mean(), 0) if len(lab) else np.nan,
                    "comps_starter_or_better_%": round(100 * lab.comp_tier.isin(["Star", "Starter"]).mean(), 0) if len(lab) else np.nan,
                    "comps_bust_%": round(100 * (lab.comp_tier == "Bust").mean(), 0) if len(lab) else np.nan,
                    "comp_avg_best_pts": round(lab.comp_best_pts.mean(), 0) if len(lab) else np.nan})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------------- validation
def validate(m, k, pick_weight):
    """Walk-forward tests, classes whose five seasons had finished only:
       A) comps-based prediction (mean first-5-year best season of his k comps), by mode
       B) ridge models: draft slot only vs + athletic vs + college vs both."""
    print("\n=== A) DO COMPS PREDICT CAREERS? (walk-forward, no peeking) ===")
    last = rm.LAST_COMPLETE_SEASON - rm.WINDOW + 1
    print(f"Each rookie in classes 2012-{last} is matched only to players from classes at least {rm.WINDOW} years earlier "
          f"(and 2005 or later, when college data starts). Prediction = the average first-{rm.WINDOW}-year best season of his {k} comps.")
    comp = m[m.complete & (m["class"] >= 2005)].copy()
    rows = []
    for pos in POS:
        res = {"y": [], "naive": [], "pick": []}
        for mode in ("athletic", "college", "both"):
            res[mode] = []
        keep = None
        for Y in range(2012, last + 1):
            tg = comp[(comp.pos == pos) & (comp["class"] == Y)]
            pool = comp[comp["class"] <= Y - rm.WINDOW]
            if tg.empty or len(pool[pool.pos == pos]) < 50:
                continue
            preds = {}
            for mode in ("athletic", "college", "both"):
                cm, _ = comps_for(tg.assign(_class=Y), pool, k, pick_weight, mode)
                preds[mode] = cm.groupby("player").comp_first5_best_pts.mean() if len(cm) else pd.Series(dtype=float)
            tr = pool[pool.pos == pos]
            pk = rm.fit_predict(tr, tg, rm.BASE)["best_pts"]
            for i, r in enumerate(tg.to_dict("records")):
                if all(r["player"] in preds[md].index for md in preds):
                    res["y"].append(r["best_pts"]); res["naive"].append(tr.best_pts.mean()); res["pick"].append(pk[i])
                    for md in preds:
                        res[md].append(preds[md][r["player"]])
        if len(res["y"]) < 50:
            continue
        y = np.array(res["y"])
        rmse = lambda p: float(np.sqrt(np.mean((y - np.array(p)) ** 2)))
        rows.append({"pos": pos, "n": len(y), "rmse_naive": rmse(res["naive"]), "rmse_draft_slot": rmse(res["pick"]),
                     "rmse_athletic": rmse(res["athletic"]), "rmse_college": rmse(res["college"]), "rmse_both": rmse(res["both"]),
                     "rho_draft_slot": spearmanr(res["pick"], y)[0], "rho_athletic": spearmanr(res["athletic"], y)[0],
                     "rho_college": spearmanr(res["college"], y)[0], "rho_both": spearmanr(res["both"], y)[0]})
    ta = pd.DataFrame(rows)
    print(ta.round(3).to_string(index=False))
    print("rmse: typical miss in best-season fantasy points (lower is better). rho: rank correlation with what happened (higher is better).")

    print("\n=== B) DOES COLLEGE PRODUCTION ADD ANYTHING BEYOND DRAFT SLOT? (ridge model, walk-forward) ===")
    rows = []
    for pos in POS:
        cfeat = list(COL_W[pos])
        sets = {"draft slot": rm.BASE, "slot + athletic": rm.WITH_AGE + [a for a in rm.ATHLETIC if a != "age"],
                "slot + college": rm.WITH_AGE + cfeat, "slot + both": rm.WITH_AGE + [a for a in rm.ATHLETIC if a != "age"] + cfeat}
        y, out = [], {k_: [] for k_ in sets}
        for Y in range(2012, last + 1):
            tg = comp[(comp.pos == pos) & (comp["class"] == Y)]
            tr = comp[(comp.pos == pos) & (comp["class"] <= Y - rm.WINDOW)]
            if tg.empty or len(tr) < 50:
                continue
            y += list(tg.best_pts)
            for name_, feats in sets.items():
                out[name_] += list(rm.fit_predict(tr, tg, feats)["best_pts"])
        if len(y) < 50:
            continue
        y = np.array(y)
        row = {"pos": pos, "n": len(y)}
        for name_ in sets:
            row[f"rmse {name_}"] = float(np.sqrt(np.mean((y - np.array(out[name_])) ** 2)))
        for name_ in sets:
            row[f"rho {name_}"] = spearmanr(out[name_], y)[0]
        rows.append(row)
    tb = pd.DataFrame(rows)
    print(tb.round(3).to_string(index=False))
    print("If 'slot + college' is not clearly better than 'draft slot', college numbers describe how a rookie played but add little forecast "
          "power beyond where he was drafted.")
    return ta, tb


# ----------------------------------------------------------------------------------------------------- output
def show(comps, summary, profiles, limit):
    names = list(dict.fromkeys(comps.player))[:limit]
    for p in names:
        g = comps[comps.player == p]
        s = summary[summary.player == p].iloc[0]
        pr = profiles[profiles.player == p]
        head = f"\n{p} ({g.pos.iloc[0]}, pick {int(g.pick.iloc[0]) if pd.notna(g.pick.iloc[0]) else '?'})"
        print(head)
        if len(pr):
            pr = pr.iloc[0]
            for title, keys in (("athletic", list(ATH_W)), ("college ", [f for f in COL_W[g.pos.iloc[0]]])):
                bits = [f"{LABEL[f]} {pr[f]:.3g} ({ordinal(int(pr['pct_' + f]))})" for f in keys
                        if f in pr and pd.notna(pr[f]) and f"pct_{f}" in pr and pd.notna(pr[f"pct_{f}"])]
                if bits:
                    print(f"   {title}: " + ", ".join(bits))
        t = g[["rank", "comp", "comp_class", "comp_pick", "similarity", "sim_athletic", "sim_college", "comp_tier",
               "comp_best_pts", "comp_top12_seasons", "comp_top24_seasons"]]
        print(t.rename(columns={"comp_class": "class", "comp_pick": "pick", "comp_tier": "tier", "comp_best_pts": "best_pts",
                                "comp_top12_seasons": "top12", "comp_top24_seasons": "top24", "sim_athletic": "ath",
                                "sim_college": "col"}).to_string(index=False))
        print(f"   comps: {s['comps_starter_or_better_%']:.0f}% became starters or better, {s['comps_bust_%']:.0f}% busts "
              f"(avg comp drafted at pick {s.comp_avg_pick:.0f}; avg similarity {s.avg_similarity:.0f})" if pd.notna(s["comps_starter_or_better_%"]) else
              "   comps: mostly recent players, careers too early to label")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", type=int, default=2026)
    ap.add_argument("--player", help="one player by name (any class)")
    ap.add_argument("--pos", help="QB, RB, WR or TE")
    ap.add_argument("--prospects", help="CSV of prospects (see rookie_model.py; add c_* college columns if you have them)")
    ap.add_argument("--mode", default="both", choices=["both", "athletic", "college"])
    ap.add_argument("--athletic-weight", type=float, default=1.0)
    ap.add_argument("--college-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5, help="comps per player")
    ap.add_argument("--pick-weight", type=float, default=0.0, help="weight on draft slot in the match (0 = none)")
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
        raw = pd.read_csv(a.prospects)
        raw.columns = [c.strip().lower() for c in raw.columns]
        for c in COLLEGE_ALL:
            tg[c] = pd.to_numeric(raw[c], errors="coerce").to_numpy() if c in raw else np.nan
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

    comps, prof = comps_for(tg, pool, a.k, a.pick_weight, a.mode, a.athletic_weight, a.college_weight)
    if comps.empty:
        raise SystemExit("No comps found (the player has too few measurements).")
    summary = summarize(comps)
    comps = comps.merge(summary[["player", "pos", "avg_similarity"]], on=["player", "pos"], how="left", suffixes=("", "_avg"))
    comps.to_csv(f"rookie_comps_{name}.csv", index=False)
    prof.to_csv(f"rookie_profiles_{name}.csv", index=False)
    summary.to_csv(f"rookie_comp_summary_{name}.csv", index=False)
    print(f"{len(tg)} players -> {comps.player.nunique()} with comps (mode: {a.mode}); wrote rookie_comps_{name}.csv, "
          f"rookie_profiles_{name}.csv, rookie_comp_summary_{name}.csv")
    show(comps, summary, prof, a.limit)
    print("\nSimilarity covers how a player is built and how he played in college. Run --validate to see how much it says about NFL careers.")


if __name__ == "__main__":
    main()