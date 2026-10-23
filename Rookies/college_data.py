#!/usr/bin/env python3
"""
College production and usage for drafted QB / RB / WR / TE, linked to the draft table in rookie_model.py.

Source: free ESPN-derived college football data published by sportsdataverse on GitHub (no key needed):
    espn_cfb_usage_players/usage_players_<year>.parquet   per player-season: rushes, targets, touches, shares, EPA, red-zone,
                                                          3rd-down, explosive and success rates (play-by-play based), 2004-
    espn_cfb_player_box/player_box_<year>.parquet         per player-game box scores (passing / rushing / receiving)
Coverage starts with the 2004 college season, so draft classes 2005+ get college numbers (earlier classes simply have none).

attach_college(m) adds columns prefixed c_ to the rookie_model dataset `m` for each drafted player's FINAL college season
(class year - 1; one season earlier if he did not play that year) plus career markers:

    all     c_found, c_seasons, c_games, c_scr_ypg (rush+rec yards / game), c_td_pg, c_touch_share, c_epa_opp,
            c_explosive, c_success, c_third_rate, c_rz_touch_pg
    WR/TE   c_rec_ypg, c_rec_pg, c_tgt_pg, c_ypr, c_ypt (yards per target), c_tgt_share, c_rec_yd_share, c_rec_td_share,
            c_dom (average of the two shares), c_best_dom, c_breakout_age (first season with dom >= 20%)
    RB      c_rush_ypg, c_rush_att_pg, c_ypc, c_rec_pg, c_rush_yd_share, c_scr_yd_share, c_best_dom
    QB      c_pass_att_pg, c_ypa, c_cmp_pct, c_pass_td_pg, c_int_pct, c_pass_ypg, c_rush_att_pg, c_rush_ypg
Cached in ./rookie_cache/college/ after the first run.
"""
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import rookie_model as rm

BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
FIRST_SEASON = 2004
CDIR = rm.CACHE / "college"
POSG = {"QB", "RB", "WR", "TE", "FB"}


def _download(url, dest, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 404:
                return False
            if i == tries - 1:
                raise
            time.sleep(3)
    return False


def fetch(kind, year):
    """kind: usage | box. Returns a DataFrame or None if that season is not published."""
    CDIR.mkdir(parents=True, exist_ok=True)
    name = {"usage": f"espn_cfb_usage_players/usage_players_{year}.parquet",
            "box": f"espn_cfb_player_box/player_box_{year}.parquet"}[kind]
    p = CDIR / f"{kind}_{year}.parquet"
    if not p.exists():
        if not _download(f"{BASE}/{name}", p):
            return None
    return pd.read_parquet(p)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def box_season(year):
    b = fetch("box", year)
    if b is None:
        return None
    b = b[b.athlete_name != "Team"].copy()
    cp = b["completions/passingAttempts"].astype(str).str.extract(r"^(\d+)/(\d+)$")
    b["pass_cmp"], b["pass_att"] = num(cp[0]), num(cp[1])
    for c, n in [("passingYards", "pass_yds"), ("passingTouchdowns", "pass_td"), ("interceptions", "pass_int"),
                 ("rushingAttempts", "rush_att"), ("rushingYards", "rush_yds"), ("rushingTouchdowns", "rush_td"),
                 ("receptions", "rec"), ("receivingYards", "rec_yds"), ("receivingTouchdowns", "rec_td")]:
        b[n] = num(b[c])
    # interceptions column also holds defensive interceptions; only count it on passing rows
    b.loc[b.category != "passing", "pass_int"] = np.nan
    g = b.groupby(["athlete_id", "team_id"], as_index=False).agg(
        games=("game_id", "nunique"), pass_cmp=("pass_cmp", "sum"), pass_att=("pass_att", "sum"), pass_yds=("pass_yds", "sum"),
        pass_td=("pass_td", "sum"), pass_int=("pass_int", "sum"), rush_att=("rush_att", "sum"), rush_yds=("rush_yds", "sum"),
        rush_td=("rush_td", "sum"), rec=("rec", "sum"), rec_yds=("rec_yds", "sum"), rec_td=("rec_td", "sum"))
    g["season"] = year
    t = g.groupby("team_id")[["rush_yds", "rush_td", "rec_yds", "rec_td", "pass_att"]].sum().add_prefix("team_").reset_index()
    return g.merge(t, on="team_id")


def build_player_seasons():
    parts = []
    for y in range(FIRST_SEASON, rm.LAST_COMPLETE_SEASON + 1):
        u, b = fetch("usage", y), box_season(y)
        if u is None or b is None:
            continue
        u = u[u.position_group.isin(POSG) | u.position_group.isna()].copy()      # seasons before 2014 carry no position label
        u["athlete_id"] = num(u.player_id)
        b["athlete_id"] = num(b.athlete_id)
        j = u.merge(b.drop(columns=["season"]), left_on=["athlete_id", "pos_team_id"], right_on=["athlete_id", "team_id"],
                    how="left", suffixes=("", "_box"))
        parts.append(j)
        print(f"  college {y}: {len(j)} player-seasons", flush=True)
    ps = pd.concat(parts, ignore_index=True)
    ps["nk"] = ps.player_name.map(rm.name_key)
    ps["school_key"] = ps.pos_team.map(_school_tokens)
    for c in ["games", "rushes", "targets", "receptions", "rush_yards", "receiving_yards", "touchdowns", "rz_touches", "touches"]:
        ps[c] = num(ps[c])
    g = ps["games"].where(ps["games"] > 0)
    ps["gp"] = g.fillna(num(ps["games_box"]) if "games_box" in ps else np.nan)
    ps["ypg_scr"] = (ps.rush_yards.fillna(0) + ps.receiving_yards.fillna(0)) / ps.gp
    return ps


def _school_tokens(s):
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("&", " ").replace("st.", "state ")
    toks = re.findall(r"[a-z]+", s)
    toks = ["state" if t == "st" else t for t in toks]
    return frozenset(toks)


MASCOTS = None


def school_match(draft_college, team_name_tokens):
    """Share of the draft-list college's words found in the ESPN team name (which also carries a mascot)."""
    d = _school_tokens(draft_college)
    if not d:
        return 0.0
    return len(d & team_name_tokens) / len(d)


def season_features(r):
    """r: one player-season row -> feature dict (rates per game played)."""
    gp = r["gp"] if pd.notna(r["gp"]) and r["gp"] > 0 else np.nan
    f = {}
    rush_att = r["rush_att"] if pd.notna(r["rush_att"]) else r["rushes"]
    rush_yds = r["rush_yds"] if pd.notna(r["rush_yds"]) else r["rush_yards"]
    rec = r["rec"] if pd.notna(r["rec"]) else r["receptions"]
    rec_yds = r["rec_yds"] if pd.notna(r["rec_yds"]) else r["receiving_yards"]
    f["c_games"] = gp
    f["c_rec_ypg"] = rec_yds / gp
    f["c_rec_pg"] = rec / gp
    f["c_tgt_pg"] = r["targets"] / gp
    f["c_ypr"] = rec_yds / rec if pd.notna(rec) and rec >= 8 else np.nan                  # too few catches = noise
    f["c_ypt"] = rec_yds / r["targets"] if pd.notna(r["targets"]) and r["targets"] >= 10 else np.nan
    f["c_rush_ypg"] = rush_yds / gp
    f["c_rush_att_pg"] = rush_att / gp
    f["c_ypc"] = rush_yds / rush_att if pd.notna(rush_att) and rush_att >= 20 else np.nan
    z = lambda x: 0.0 if pd.isna(x) else x
    f["c_scr_ypg"] = (z(rush_yds) + z(rec_yds)) / gp
    f["c_td_pg"] = (r["touchdowns"] if pd.notna(r["touchdowns"]) else 0) / gp
    f["c_tgt_share"] = r["target_share"]
    f["c_touch_share"] = r["touch_share"]
    f["c_epa_opp"] = r["epa_per_opportunity"]
    f["c_explosive"] = r["explosive_rate"]
    f["c_success"] = r["success_rate"]
    f["c_third_rate"] = r["third_down_rate"]
    f["c_rz_touch_pg"] = r["rz_touches"] / gp
    f["c_first_down_share"] = r["first_down_share"]
    f["c_rec_yd_share"] = rec_yds / r["team_rec_yds"] if pd.notna(r["team_rec_yds"]) and r["team_rec_yds"] > 0 else np.nan
    f["c_rec_td_share"] = (r["rec_td"] / r["team_rec_td"]) if pd.notna(r["team_rec_td"]) and r["team_rec_td"] > 0 and pd.notna(r["rec_td"]) else np.nan
    f["c_dom"] = np.nanmean([f["c_rec_yd_share"], f["c_rec_td_share"]]) if not (pd.isna(f["c_rec_yd_share"]) and pd.isna(f["c_rec_td_share"])) else np.nan
    f["c_rush_yd_share"] = rush_yds / r["team_rush_yds"] if pd.notna(r["team_rush_yds"]) and r["team_rush_yds"] > 0 else np.nan
    scr_team = z(r["team_rush_yds"]) + z(r["team_rec_yds"]) if pd.notna(r["team_rush_yds"]) else np.nan
    f["c_scr_yd_share"] = (z(rush_yds) + z(rec_yds)) / scr_team if pd.notna(scr_team) and scr_team > 0 else np.nan
    # passing
    pa = r["pass_att"] if pd.notna(r["pass_att"]) and r["pass_att"] >= 50 else np.nan      # passing rates only for real passers
    f["c_pass_att_pg"] = pa / gp if pd.notna(pa) else np.nan
    f["c_ypa"] = r["pass_yds"] / pa if pd.notna(pa) else np.nan
    f["c_cmp_pct"] = 100 * r["pass_cmp"] / pa if pd.notna(pa) else np.nan
    f["c_pass_td_pg"] = r["pass_td"] / gp if pd.notna(pa) else np.nan
    f["c_int_pct"] = 100 * r["pass_int"] / pa if pd.notna(pa) else np.nan
    f["c_pass_ypg"] = r["pass_yds"] / gp if pd.notna(pa) else np.nan
    return f


def attach_college(m):
    """Adds c_* columns to the rookie_model dataset (drafted players, 2000+)."""
    ps = build_player_seasons()
    ps["feat"] = [season_features(r) for r in ps.to_dict("records")]
    by_name = {k: g for k, g in ps.groupby("nk")}
    rows = []
    for r in m.to_dict("records"):
        out = {"_idx": r["_idx"], "c_found": 0}
        cls = int(r["class"])
        cand = by_name.get(rm.name_key(r["player"]))
        if cand is None or cls - 1 < FIRST_SEASON:
            rows.append(out)
            continue
        cand = cand[(cand.season <= cls - 1) & (cand.season >= cls - 3)].copy()
        if cand.empty:
            rows.append(out)
            continue
        cand["score"] = [school_match(r.get("college", ""), st) * 2 + (1.0 if s == cls - 1 else 0.0) + (0.5 if pg == r["pos"] else 0.0)
                         for st, s, pg in zip(cand.school_key, cand.season, cand.position_group)]
        best = cand.sort_values(["score", "season", "opportunities"], ascending=False).iloc[0]
        if school_match(r.get("college", ""), best.school_key) < 0.5 and cand.player_id.nunique() > 1:
            rows.append(out)                                          # ambiguous name at a different school
            continue
        career = ps[(ps.athlete_id == best.athlete_id) & (ps.season <= cls - 1)].sort_values("season")
        career = career[career.opportunities.fillna(0) >= 20]
        f = dict(best["feat"])
        out.update({k: v for k, v in f.items()})
        out["c_found"] = 1
        out["c_final_season"] = int(best.season)
        out["c_seasons"] = int(len(career))
        if len(career):
            doms = [season_features(x)["c_dom"] for x in career.to_dict("records")]
            out["c_best_dom"] = np.nanmax(doms) if not all(pd.isna(doms)) else np.nan
            age = r.get("age")
            for s, d in zip(career.season, doms):
                if pd.notna(d) and d >= 0.20 and pd.notna(age):
                    out["c_breakout_age"] = float(age) - (cls - 1 - int(s))
                    break
        rows.append(out)
    return rows


def college_features(m):
    """c_* columns for the rookie_model dataset, cached (delete rookie_cache/college_feats.pkl or use --refresh to rebuild)."""
    p = rm.CACHE / "college_feats.pkl"
    key = len(m)
    if p.exists():
        f = pd.read_pickle(p)
        if f.attrs.get("n") == key:
            return f
    m = m.reset_index(drop=True).copy()
    m["_idx"] = np.arange(len(m))
    f = pd.DataFrame(attach_college(m)).sort_values("_idx").reset_index(drop=True)
    f.attrs["n"] = key
    f.to_pickle(p)
    return f


def main():
    c, d, st = rm.load_raw()
    m = rm.build_dataset(c, d, st)
    m = m.reset_index(drop=True)
    m["_idx"] = np.arange(len(m))
    feats = pd.DataFrame(attach_college(m))
    m = m.merge(feats, on="_idx", how="left")
    m.to_pickle(rm.CACHE / "dataset_with_college.pkl")
    chk = m[(m["class"] >= 2006) & (m["class"] <= 2025)]
    print("\nCollege match rate by position (classes 2006-2025):")
    print(chk.groupby("pos").c_found.mean().round(2).to_string())
    print("\nBy class:")
    print(m[m["class"] >= 2005].groupby("class").c_found.mean().round(2).to_string())
    return m


if __name__ == "__main__":
    main()