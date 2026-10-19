#!/usr/bin/env python3
"""
Grade prop picks against what actually happened.

Usage (from the repo root):
    python grade_props.py output/prop_edges_2026_wk06.csv
    python grade_props.py output/history/prop_edges_2026_wk0*.csv       # several weeks / snapshots at once
    python grade_props.py FILE --line-type all                          # include alternate lines too
    python grade_props.py FILE --min-play 0.7                           # skip players we gave < 70% to play
    python grade_props.py FILE --week 6 --season 2026                   # only needed if the name has no wkNN

Needs: pip install nflreadpy pandas numpy.  The week and season come from the file name (prop_edges_2026_wk06.csv).

How a pick is graded
  * OVER wins if actual > line, UNDER wins if actual < line, equal = push.
  * A player with no stat row while his team DID play = did not play -> sportsbooks void the bet; counted as "void".
  * A game with no stats yet = not played -> skipped (so you can run it mid-week and only the finished games count).
  * ROI uses the listed odds on 1 unit; picks with no listed odds are left out of ROI only.
    Break-even is about 52.4% at -110.

Outputs: console tables, graded_props_detail.csv (every pick) and graded_props_summary.csv (every group).
"""
import argparse
import glob
import math
import re
import sys

import numpy as np
import pandas as pd

STAT_COL = {
    "pass_att": "attempts", "pass_cmp": "completions", "pass_yds": "passing_yards", "pass_td": "passing_tds",
    "pass_int": "passing_interceptions", "int": "passing_interceptions",
    "car": "carries", "rush_yds": "rushing_yards", "rush_td": "rushing_tds",
    "tgt": "targets", "rec": "receptions", "rec_yds": "receiving_yards", "rec_td": "receiving_tds",
}
MIN_N = 20          # groups smaller than this are flagged as noise
Z_BINS = [0, 0.5, 1.0, 1.5, 99]
Z_LAB = ["z<0.5", "0.5-1.0", "1.0-1.5", "z>=1.5"]


def norm(s):
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_player(cell):
    m = re.match(r"^(.*?)\s*\(([^,()]*),\s*([^()]*)\)\s*$", str(cell))
    return (m.group(1), m.group(2).strip(), m.group(3).strip()) if m else (str(cell), "", "")


def wilson(w, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def payout(odds):
    if pd.isna(odds) or odds == 0:
        return np.nan
    return odds / 100 if odds > 0 else 100 / abs(odds)


def load_picks(paths, season, week):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        m = re.search(r"(\d{4})_wk(\d+)", p)
        s = int(m.group(1)) if m else season
        w = int(m.group(2)) if m else week
        if s is None or w is None:
            sys.exit(f"Can't tell the season/week from '{p}'. Pass --season and --week.")
        d["season"], d["week"], d["src"] = s, w, p
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    parts = d.player.map(split_player)
    d["name"] = [x[0] for x in parts]
    d["pos"] = [x[1] for x in parts]
    d["team"] = [x[2] for x in parts]
    if "line_type" not in d:
        d["line_type"] = "standard"
    # the same pick can sit in several snapshots; keep one (the last file given)
    d = d.drop_duplicates(["season", "week", "name", "team", "stat", "line", "pick"], keep="last")
    return d


def load_actuals(pairs):
    import nflreadpy as nfl
    out = []
    for s in sorted({s for s, _ in pairs}):
        a = nfl.load_player_stats(seasons=[s]).to_pandas()
        a = a[a.season_type == "REG"] if "season_type" in a else a
        out.append(a)
    a = pd.concat(out, ignore_index=True)
    a["key"] = a.player_display_name.map(norm)
    return a


def grade(picks, actuals):
    rows = []
    for (s, w), g in picks.groupby(["season", "week"]):
        a = actuals[(actuals.season == s) & (actuals.week == w)]
        played_teams = set(a.team)
        idx = {(k, t): r for (k, t), r in zip(zip(a.key, a.team), a.to_dict("records"))}
        by_name = {}
        for r in a.to_dict("records"):
            by_name.setdefault(r["key"], []).append(r)
        for r in g.to_dict("records"):
            col = STAT_COL.get(r["stat"])
            if col is None:
                continue
            k = norm(r["name"])
            rec = idx.get((k, r["team"])) or (by_name.get(k, [None])[0] if len(by_name.get(k, [])) == 1 else None)
            if rec is None:
                r["result"] = "void" if r["team"] in played_teams else "pending"
                r["actual"] = np.nan
            else:
                r["actual"] = rec[col]
                r["opp"] = rec.get("opponent_team")
                if r["actual"] == r["line"]:
                    r["result"] = "push"
                elif (r["actual"] > r["line"]) == (r["pick"] == "OVER"):
                    r["result"] = "win"
                else:
                    r["result"] = "loss"
            rows.append(r)
    d = pd.DataFrame(rows)
    d["pay"] = d.odds.map(payout) if "odds" in d else np.nan
    d["profit"] = np.where(d.result == "win", d.pay, np.where(d.result == "loss", -1.0, np.where(d.result == "push", 0.0, np.nan)))
    d["our_err"] = (d.ours - d.actual).abs()
    d["line_err"] = (d.line - d.actual).abs()
    d["z_bucket"] = pd.cut(d.z, Z_BINS, labels=Z_LAB, right=False)
    return d


def summarize(d, by, label):
    g = d[d.result.isin(["win", "loss"])]
    if g.empty:
        return pd.DataFrame()
    rows = []
    for key, x in g.groupby(by, observed=True, dropna=True):
        n = len(x)
        w = int((x.result == "win").sum())
        lo, hi = wilson(w, n)
        pr = x.profit.dropna()
        full = d[d.result.isin(["win", "loss", "push"])]
        rows.append({
            "group": label, "value": key if not isinstance(key, tuple) else " | ".join(map(str, key)),
            "n": n, "wins": w, "hit_%": round(100 * w / n, 1), "ci_low_%": round(100 * lo, 1), "ci_high_%": round(100 * hi, 1),
            "roi_%": round(100 * pr.mean(), 1) if len(pr) else np.nan,
            "our_mae": round(x.our_err.mean(), 2), "line_mae": round(x.line_err.mean(), 2),
            "we_closer_%": round(100 * (x.our_err < x.line_err).mean(), 1),
            "small_sample": n < MIN_N,
        })
    return pd.DataFrame(rows)


def show(df, title):
    if df.empty:
        return
    print(f"\n=== {title} ===")
    print(df.drop(columns=["group"]).sort_values("n", ascending=False).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="prop_edges CSV file(s) or globs")
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--line-type", default="standard", choices=["standard", "alternate", "all"])
    ap.add_argument("--min-play", type=float, default=0.0, help="skip picks where p_play is below this (e.g. 0.7)")
    a = ap.parse_args()

    paths = sorted({p for f in a.files for p in (glob.glob(f) or [f])})
    picks = load_picks(paths, a.season, a.week)
    if a.line_type != "all":
        picks = picks[picks.line_type == a.line_type]
    if a.min_play > 0 and "p_play" in picks:
        picks = picks[picks.p_play.fillna(1) >= a.min_play]
    print(f"{len(picks)} picks loaded from {len(paths)} file(s); line type: {a.line_type}")

    actuals = load_actuals(set(zip(picks.season, picks.week)))
    d = grade(picks, actuals)
    d.drop(columns=["key"], errors="ignore").to_csv("graded_props_detail.csv", index=False)

    counts = d.result.value_counts().to_dict()
    print("results:", counts, "(void = player didn't play; pending = game not played yet)")
    played = d[d.result.isin(["win", "loss"])]
    if played.empty:
        sys.exit("Nothing graded yet - the games may not have finished (or no names matched).")

    allr = summarize(played.assign(all="ALL"), ["all"], "overall")
    groups = [
        ("stat", ["stat"], "Prop type"),
        ("pos", ["pos"], "Position"),
        ("team", ["team"], "Team (player's team)"),
        ("opp", ["opp"], "Opponent defense"),
        ("pick", ["pick"], "Over vs Under"),
        ("z_bucket", ["z_bucket"], "Edge size (z)  - do bigger edges hit more?"),
        ("stat_pick", ["stat", "pick"], "Prop type x Over/Under"),
        ("line_type", ["line_type"], "Line type"),
    ]
    out = [allr]
    show(allr, "OVERALL")
    for _, by, title in groups:
        r = summarize(played, by, title)
        out.append(r)
        show(r, title)
    pd.concat(out).to_csv("graded_props_summary.csv", index=False)

    print("\nHow to read this: hit_% needs to beat ~52.4% to profit at -110; ci_low/high is the 95% range, so a wide range")
    print("means the sample is too small to trust. we_closer_% = how often our number was nearer the actual result than the")
    print("book's line (over 50% = we beat the book on accuracy). Groups flagged small_sample (n<20) are mostly noise.")
    print("Wrote graded_props_detail.csv and graded_props_summary.csv")


if __name__ == "__main__":
    main()