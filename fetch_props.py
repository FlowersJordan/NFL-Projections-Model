#!/usr/bin/env python3
"""
Fetch NFL player-prop lines from SharpAPI (free tier, 12 requests/min); DraftKings drives the model file; FanDuel is kept alongside for line shopping, and write the files
weekly_projections.py reads with --props-csv.

Setup:   pip install requests python-dotenv     and put SHARPAPI_KEY=... in a .env file (or the environment / a GitHub secret)
Run:     python fetch_props.py                  -> props_latest.csv (+ props_raw.csv with every line, both books)
Use:     python weekly_projections.py --props-csv props_latest.csv

Only full-game lines are kept (no 1st-half / quarter markets) for games starting in the next 8 days.
It prints every prop market type it saw and how each was mapped, so anything unmapped is visible.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
KEY = os.environ.get("SHARPAPI_KEY")
if not KEY:
    sys.exit("Set SHARPAPI_KEY (in a .env file or the environment) first.")

BASE = "https://api.sharpapi.io/api/v1"
LEAGUE = "nfl"
PAUSE = 5.5                    # free tier = 12 requests/minute
DAYS_AHEAD = 8
S = requests.Session()
S.headers.update({"X-API-Key": KEY})

# market name (lower-case, SharpAPI) -> our stat code. Several spellings are tried; unknown ones are reported.
CANDIDATES = {
    "player_passing_yards": "pass_yds", "player_passing_attempts": "pass_att", "player_passing_touchdowns": "pass_td",
    "player_passing_interceptions": "int", "player_rushing_yards": "rush_yds", "player_rushing_attempts": "car",
    "player_rushing_touchdowns": "rush_td", "player_receiving_yards": "rec_yds", "player_receptions": "rec",
    "player_receiving_touchdowns": "rec_td", "player_targets": "tgt",
}
PERIOD = re.compile(r"^(1st|2nd|3rd|4th|first|second|third|fourth|half|quarter|q[1-4])|(_|^)(1h|2h|1q|2q|3q|4q)(_|$)|half|quarter")


def stat_from_text(s):
    """Map a market_type / stat_category string to our stat code (None if it isn't one we model)."""
    s = (s or "").lower().replace(" ", "_").replace("-", "_")
    if PERIOD.search(s) or "longest" in s or "combo" in s or "+" in s or "alt" in s.split("_"):
        return None
    pairs = [("passing_yards", "pass_yds"), ("pass_yards", "pass_yds"), ("passing_attempts", "pass_att"),
             ("pass_attempts", "pass_att"), ("passing_tds", "pass_td"), ("passing_touchdowns", "pass_td"),
             ("interceptions", "int"), ("rushing_yards", "rush_yds"), ("rush_yards", "rush_yds"),
             ("rushing_attempts", "car"), ("rush_attempts", "car"), ("carries", "car"),
             ("receiving_yards", "rec_yds"), ("receiving_touchdowns", "rec_td"), ("receptions", "rec"),
             ("rushing_touchdowns", "rush_td"), ("targets", "tgt")]
    for k, v in pairs:
        if k in s:
            return v
    return None


def get(path, **params):
    params = {k: v for k, v in params.items() if v is not None}
    for attempt in range(3):
        r = S.get(BASE + path, params=params, timeout=40)
        if r.status_code == 429:
            time.sleep(15)
            continue
        break
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {json.dumps(j)[:250]}"
    d = j.get("data", j) if isinstance(j, dict) else j
    return (d if isinstance(d, list) else [d]), None


def pull(market):
    rows, off = [], 0
    while True:
        d, err = get("/odds", league=LEAGUE, market=market, limit=500, offset=off)
        time.sleep(PAUSE)
        if err:
            return rows, err
        rows += d
        if len(d) < 500 or off > 4000:
            return rows, None
        off += 500


now = datetime.now(timezone.utc)
end = now + timedelta(days=DAYS_AHEAD)
allrows, report = [], []
print(f"Fetching NFL player props ({len(CANDIDATES)} markets, ~{int(len(CANDIDATES) * PAUSE)}s)...")
for market, stat in CANDIDATES.items():
    rows, err = pull(market)
    n_player = sum(1 for r in rows if r.get("player_name"))
    report.append((market, stat, len(rows), n_player, err))
    for r in rows:
        r["_market_asked"] = market
    allrows += rows
    print(f"  {market:<32} -> {stat:<8} {len(rows):>5} lines, {n_player:>5} with a player" + (f"   [{err}]" if err else ""))

if not allrows:
    sys.exit("No lines came back. Send me the output above.")

df = pd.DataFrame(allrows)
df = df[df.get("player_name").notna()].copy()
df["stat"] = [stat_from_text(m) or stat_from_text(c) for m, c in zip(df.get("market_type", ""), df.get("stat_category", ""))]
df = df[df.stat.notna()]
df["start"] = pd.to_datetime(df.get("event_start_time"), utc=True, errors="coerce")
df = df[(df.start >= now - timedelta(hours=1)) & (df.start <= end)]
df = df[df.get("is_live", False).fillna(False) == False] if "is_live" in df else df
df["line"] = pd.to_numeric(df.line, errors="coerce")
df = df.dropna(subset=["line"])
df["side"] = df.selection.astype(str).str.lower().str.extract(r"(over|under)")[0]
MODEL_BOOKS = ["draftkings"]         # the model's picks (props_latest.csv -> prop_edges) are measured against DraftKings only
SHOP_BOOKS = ["draftkings", "fanduel"]   # both books are kept for line shopping (props_books.csv)
BOOK_PRIORITY = ["draftkings"]
MAX_ONE_SIDED_ODDS = 250             # a one-sided price beyond this is an alternate rung, never the standard line
FLAGS_MAIN = ["is_main", "main", "is_main_line", "is_primary", "primary"]
FLAGS_ALT = ["is_alternate", "alternate", "is_alt", "alt"]

df["sportsbook"] = df.sportsbook.astype(str).str.lower()
df_all = df[df.sportsbook.isin(SHOP_BOOKS)].copy()          # DraftKings + FanDuel
df = df_all[df_all.sportsbook.isin(MODEL_BOOKS)].copy()     # DraftKings only: feeds the model exactly as before
df_all.to_csv("props_raw.csv", index=False)
print(f"\n{len(df)} full-game prop lines for games before {end:%a %b %d}; "
      f"{df_all.player_name.nunique()} players, {df_all.event_id.nunique()} games, books: {sorted(df_all.sportsbook.unique())}")

# ---------------------------------------------------------------------------------------------------------------
# Standard line vs alternate rungs. A book lists a whole ladder per player and stat (e.g. 179.5 ... 429.5 passing
# yards). The STANDARD line is the one a book prices as a real two-way market: both Over and Under quoted, at
# odds closest to even. The other rungs are ALTERNATE lines (one-sided "X+" ladders, priced far from even).
# If the data carries an explicit main/alternate flag, that is used instead.


def implied(odds):
    """American odds -> implied probability (NaN where missing)."""
    o = pd.to_numeric(odds, errors="coerce")
    return pd.Series([(100.0 / (x + 100.0) if x > 0 else -x / (-x + 100.0)) if pd.notna(x) else float("nan") for x in o],
                     index=o.index)


def _ladders(df):
    """df: one row per quoted selection (event_id, player_name, stat, sportsbook, line, side, odds_american).
    Returns (d, lad, mains): the rows, one row per book/player/stat/line with both prices, and each book's main line."""
    flag_main = next((c for c in FLAGS_MAIN if c in df.columns and df[c].dropna().astype(str).str.lower().isin(["true", "false"]).any()), None)
    flag_alt = next((c for c in FLAGS_ALT if c in df.columns and df[c].dropna().astype(str).str.lower().isin(["true", "false"]).any()), None)
    d = df.copy()
    d["_flagmain"] = None
    if flag_main:
        d["_flagmain"] = d[flag_main].astype(str).str.lower().eq("true")
    elif flag_alt:
        d["_flagmain"] = ~d[flag_alt].astype(str).str.lower().eq("true")
    keys = ["event_id", "player_name", "stat", "sportsbook", "line"]
    ov = d[d.side == "over"].groupby(keys).odds_american.median().rename("over_odds")
    un = d[d.side == "under"].groupby(keys).odds_american.median().rename("under_odds")
    fm = d.groupby(keys)._flagmain.agg(lambda s: bool(s.dropna().any()) if s.notna().any() else None).rename("flagmain")
    lad = pd.concat([ov, un, fm], axis=1).reset_index()
    lad["p_over"], lad["p_under"] = implied(lad.over_odds), implied(lad.under_odds)
    lad["two_sided"] = lad.p_over.notna() & lad.p_under.notna()
    lad["balance"] = (lad.p_over / (lad.p_over + lad.p_under) - 0.5).abs()
    lad.loc[~lad.two_sided, "balance"] = (lad.p_over - 0.5).abs()           # one-sided: closeness of the Over price to even

    bk = ["event_id", "player_name", "stat", "sportsbook"]
    has_flags = bool(flag_main or flag_alt)
    mains = {}
    for key, g in lad.groupby(bk):
        m = basis = None
        if has_flags:
            f = g[g.flagmain.eq(True)]
            if len(f):                                    # the book's own "main line" marker is authoritative
                m, basis = f.sort_values(["balance", "line"]).iloc[0], "flag"
        elif g.two_sided.any():
            m, basis = g[g.two_sided].sort_values(["balance", "line"]).iloc[0], "two_sided"
        else:
            m, basis = g.sort_values(["balance", "line"]).iloc[0], "one_sided"
        if m is not None and not m.two_sided:
            px = m.over_odds if pd.notna(m.over_odds) else m.under_odds
            if pd.isna(px) or abs(px) > MAX_ONE_SIDED_ODDS:
                m = None                                  # a far-from-even one-sided price is an alternate rung
        if m is not None:
            mains[key] = (float(m.line), basis)           # no entry = the book has no standard line for this stat
    return d, lad, mains


def build_book_table(df):
    """One row per book / player / stat / line (nothing blended across books): the input for line shopping."""
    d, lad, mains = _ladders(df)
    ev = d.drop_duplicates("event_id").set_index("event_id")
    out = lad.copy()
    out["is_main"] = [mains.get((e, p, s, b), (None,))[0] == ln for e, p, s, b, ln in
                      zip(out.event_id, out.player_name, out.stat, out.sportsbook, out.line)]
    out["game"] = [f"{ev.loc[e, 'away_team']} @ {ev.loc[e, 'home_team']}" for e in out.event_id]
    out["start"] = [ev.loc[e, "start"] for e in out.event_id]
    out = out.rename(columns={"player_name": "player", "sportsbook": "book"})
    return out[["game", "start", "player", "stat", "book", "line", "over_odds", "under_odds", "is_main"]] \
        .sort_values(["start", "game", "player", "stat", "line", "book"]).reset_index(drop=True)


def build_lines(df):
    """Returns one row per player/stat/line with line_type = standard | alternate (the model's DraftKings-only table)."""
    d, lad, mains = _ladders(df)
    ev = d.drop_duplicates("event_id").set_index("event_id")
    out = []
    for (e, p, st), g in lad.groupby(["event_id", "player_name", "stat"]):
        books = [b for b in BOOK_PRIORITY if (e, p, st, b) in mains] + sorted(
            {b for b in g.sportsbook if (e, p, st, b) in mains and b not in BOOK_PRIORITY})
        rank = {"flag": 0, "two_sided": 1, "one_sided": 2}
        books.sort(key=lambda b: (rank[mains[(e, p, st, b)][1]], BOOK_PRIORITY.index(b) if b in BOOK_PRIORITY else 9))
        if books:
            std_line, basis = mains[(e, p, st, books[0])]
        else:
            std_line, basis = None, "none"               # no standard line offered: every rung is an alternate
        base = {"player": p, "stat": st, "game": f"{ev.loc[e, 'away_team']} @ {ev.loc[e, 'home_team']}",
                "start": ev.loc[e, "start"], "main_basis": basis}
        for b in sorted(d.sportsbook.unique()):
            base[f"main_{b}"] = mains.get((e, p, st, b), (None, None))[0]
        for ln, gl in g.groupby("line"):
            row = dict(base, line=float(ln), line_type="standard" if (std_line is not None and float(ln) == std_line) else "alternate",
                       books="/".join(sorted(gl.sportsbook.unique())), n_books=gl.sportsbook.nunique(),
                       over_odds=gl.over_odds.median(), under_odds=gl.under_odds.median())
            out.append(row)
    res = pd.DataFrame(out)
    # alternate rungs that sit between the books' two different standard lines are still alternates of the chosen one
    return res.sort_values(["start", "game", "player", "stat", "line"]).reset_index(drop=True)


res = build_lines(df)
cols = ["player", "stat", "line", "line_type", "game", "start", "books", "n_books", "over_odds", "under_odds", "main_basis"] + \
       [c for c in res.columns if c.startswith("main_") and c != "main_basis"]
res[cols].to_csv("props_latest.csv", index=False)
std = res[res.line_type == "standard"]
print(f"Saved props_latest.csv: {len(std)} standard lines + {(res.line_type == 'alternate').sum()} alternate lines "
      f"({res.groupby(['player', 'stat']).ngroups} player-stats).")
print("how the standard line was identified:\n" + std.main_basis.value_counts().to_string())
print("\nstandard lines per stat:\n" + std.stat.value_counts().to_string())
print("\nexample ladders (standard line marked *):")
for (p, st), g in list(res.groupby(["player", "stat"]))[:4]:
    print(f"  {p} / {st}: " + ", ".join(f"{r.line:g}{'*' if r.line_type == 'standard' else ''}" for r in g.itertuples()))
extra = [c for c in df.columns if any(k in c.lower() for k in ("main", "alt", "primary", "segment", "type"))]
print("\nraw columns that might flag main/alternate lines:", extra)
# ---- both books side by side, for line shopping (python line_shop.py) -----------------------------------------------
bt = build_book_table(df_all)
bt.to_csv("props_books.csv", index=False)
m = bt[bt.is_main]
both = m.groupby(["game", "player", "stat"]).book.nunique()
print(f"\nSaved props_books.csv: {len(bt)} rungs across {bt.book.nunique()} books ({bt.book.value_counts().to_dict()}).")
print(f"{int((both == 2).sum())} player-stats have a main line at BOTH books; "
      f"{int(((m.groupby(['game','player','stat']).line.nunique() > 1) & (both == 2)).sum())} of those have different main lines.")
print("\nNext: python weekly_projections.py --props-csv props_latest.csv   then   python line_shop.py")