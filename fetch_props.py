#!/usr/bin/env python3
"""
Fetch NFL player-prop lines from SharpAPI (free tier: DraftKings + FanDuel, 12 requests/min) and write the file
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
df.to_csv("props_raw.csv", index=False)
print(f"\n{len(df)} full-game prop lines for games before {end:%a %b %d}; "
      f"{df.player_name.nunique()} players, {df.event_id.nunique()} games, books: {sorted(df.sportsbook.unique())}")

# one row per player/stat: consensus (median) line across books, plus each book's line and the over/under prices
keys = ["event_id", "player_name", "stat"]
over = df[df.side == "over"]
g = over.groupby(keys)
res = g.line.median().rename("line").to_frame()
res["books"] = g.sportsbook.agg(lambda s: "/".join(sorted(set(s))))
res["n_books"] = g.sportsbook.nunique()
res["line_spread"] = g.line.max() - g.line.min()
for b in sorted(df.sportsbook.unique()):
    res[f"line_{b}"] = over[over.sportsbook == b].groupby(keys).line.first()
res["over_odds"] = g.odds_american.median()
und = df[df.side == "under"].groupby(keys).odds_american.median().rename("under_odds")
res = res.join(und).reset_index()
ev = df.drop_duplicates("event_id").set_index("event_id")
res["game"] = res.event_id.map(ev.away_team.astype(str) + " @ " + ev.home_team.astype(str))
res["start"] = res.event_id.map(ev.start)
res = res.rename(columns={"player_name": "player"}).sort_values(["start", "game", "player", "stat"])
cols = ["player", "stat", "line", "game", "start", "books", "n_books", "line_spread", "over_odds", "under_odds"] + \
       [c for c in res.columns if c.startswith("line_") and c != "line_spread"]
res[cols].to_csv("props_latest.csv", index=False)
print(f"Saved props_latest.csv ({len(res)} player-stat lines) and props_raw.csv.")
print("\nlines per stat:\n" + res.stat.value_counts().to_string())
print("\nsample:\n" + res[["player", "stat", "line", "game", "books"]].head(12).to_string(index=False))
print("\nNext: python weekly_projections.py --props-csv props_latest.csv")