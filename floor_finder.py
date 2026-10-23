#!/usr/bin/env python3
"""
Floor finder: which lines has a player cleared in EVERY game he played this season, and where is the best price?

  OVER  "floor" bets: a rung BELOW his season low (the book's line is 85.5 receiving yards, but he has never had fewer than
        72 in a game he played, so Over 69.5 has hit every week).
  UNDER "ceiling" bets: a rung ABOVE his season high (he has never topped 61 yards, so Under 65.5 has hit every week).
        Books mostly quote Unders on the main line only, so ceiling bets are rarer.

With props_books.csv (DraftKings + FanDuel side by side) every rung is shopped across both books and the best price is shown.

Usage (repo root):
    python floor_finder.py                                   # props_books.csv (both books), else props_latest.csv; checks the current folder, then output/
    python floor_finder.py props_books.csv --side over       # floors only (or --side under for ceilings)
    python floor_finder.py --sort odds                       # best payout first (default: best expected value first)
    python floor_finder.py --positions WR,TE --stats rec_yds,rec --min-games 5 --min-odds -300

Needs: pip install nflreadpy pandas numpy

Rules
  * A game counts if the player was on the field for offensive snaps (nflverse snap counts), so a game where he played
    and had 0 catches counts as a 0 -- no hiding bad games. Games he didn't play (inactive/injured) are ignored.
  * Only games played BEFORE the line's kickoff are used (so you can also run it on an old props file to backtest it).
  * A rung "clears" if every game this season went the right way (lines are x.5, so no pushes).
  * Per player and stat the best-priced cleared rung is reported, taking the better of the two books. other_odds is the other
    book's price on the same rung, so you can see what shopping gained.
  * hit_l17 is that rung's record over his last 17 games across seasons, a sturdier check than a 5-game streak.
    ev_pct = (smoothed hit rate x payout) - 1, using (hits+1)/(n+2) on those 17 games: a conservative guess, not a promise.
  * cushion = how far the line sits beyond his worst game (floor - line for Overs, line - ceiling for Unders).
  * "implied_%" comes from the odds and includes the book's vig, so the true break-even is slightly lower.

Writes floor_teases.csv.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAT_COL = {
    "rec_yds": "receiving_yards", "rec": "receptions", "rush_yds": "rushing_yards", "car": "carries",
    "pass_yds": "passing_yards", "pass_att": "attempts", "tgt": "targets",
    "rec_td": "receiving_tds", "rush_td": "rushing_tds", "pass_td": "passing_tds",
}
SKILL = {"WR", "TE", "RB", "FB"}


def norm(s):
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def implied(o):
    if pd.isna(o) or o == 0:
        return np.nan
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def load_history(season):
    import nflreadpy as nfl
    seasons = [season - 1, season]
    st = nfl.load_player_stats(seasons=seasons).to_pandas()
    st = st[st.season_type == "REG"] if "season_type" in st else st
    sc = nfl.load_schedules(seasons=seasons).to_pandas()
    sc = sc[sc.game_type == "REG"][["game_id", "season", "week", "gameday", "home_team", "away_team"]]
    days = pd.concat([
        sc.rename(columns={"home_team": "team"})[["season", "week", "team", "gameday"]],
        sc.rename(columns={"away_team": "team"})[["season", "week", "team", "gameday"]]])
    days["gameday"] = pd.to_datetime(days.gameday)

    ro = nfl.load_rosters(seasons=seasons).to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    sn = nfl.load_snap_counts(seasons=seasons).to_pandas()
    sn = sn[sn.game_type == "REG"].merge(ro, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    sn = sn.rename(columns={"gsis_id": "player_id"})
    sn = sn[sn.offense_snaps > 0].drop_duplicates(["player_id", "season", "week"])
    sn = sn[["player_id", "season", "week", "team", "position", "offense_pct"]]

    cols = ["player_id", "season", "week", "player_display_name"] + sorted(set(STAT_COL.values()))
    g = sn.merge(st[cols], on=["player_id", "season", "week"], how="left")
    for c in set(STAT_COL.values()):
        g[c] = g[c].fillna(0.0)                 # played but no stat line = a genuine zero
    g["player_display_name"] = g.player_display_name.fillna("")
    # names for players who had no stat row at all
    nm = st.drop_duplicates("player_id").set_index("player_id").player_display_name
    g["name"] = np.where(g.player_display_name != "", g.player_display_name, g.player_id.map(nm))
    g["key"] = g["name"].map(norm)
    g = g.merge(days, on=["season", "week", "team"], how="left")
    return g.dropna(subset=["gameday"]).sort_values(["player_id", "gameday"])


def dec(o):
    return 1 + (o / 100 if o > 0 else 100 / abs(o))


def load_props(path):
    pr = pd.read_csv(path)
    if "book" not in pr.columns:                       # DraftKings-only props_latest.csv
        pr["book"] = pr["books"] if "books" in pr.columns else "draftkings"
    pr["start"] = pd.to_datetime(pr.start, utc=True)
    return pr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("props", nargs="?", default=None)
    ap.add_argument("--season", type=int, default=None, help="default: the season of the props file's games")
    ap.add_argument("--min-games", type=int, default=4, help="games played this season required (default 4)")
    ap.add_argument("--positions", default="WR,TE,RB", help="comma list; add QB for passing props")
    ap.add_argument("--stats", default="rec_yds,rec,rush_yds,car", help="comma list of prop stats to check")
    ap.add_argument("--side", default="both", choices=["over", "under", "both"])
    ap.add_argument("--sort", default="ev", choices=["ev", "odds", "cushion"], help="ev (default), odds (best payout), cushion")
    ap.add_argument("--min-odds", type=float, default=None, help="drop bets priced worse than this (e.g. -300)")
    ap.add_argument("--top", type=int, default=25, help="rows printed per side")
    a = ap.parse_args()

    path = a.props
    if path is None:
        path = next((p for p in ("props_books.csv", "output/props_books.csv", "props_latest.csv", "output/props_latest.csv") if Path(p).exists()), None)
        if path is None:
            sys.exit("No props file found. Run fetch_props.py first or pass the path.")
    pr = load_props(path)
    stats = [s.strip() for s in a.stats.split(",")]
    pr = pr[pr.stat.isin(stats)]
    if pr.empty:
        sys.exit("No rungs for those stats in the props file.")
    season = a.season or int(pr.start.min().year if pr.start.min().month > 6 else pr.start.min().year - 1)
    pos_ok = {p.strip().upper() for p in a.positions.split(",")}
    hist = load_history(season)
    hist = hist[hist.position.isin(pos_ok | ({"FB"} if "RB" in pos_ok else set()))]
    print(f"{path}: {len(pr)} rungs, books: {sorted(pr.book.unique())}; history through {hist.gameday.max():%Y-%m-%d}; season {season}")

    rows = []
    for (player, stat, game), lad in pr.groupby(["player", "stat", "game"]):
        start = lad.start.min().tz_convert(None).normalize()
        cand = hist[(hist.key == norm(player)) & (hist.gameday < start)]
        if cand.empty:
            continue
        pid = cand[cand.season == season].player_id.value_counts()
        pid = pid.index[0] if len(pid) else cand.player_id.iloc[-1]
        h = cand[cand.player_id == pid]
        cur = h[h.season == season]
        if len(cur) < a.min_games:
            continue
        col = STAT_COL[stat]
        v = cur[col].to_numpy()
        last17 = h.tail(17)[col].to_numpy()
        std = lad[lad.is_main] if "is_main" in lad else lad[lad.get("line_type", "") == "standard"]
        main_line = float(std.line.mode().iloc[0]) if len(std) else np.nan
        snap_lo = round(float(cur.offense_pct.min() * (100 if cur.offense_pct.max() <= 1 else 1)), 0)
        sides = [("OVER", "over_odds"), ("UNDER", "under_odds")] if a.side == "both" else \
                [("OVER", "over_odds")] if a.side == "over" else [("UNDER", "under_odds")]
        for side, ocol in sides:
            q = lad[lad[ocol].notna()].copy()
            q = q[q.line < v.min()] if side == "OVER" else q[q.line > v.max()]
            if q.empty:
                continue
            q["dec"] = q[ocol].map(dec)
            q = q.sort_values(["dec"], ascending=False)
            best = q.iloc[0]                                            # best price among all cleared rungs and both books
            same = lad[(lad.line == best.line) & (lad.book != best.book) & lad[ocol].notna()]
            other = same.iloc[0] if len(same) else None
            hits = int((last17 > best.line).sum()) if side == "OVER" else int((last17 < best.line).sum())
            n17 = len(last17)
            odds = float(best[ocol])
            imp = implied(odds)
            smooth = (hits + 1) / (n17 + 2)
            flags = (v > best.line) if side == "OVER" else (v < best.line)
            streak = 0
            for ok in flags[::-1]:
                if not ok:
                    break
                streak += 1
            rows.append({
                "player": player, "team": h.team.iloc[-1], "pos": h.position.iloc[-1], "game": game, "stat": stat,
                "bet": f"{side} {best.line:g}", "side": side, "book": best.book, "line": float(best.line), "odds": odds,
                "other_book": other.book if other is not None else None, "other_odds": float(other[ocol]) if other is not None else np.nan,
                "main_line": main_line, "games_played": len(cur), "season_min": float(v.min()), "season_max": float(v.max()),
                "season_avg": round(float(v.mean()), 1),
                "cushion": round(float(v.min() - best.line) if side == "OVER" else float(best.line - v.max()), 1),
                "implied_%": round(100 * imp, 1), "hit_l17": f"{hits}/{n17}", "l17_smoothed_%": round(100 * smooth, 1),
                "ev_pct": round(100 * (smooth * dec(odds) - 1), 1),
                "lowest_snap_%": snap_lo, "season_games": " ".join(f"{x:g}" for x in v),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("No player cleared a rung in every game he played with those filters.")
    if a.min_odds is not None:
        out = out[out.odds >= a.min_odds]
    key = {"ev": ["ev_pct", "games_played"], "odds": ["odds", "ev_pct"], "cushion": ["cushion", "ev_pct"]}[a.sort]
    out = out.sort_values(key, ascending=False).reset_index(drop=True)
    out.to_csv("floor_teases.csv", index=False)
    show = ["player", "team", "stat", "bet", "book", "odds", "other_book", "other_odds", "main_line", "season_min", "season_max",
            "games_played", "cushion", "hit_l17", "implied_%", "ev_pct", "lowest_snap_%"]
    for side, title in (("OVER", "FLOOR BETS (Over a line he has cleared every game)"),
                        ("UNDER", "CEILING BETS (Under a line he has stayed below every game)")):
        t = out[out.side == side].head(a.top)
        print(f"\n=== {title}: {int((out.side == side).sum())} found, sorted by {a.sort} ===")
        print(t[show].to_string(index=False) if len(t) else "none")
    print("\nWrote floor_teases.csv (each game's number is in season_games).")
    print("Caution: a 5-6 game streak is short. A -300 price needs ~75% to break even (see implied_%), and week 6's floors hit only 7 of 13.")
    print("Weigh cushion, hit_l17 and lowest_snap_% (a low snap % means he left a game hurt or benched, which can repeat).")


if __name__ == "__main__":
    main()