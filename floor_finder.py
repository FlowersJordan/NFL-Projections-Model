#!/usr/bin/env python3
"""
Floor finder: which alternate (tease-down) DraftKings lines has a player cleared in EVERY game he played this season?

Example: the book's main line for a WR is 85.5 receiving yards, but he has never had fewer than 72 in a game he played,
so the 69.5 rung (priced far below the 85.5 line) has hit every week.

Usage (repo root):
    python floor_finder.py                                   # uses props_latest.csv (all rungs, standard + alternate)
    python floor_finder.py output/props_latest.csv --min-games 5
    python floor_finder.py --positions WR,TE --stats rec_yds,rec
    python floor_finder.py --season 2026

Needs: pip install nflreadpy pandas numpy

Rules
  * A game counts if the player was on the field for offensive snaps (nflverse snap counts), so a game where he played
    and had 0 catches counts as a 0 -- no hiding bad games. Games he didn't play (inactive/injured) are ignored.
  * Only games played BEFORE the line's kickoff are used (so you can also run it on an old props file to backtest it).
  * A rung "clears" if the player's lowest stat this season is above it (the line is x.5, so that means every game was an OVER).
  * The reported rung is the highest one he cleared every game (best price). hit_l17 is the same rung's record over his last
    17 games across seasons, a sturdier check than a 6-game streak. edge_pts compares that (smoothed) rate with the price.
  * "implied_%" comes from the odds and includes the book's vig, so the true break-even is slightly lower.

Writes floor_teases.csv.
"""
import argparse
import re
import sys

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("props", nargs="?", default="output/props_latest.csv")
    ap.add_argument("--season", type=int, default=None, help="default: the season of the props file's games")
    ap.add_argument("--min-games", type=int, default=4, help="games played this season required (default 4)")
    ap.add_argument("--positions", default="WR,TE,RB", help="comma list; add QB for passing props")
    ap.add_argument("--stats", default="rec_yds,rec,rush_yds,car", help="comma list of prop stats to check")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    pr = pd.read_csv(a.props)
    pr["start"] = pd.to_datetime(pr.start, utc=True)
    pr = pr[pr.over_odds.notna()]
    stats = [s.strip() for s in a.stats.split(",")]
    pr = pr[pr.stat.isin(stats)]
    if pr.empty:
        sys.exit("No usable rungs (with over odds) for those stats in the props file.")
    season = a.season or int(pr.start.min().year if pr.start.min().month > 6 else pr.start.min().year - 1)
    pos_ok = {p.strip().upper() for p in a.positions.split(",")}

    hist = load_history(season)
    hist = hist[hist.position.isin(pos_ok | ({"FB"} if "RB" in pos_ok else set()))]
    print(f"{len(pr)} rungs; history through {hist.gameday.max():%Y-%m-%d}; season {season}")

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
        v26 = cur[col].to_numpy()
        lo = float(v26.min())
        cleared = lad[lad.line < lo]
        if cleared.empty:
            continue
        r = cleared.sort_values("line").iloc[-1]          # highest cleared rung = best price
        std = lad[lad.line_type == "standard"].line
        last17 = h.tail(17)[col].to_numpy()
        hits17 = int((last17 > r.line).sum())
        n17 = len(last17)
        imp = implied(r.over_odds)
        smooth = (hits17 + 1) / (n17 + 2)
        rows.append({
            "player": player, "team": h.team.iloc[-1], "pos": h.position.iloc[-1], "game": game, "stat": stat,
            "games_played": len(cur), "season_min": lo, "season_avg": round(float(v26.mean()), 1),
            "season_games": " ".join(f"{x:g}" for x in v26),
            "standard_line": float(std.iloc[0]) if len(std) else np.nan,
            "tease_line": float(r.line),
            "gap_to_standard": round(float(std.iloc[0]) - float(r.line), 1) if len(std) else np.nan,
            "over_odds": float(r.over_odds), "implied_%": round(100 * imp, 1),
            "hit_l17": f"{hits17}/{n17}", "l17_smoothed_%": round(100 * smooth, 1),
            "edge_pts": round(100 * (smooth - imp), 1),
            "lowest_snap_%": round(float(cur.offense_pct.min() * (100 if cur.offense_pct.max() <= 1 else 1)), 0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("No player cleared a rung in every game he played with those filters.")
    out = out.sort_values(["edge_pts", "games_played"], ascending=False).reset_index(drop=True)
    out.to_csv("floor_teases.csv", index=False)
    show = out.drop(columns=["season_games"]).head(a.top)
    print(f"\n{len(out)} player/stat floors found (showing top {len(show)} by edge_pts):\n")
    print(show.to_string(index=False))
    print("\nWrote floor_teases.csv (includes each game's number in season_games).")
    print("Caution: 4-7 game streaks are short. A -300 price needs ~75% to break even; weigh hit_l17 and snap % (a low")
    print("lowest_snap_% means he was hurt or benched early in a game, a risk that can repeat).")


if __name__ == "__main__":
    main()