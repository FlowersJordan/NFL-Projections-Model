#!/usr/bin/env python3
"""
Weekly projected STAT LINES and expected fantasy points (TE-premium, full PPR),
using ONLY pre-game information, including injuries.

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
    his own injury report tag + practice status            <- hurt but playing -> discount
    work missing from his team because teammates are out   <- Out teammates -> more work for him
Touchdowns and interceptions use Poisson regression (counts); everything else uses OLS.

INJURIES (from the nflverse injury reports, 2018+):
  * Own status: Questionable / Doubtful tags and Limited / DNP practice are model inputs; the fitted
    coefficients say how much a player who DOES play is held back.
  * Play probability: a table learned from 2018-2021 reports gives P(plays | tag, practice).
    proj      = projection if he plays (already discounted for being hurt)
    p_play    = chance he plays
    proj_ev   = p_play x proj   (rankings are sorted by this)
  * Out teammates: for every team we sum the expected targets (by WR/TE/RB), RB carries and QB pass
    attempts that are missing, weighted by (1 - P(plays)) and by how recently the player was actually
    in the lineup (someone who has been out for a month is already baked into his teammates' recent
    form; a NEW absence is not). Players listed Out are dropped from the rankings.
  * inj_own / inj_team in the output = points added or removed by his own status / by teammates' absences.
  Run it Friday or Saturday: on Wednesday most tags haven't been issued yet.

ROSTER FILTER (who is actually available, from the nflverse weekly rosters + game logs):
  * Dropped outright: anyone whose roster status is not Active (reserve/IR, cut, suspended, retired,
    practice squad, ...), anyone missing from the roster file, and anyone who has changed teams.
    Game-day "inactive" (INA) in an earlier week only caps the play chance (32% of those players play next week).
  * Missed his team's last 2+ games without being on the injury report: his play chance is capped at
    P_MISSED_GAMES (0.40; about 44% of such regulars still played in 2018-2025). --drop-missed removes
    them instead. Pass --no-roster-filter to switch all of this off.
  * If the schedule's listed starting QB is dropped, the QB with the most recent passing volume takes over.

Usage:
    python weekly_projections.py                    # next unplayed week of the current season
    python weekly_projections.py --week 6           # a specific week (needs posted lines)
    python weekly_projections.py --backtest         # also run the week-by-week backtest (~3 min)
    python weekly_projections.py --te-premium 1.0   # 2.0 PPR for TEs, etc.
    python weekly_projections.py --pass-td 6        # 6-pt passing TDs
    python weekly_projections.py --drop-missed      # hard-drop (not just discount) regulars who missed 2+ games
    python weekly_projections.py --no-roster-filter # old behavior: trust the last-played team and no status check
    python weekly_projections.py --sleeper          # add Sleeper's projections as a column
    python weekly_projections.py --compare-csv espn.csv --compare-name ESPN
                                                    # any other source: CSV with columns
                                                    # player, proj [, pos, team]

OPPORTUNITIES: opportunities_<season>_wk<NN>.csv lists the top 5 selections per game (our total / spread vs
  Vegas, and players whose projection is furthest above or below what their form alone says). --props-csv adds
  a player-prop comparison (prop_edges_<season>_wk<NN>.csv).

GAME SCORES: each team's points are built from its players' projected stat lines (QB passing TDs, all
rushing TDs, passing and rushing yards, weighted by play chance) and printed next to Vegas's total and
spread for every game, plus saved to game_projections_<season>_wk<week>.csv. --backtest refits the
formula and prints how it scored against Vegas (it matches Vegas but does not beat it, as the stat lines
are anchored to the Vegas total).

Everything uses data available before kickoff, so backtest numbers are honest.
QBs are limited to the probable starter listed in the nflverse schedule (or his replacement, see above).
Known gaps: no depth-chart/snap-count info, no weather, late inactives and healthy scratches announced after the data updates, and players with <4 prior
games (rookies, new starters) get no projection. The backtest scores players who actually played.
"""
import argparse
import re
import warnings

import numpy as np
import pandas as pd
import nflreadpy as nfl
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import def_injuries as DI          # defender absences (free nflverse snap counts + injury reports)
except Exception:                      # file missing next to this script: run without it
    DI = None

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

CFG = {"te_premium": 0.5, "pass_td": 4.0, "def_inj": True}             # set from the command line in main()

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

# ---- DraftKings Classic scoring (no TE premium; INT and fumble lost cost 1; +3 for 100+ rush/rec yds, 300+ pass yds)
DK_W = {"passing_yards": 0.04, "passing_tds": 4.0, "passing_interceptions": -1.0,
        "rushing_yards": 0.1, "rushing_tds": 6.0, "receptions": 1.0,
        "receiving_yards": 0.1, "receiving_tds": 6.0}
DK_BONUS = {"passing_yards": 300, "rushing_yards": 100, "receiving_yards": 100}   # +3 pts at or above
# A mean projection can't see bonuses (a player projected for 85 yds still reaches 100 some weeks), so the
# chance of reaching each threshold is modeled as Gamma(shape k, mean = projected yards). The shapes were
# fitted on the 2022-2026 walk-forward backtest (projected yards vs the yards players actually had).
BONUS_SHAPE = {("QB", "passing_yards"): 7.5, ("QB", "rushing_yards"): 3.0, ("RB", "rushing_yards"): 1.5,
               ("RB", "receiving_yards"): 1.1, ("WR", "receiving_yards"): 1.5, ("TE", "receiving_yards"): 2.0}
BONUS_SHAPE_DEFAULT = {"passing_yards": 7.5, "rushing_yards": 1.5, "receiving_yards": 1.5}
DK_EXTRA = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost", "passing_2pt_conversions",
            "rushing_2pt_conversions", "receiving_2pt_conversions", "special_teams_tds"]

# ---- roster filter
AVAILABLE_STATUS = {"ACT"}   # roster statuses that can play; RES/IR, CUT, SUS, DEV, RET ... are dropped
# "INA" (game-day inactive) is NOT permanent: 0% play in the same week, but 32% play the next week
# (2018-2026 regulars), so an INA tag from an EARLIER week only caps the play chance.
P_PREV_INACTIVE = 0.35
P_MISSED_GAMES = 0.40        # play chance for regulars who missed MISSED_GAMES_RULE+ team games, not on the report
MISSED_GAMES_RULE = 2        # (measured 2018-2025: about 44% of them still played)

# ---- team points from projected stat lines: points = const + b x (QB passing TDs, rushing TDs, passing yds, rushing yds)
# Fitted on the 2022-2026 backtest (--backtest refits it); touchdowns carry the extra point and the field goals
# that come with long drives, and the players we don't project (backups, kickers) are absorbed by the constant.
# ---- best-opportunities report (calibrated on the 2022-2026 backtest)
ADJ_SD = 0.95            # sd of (our points - form-only points) across players
TOTAL_DIFF_SD, MARGIN_DIFF_SD = 2.8, 3.2    # sd of (our - Vegas) game total / margin across games
ADJ_HIT = [(3.0, 0.65), (2.0, 0.65), (1.0, 0.59), (0.0, 0.54)]   # |adjustment| >= x -> share that moved our way
PROP_SD = {("QB", "pass_yds"): 81.3, ("QB", "pass_td"): 1.1, ("QB", "pass_att"): 9.9, ("QB", "car"): 2.4,
           ("QB", "rush_yds"): 16.9, ("RB", "car"): 5.3, ("RB", "rush_yds"): 32.3, ("RB", "rush_td"): 0.59,
           ("RB", "tgt"): 2.0, ("RB", "rec"): 1.7, ("RB", "rec_yds"): 17.0, ("WR", "tgt"): 2.9, ("WR", "rec"): 2.2,
           ("WR", "rec_yds"): 34.0, ("WR", "rush_yds"): 4.4, ("TE", "tgt"): 2.4, ("TE", "rec"): 1.9,
           ("TE", "rec_yds"): 24.0}   # sd of (actual - projected) per stat; a prop edge is scored in these units
TEAM_PTS_X = ["ptd", "rtd", "pyd", "ryd"]
TEAM_PTS = {"const": 19.2165, "ptd": 11.8472, "rtd": 5.7382, "pyd": -0.0593, "ryd": -0.0422}

# ---- defender absences: only cornerbacks showed a consistent effect (more passing yards/TDs for the opposing
# QB and WRs); safeties, linebackers and linemen were noise in the 2022-2026 residual test, so they are only reported
DEF_CB_STATS = {"passing_yards", "passing_tds", "receptions", "receiving_yards", "receiving_tds"}

# ---- injury features
PPLAY_TRAIN_END = 2021     # play-probability table is learned from seasons <= this (backtest starts 2022)
INJ_OWN = ["inj_q", "inj_dbt", "pr_lim", "pr_dnp"]    # own Questionable / Doubtful / Limited / DNP flags
_MISS_RECV = ["m_tgt_WR", "m_tgt_TE", "m_tgt_RB", "m_att_QB"]
_MISS_RUSH = ["m_car_RB"]
_MISS_PASS = ["m_tgt_WR", "m_tgt_TE", "m_tgt_RB"]
MISS_ALL = ["m_tgt_WR", "m_tgt_TE", "m_tgt_RB", "m_car_RB", "m_att_QB"]   # expected usage missing from the team
MISS_FOR = {"targets": _MISS_RECV, "receptions": _MISS_RECV, "receiving_yards": _MISS_RECV,
            "receiving_tds": _MISS_RECV,
            "carries": _MISS_RUSH, "rushing_yards": _MISS_RUSH, "rushing_tds": _MISS_RUSH,
            "attempts": _MISS_PASS, "passing_yards": _MISS_PASS, "passing_tds": _MISS_PASS,
            "passing_interceptions": _MISS_PASS}


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
    # actual DraftKings points (for calibrating the DK projection): scored stats + fumbles/2pt/ST TDs + bonuses
    for c in DK_EXTRA:
        stats[c] = stats[c].fillna(0)
    stats["dk_bonus"] = 3.0 * ((stats.passing_yards >= 300).astype(float) + (stats.rushing_yards >= 100)
                               + (stats.receiving_yards >= 100))
    stats["dk_pts"] = (DK_W["passing_yards"] * stats.passing_yards + DK_W["passing_tds"] * stats.passing_tds
                       + DK_W["passing_interceptions"] * stats.passing_interceptions
                       + DK_W["rushing_yards"] * stats.rushing_yards + DK_W["rushing_tds"] * stats.rushing_tds
                       + DK_W["receptions"] * stats.receptions + DK_W["receiving_yards"] * stats.receiving_yards
                       + DK_W["receiving_tds"] * stats.receiving_tds
                       - (stats.sack_fumbles_lost + stats.rushing_fumbles_lost + stats.receiving_fumbles_lost)
                       + 2.0 * (stats.passing_2pt_conversions + stats.rushing_2pt_conversions
                                + stats.receiving_2pt_conversions)
                       + 6.0 * stats.special_teams_tds + stats.dk_bonus)
    stats = stats[["player_id", "player_display_name", "position", "season", "week",
                   "team", "pts", "dk_pts", "dk_bonus"] + STAT_COLS]

    sched = nfl.load_schedules(seasons=seasons).to_pandas()
    sched = sched[sched.game_type == "REG"].copy()
    sched["home_team"] = sched.home_team.replace(TEAM_FIX)
    sched["away_team"] = sched.away_team.replace(TEAM_FIX)
    sched["gameday"] = pd.to_datetime(sched.gameday)
    sched["played"] = sched.home_score.notna()

    # one row per team per game, from that team's point of view
    # nflverse spread_line: positive = home team favored
    home = sched.assign(team=sched.home_team, opp=sched.away_team, is_home=1,
                        team_spread=sched.spread_line, team_qb_id=sched.home_qb_id,
                        team_score=sched.home_score)
    away = sched.assign(team=sched.away_team, opp=sched.home_team, is_home=0,
                        team_spread=-sched.spread_line, team_qb_id=sched.away_qb_id,
                        team_score=sched.away_score)
    games = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "week", "gameday", "played", "team", "opp", "is_home",
         "team_spread", "total_line", "team_qb_id", "team_score"]]
    games["team_implied"] = games.total_line / 2 + games.team_spread / 2
    return stats, games


def build_frame(stats, games, season, week, max_gap_days, rosters=None):
    """History rows (played games) + one 'upcoming' row per active player for the target week."""
    hist = stats.merge(games, on=["season", "week", "team"], how="inner")
    hist = hist.dropna(subset=["team_spread", "total_line"])
    hist["upcoming"] = False
    hist["prev_team"] = None

    last = (hist.sort_values(["season", "week"]).groupby("player_id").tail(1)
            [["player_id", "player_display_name", "position", "team", "gameday"]]
            .rename(columns={"gameday": "last_gameday"}))
    ug = games[(games.season == season) & (games.week == week) & (~games.played)]
    last["prev_team"] = None
    if rosters is not None and (rosters.week <= week).any():
        # traded / signed players: an Active player whose roster team differs from the team he last played for
        # is projected with his NEW team's game this week (his own form carries over)
        r = rosters[rosters.week == rosters.week[rosters.week <= week].max()].set_index("player_id")
        rt, rs = last.player_id.map(r.roster_team), last.player_id.map(r.roster_status)
        mv = rs.isin(["ACT", "INA"]) & rt.notna() & (rt != last.team)
        last.loc[mv, "prev_team"] = last.loc[mv, "team"]
        last.loc[mv, "team"] = rt[mv]
    up = last.merge(ug, on="team")
    up = up[(up.gameday - up.last_gameday).dt.days <= max_gap_days].dropna(
        subset=["team_spread", "total_line"])
    # (QBs are narrowed to one starter per team later, in choose_starting_qbs, after availability checks)
    up["season"], up["week"] = season, week
    for c in ["pts", "dk_pts", "dk_bonus"] + STAT_COLS:
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

    # ---- "as of the end of that game" usage (includes the game itself); used to size what an
    #      injured player's absence takes away, even if he sat out the week being projected
    ea = (d.groupby("player_id")[["pts", "targets", "carries", "attempts"]]
            .ewm(halflife=HALFLIFE, min_periods=MIN_PRIOR_GAMES).mean()
            .droplevel(0).reindex(d.index))
    for c in ea.columns:
        d["ea_" + c] = ea[c]

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


# ----------------------------------------------------------------------------- injuries
def load_injuries_all(seasons):
    """nflverse injury reports for skill positions, one row per player-week (final report of the week)."""
    frames = []
    for s in seasons:
        for attempt in range(3):          # the nflverse download occasionally times out
            try:
                frames.append(nfl.load_injuries(seasons=s).to_pandas())
                break
            except Exception as e:
                if attempt == 2:
                    print(f"(injury report for {s} unavailable: {str(e)[:70]})")
    if not frames:
        return None
    inj = pd.concat(frames, ignore_index=True)
    inj = inj[(inj.game_type == "REG") & inj.position.isin(POS)].dropna(subset=["gsis_id", "season", "week"]).copy()
    inj["season"], inj["week"] = inj.season.astype(int), inj.week.astype(int)
    inj["team"] = inj.team.replace(TEAM_FIX)
    inj["report_status"] = inj.report_status.where(inj.report_status.isin(["Out", "Doubtful", "Questionable"]), None)
    ps = inj.practice_status.fillna("")
    inj["practice"] = np.select([ps.str.startswith("Full"), ps.str.startswith("Limited"),
                                 ps.str.startswith("Did Not")], ["full", "limited", "dnp"], default="")
    inj = inj.drop_duplicates(["season", "week", "gsis_id"], keep="last")
    return inj.rename(columns={"gsis_id": "player_id"})[
        ["season", "week", "team", "player_id", "full_name", "position", "report_status", "practice"]]


def prepare_injury_panel(inj, d, stats, games):
    """Injury rows + how recently each player was in the lineup, his as-of usage, and P(plays)."""
    tg = (games[["team", "season", "week"]].drop_duplicates()
          .sort_values(["team", "season", "week"]).reset_index(drop=True))
    tg["gidx"] = tg.groupby("team").cumcount()                   # team's game number across seasons
    p = inj.merge(tg, on=["team", "season", "week"], how="inner").reset_index(drop=True)
    p["rid"] = np.arange(len(p))
    have = stats[["player_id", "season", "week"]].drop_duplicates().assign(_h=1.0)

    # did he actually play that week? (only used to learn play rates from old seasons)
    p = p.merge(have, on=["player_id", "season", "week"], how="left").rename(columns={"_h": "played"})
    p["played"] = p.played.fillna(0.0)

    # presence = share of the team's previous 4 games he played in (recent games weigh more)
    num, den = np.zeros(len(p)), np.zeros(len(p))
    for lag in range(1, 5):
        prev = (tg.assign(gidx=tg.gidx + lag)[["team", "gidx", "season", "week"]]
                .rename(columns={"season": "ps", "week": "pw"}))
        j = p[["rid", "team", "gidx", "player_id"]].merge(prev, on=["team", "gidx"], how="left")
        j = j.merge(have.rename(columns={"season": "ps", "week": "pw", "_h": "_hh"}),
                    on=["player_id", "ps", "pw"], how="left")
        j = j.set_index("rid").reindex(p.rid)
        w = 0.5 ** ((lag - 1) / HALFLIFE)
        ex = j.ps.notna().to_numpy()
        num += np.where(ex, w * j._hh.fillna(0.0).to_numpy(), 0.0)
        den += np.where(ex, w, 0.0)
    p["presence"] = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0)

    # as-of usage: his EWMA targets / carries / attempts as of his last game BEFORE that week
    pool = d.loc[~d.upcoming, ["player_id", "season", "week", "ea_targets", "ea_carries", "ea_attempts", "ea_pts"]].copy()
    pool["tkey"] = pool.season * 100 + pool.week
    pool = pool.dropna(subset=["ea_pts"]).sort_values("tkey")
    p["tkey"] = p.season * 100 + p.week
    p = pd.merge_asof(p.sort_values("tkey"),
                      pool[["player_id", "tkey", "ea_targets", "ea_carries", "ea_attempts"]],
                      on="tkey", by="player_id", direction="backward", allow_exact_matches=False)
    p = p.sort_values("rid").reset_index(drop=True)

    # P(plays | report tag, practice status): learned from <= PPLAY_TRAIN_END among regulars with real
    # usage (a backup can play without recording a stat, which would look like "didn't play")
    regular = ((p.ea_targets.fillna(0) >= 2) | (p.ea_carries.fillna(0) >= 4) | (p.ea_attempts.fillna(0) >= 10))
    t = p[(p.season <= PPLAY_TRAIN_END) & (p.presence >= 0.5) & regular].copy()
    t["status"] = t.report_status.fillna("none")
    fine = t.groupby(["status", "practice"]).played.agg(["mean", "size"])
    coarse = t.groupby("status").played.mean()
    fine_d = {k: v["mean"] for k, v in fine.iterrows() if v["size"] >= 30}
    keys = zip(p.report_status.fillna("none"), p.practice)
    p["p_play"] = [fine_d.get(k, coarse.get(k[0], 1.0)) for k in keys]
    p.attrs["play_table"] = fine

    # work that goes missing with him: (1 - P(plays)) x how recently he was in the lineup x his usage
    w = (1.0 - p.p_play) * p.presence
    p["cm_tgt"] = w * p.ea_targets.fillna(0.0)
    p["cm_car"] = w * p.ea_carries.fillna(0.0)
    p["cm_att"] = w * p.ea_attempts.fillna(0.0)
    return p


def add_injury_features(d, inj, stats, games):
    """Adds own-injury flags and team 'missing usage' columns to every row (history and upcoming)."""
    if inj is None or inj.empty:
        for c in INJ_OWN + MISS_ALL:
            d[c] = 0.0
        d["p_play"], d["inj_status"], d["inj_practice"] = 1.0, None, ""
        return d, None
    p = prepare_injury_panel(inj, d, stats, games)

    q = pd.DataFrame({"season": p.season, "week": p.week, "team": p.team,
                      "m_tgt_WR": np.where(p.position == "WR", p.cm_tgt, 0.0),
                      "m_tgt_TE": np.where(p.position == "TE", p.cm_tgt, 0.0),
                      "m_tgt_RB": np.where(p.position == "RB", p.cm_tgt, 0.0),
                      "m_car_RB": np.where(p.position == "RB", p.cm_car, 0.0),
                      "m_att_QB": np.where(p.position == "QB", p.cm_att, 0.0)})
    tm = q.groupby(["season", "week", "team"], as_index=False).sum()
    d = d.merge(tm, on=["season", "week", "team"], how="left")
    d[MISS_ALL] = d[MISS_ALL].fillna(0.0)

    own = p[["season", "week", "player_id", "report_status", "practice", "p_play", "cm_tgt", "cm_car", "cm_att"]]
    own = own.rename(columns={"report_status": "inj_status", "practice": "inj_practice"})
    d = d.merge(own, on=["season", "week", "player_id"], how="left")
    for c in ("cm_tgt", "cm_car", "cm_att"):
        d[c] = d[c].fillna(0.0)
    # a player's own absence must not count as "work missing" for himself
    pos = d.position
    for pp in ("WR", "TE", "RB"):
        d[f"m_tgt_{pp}"] -= np.where(pos == pp, d.cm_tgt, 0.0)
    d["m_car_RB"] -= np.where(pos == "RB", d.cm_car, 0.0)
    d["m_att_QB"] -= np.where(pos == "QB", d.cm_att, 0.0)
    d[MISS_ALL] = d[MISS_ALL].clip(lower=0.0)
    d["p_play"] = d.p_play.fillna(1.0)
    d["inj_practice"] = d.inj_practice.fillna("")
    d["inj_q"] = (d.inj_status == "Questionable").astype(float)
    d["inj_dbt"] = (d.inj_status == "Doubtful").astype(float)
    d["pr_lim"] = (d.inj_practice == "limited").astype(float)
    d["pr_dnp"] = (d.inj_practice == "dnp").astype(float)
    return d.drop(columns=["cm_tgt", "cm_car", "cm_att"]), p


def report_play_rates(p):
    t = p.attrs.get("play_table")
    if t is None:
        return
    print(f"\n=== HOW OFTEN EACH DESIGNATION ACTUALLY PLAYS (regulars, {FIRST_SEASON}-{PPLAY_TRAIN_END}) ===")
    t = t.reset_index().rename(columns={"status": "report tag", "mean": "p_play", "size": "n"})
    t = t[t.n >= 30].copy()
    t["report tag"] = t["report tag"].replace({"none": "(no tag)"})
    t["practice"] = t.practice.replace({"": "(none)"})
    print(t.round(2).to_string(index=False))


# ----------------------------------------------------------------------------- roster filter
def load_rosters(season):
    """Weekly nflverse rosters for one season: player_id, week, roster_team, roster_status (None if unavailable)."""
    for attempt in range(3):
        try:
            r = nfl.load_rosters_weekly(seasons=season).to_pandas()
            break
        except Exception as e:
            if attempt == 2:
                print(f"(rosters for {season} unavailable: {str(e)[:70]})")
                return None
    r = r.dropna(subset=["gsis_id", "week"]).copy()
    if "game_type" in r:
        r = r[r.game_type.eq("REG") | r.game_type.isna()]
    r["week"] = r.week.astype(int)
    r["team"] = r.team.replace(TEAM_FIX)
    r = r.drop_duplicates(["week", "gsis_id"], keep="last")
    return r.rename(columns={"gsis_id": "player_id", "team": "roster_team", "status": "roster_status"})[
        ["player_id", "week", "roster_team", "roster_status"]]


def roster_filter(up, rosters, stats, games, inj, season, week, hard_drop_missed=False):
    """Returns (kept rows with roster_note / capped p_play, table of dropped players with reasons)."""
    up = up.copy()
    up["roster_note"] = ""
    if rosters is None or rosters.empty or not (rosters.week <= week).any():
        print("(rosters unavailable - roster filter skipped)")
        return up, pd.DataFrame(columns=["player", "pos", "team", "reason"])
    wu = int(rosters.week[rosters.week <= week].max())
    r = rosters[rosters.week == wu].set_index("player_id")
    status = up.player_id.map(r.roster_status)
    rteam = up.player_id.map(r.roster_team)
    reason = pd.Series("", index=up.index)
    reason[status.isna()] = "not on the roster file"
    ina = status.eq("INA")
    bad = status.notna() & ~status.isin(AVAILABLE_STATUS) & (~ina | (wu == week))   # INA this very week = inactive
    reason[bad] = "roster status " + status[bad].astype(str)
    moved = status.isin(AVAILABLE_STATUS) & rteam.notna() & (rteam != up.team)      # (movers are re-teamed in build_frame)
    reason[moved] = "moved to " + rteam[moved].astype(str)
    new_team = up.prev_team.notna() & (reason == "")
    up.loc[new_team, "roster_note"] = "joined " + up.team[new_team].astype(str) + " from " + up.prev_team[new_team].astype(str)

    # team games already played, most recent first; did the player appear in the last MISSED_GAMES_RULE of them?
    tg = games[games.played & ((games.season < season) | ((games.season == season) & (games.week < week)))]
    tg = tg[["team", "season", "week"]].drop_duplicates().sort_values(["team", "season", "week"])
    last = tg.groupby("team").tail(MISSED_GAMES_RULE)
    have = stats[["player_id", "season", "week"]].drop_duplicates().assign(_h=1)
    j = last.merge(have, on=["season", "week"], how="left")
    j = j.groupby(["team", "player_id"])._h.sum()                      # appearances in his team's last N games
    ngames = last.groupby("team").size()
    appeared = pd.Series(list(zip(up.team, up.player_id))).map(j).fillna(0).to_numpy()
    need = up.team.map(ngames).fillna(0).to_numpy()
    # "on the report" = carries an injury designation (Out / Doubtful / Questionable); a bare practice-participation
    # line (limited, DNP) with no designation does not explain two missed games
    listed = (set(inj.loc[(inj.season == season) & (inj.week == week) & inj.report_status.notna(), "player_id"])
              if inj is not None else set())
    missed = (need >= MISSED_GAMES_RULE) & (appeared == 0) & ~up.player_id.isin(listed).to_numpy() & (reason == "").to_numpy() & up.prev_team.isna().to_numpy()

    dropped = up[reason != ""].assign(reason=reason[reason != ""])
    if hard_drop_missed:
        dropped = pd.concat([dropped, up[missed].assign(reason=f"missed last {MISSED_GAMES_RULE}+ games, not on report")])
    keep = ~up.index.isin(dropped.index)
    up = up[keep].copy()
    soft = missed[keep]
    prev_ina = (ina & ~(bad | moved)).to_numpy()[keep] & ~up.player_id.isin(listed).to_numpy()
    up.loc[prev_ina, "p_play"] = up.loc[prev_ina, "p_play"].clip(upper=P_PREV_INACTIVE)
    up.loc[prev_ina & (up.roster_note == ""), "roster_note"] = f"inactive last week (week {wu}), not on report"
    if not hard_drop_missed:
        up.loc[soft, "p_play"] = up.loc[soft, "p_play"].clip(upper=P_MISSED_GAMES)
        up.loc[soft, "roster_note"] = f"missed last {MISSED_GAMES_RULE}+ games, not on report"
    dropped = dropped.rename(columns={"player_display_name": "player", "position": "pos"})[["player", "pos", "team", "reason"]]
    if prev_ina.sum():
        print(f"\nplay chance capped at {P_PREV_INACTIVE:.0%} (inactive in week {wu}, not on the injury report): "
              + ", ".join(f"{a} ({b})" for a, b in zip(up.loc[prev_ina, "player_display_name"], up.loc[prev_ina, "team"])))
    print(f"\n=== ROSTER FILTER (roster snapshot: {season} week {wu}) ===")
    print(f"dropped {len(dropped)} players who cannot play:")
    if len(dropped):
        print(dropped.sort_values(["reason", "team"]).to_string(index=False))
    if not hard_drop_missed and soft.sum():
        print(f"\nplay chance capped at {P_MISSED_GAMES:.0%} (missed {MISSED_GAMES_RULE}+ games, not on the report): "
              + ", ".join(f"{a} ({b})" for a, b in zip(up.loc[soft, "player_display_name"], up.loc[soft, "team"])))
    return up, dropped


def choose_starting_qbs(up):
    """One QB per team: the starter listed in the schedule if he is still in the pool, else the QB with the
    most recent passing volume. If the listed starter has not played for 2+ games without being on the injury
    report (the roster filter flags him), and a teammate QB has been playing, the teammate is used instead:
    the schedule's starter field is sometimes wrong. Teams with no listed starter keep every QB."""
    is_qb = up.position == "QB"
    keep = ~is_qb
    swaps = []
    for team, g in up[is_qb].groupby("team"):
        lst = g.team_qb_id.iloc[0]
        listed = g[g.player_id == lst]
        if pd.isna(lst):
            keep[g.index] = True
        elif not listed.empty:
            row = listed.iloc[0]
            alt = g[(g.player_id != lst) & (g.roster_note == "")] if "roster_note" in g else g.iloc[0:0]
            if str(row.get("roster_note", "")).startswith("missed last") and not alt.empty:
                best = alt.ea_attempts.fillna(0).idxmax()
                if alt.at[best, "ea_attempts"] > (row.ea_attempts if pd.notna(row.ea_attempts) else -1):
                    keep[best] = True
                    swaps.append((team, alt.at[best, "player_display_name"], row.player_display_name))
                    continue
            keep[listed.index] = True
        else:
            keep[g.ea_attempts.fillna(0).idxmax()] = True
    if swaps:
        print("\nStarting QB switched (the schedule's listed starter has not played in his team's last 2+ games and "
              "is not on the injury report): " + "; ".join(f"{t}: {new} instead of {old}" for t, new, old in swaps))
    return up[keep]


# ----------------------------------------------------------------------------- defender absences
def attach_defense_missing(d, season):
    """Adds opp_dm_cb / opp_dm_s / opp_dm_lb / opp_dm_dl: defensive snap share missing from the OPPOSING defense."""
    cols = ["opp_dm_cb", "opp_dm_s", "opp_dm_lb", "opp_dm_dl"]
    info = None
    if DI is not None and CFG.get("def_inj"):
        try:
            sn, ij = DI.load_defense_data(range(FIRST_SEASON, season + 1))
            m, players, _ = DI.build_missing(sn, ij)
            m = m.rename(columns={"team": "opp", **{f"dm_{g}": f"opp_dm_{g}" for g in DI.GROUPS}})
            d = d.merge(m, on=["season", "week", "opp"], how="left")
            info = players
        except Exception as e:
            print(f"(defender absences unavailable: {str(e)[:80]} - running without them)")
            CFG["def_inj"] = False
    elif DI is None:
        print("(def_injuries.py not found next to this script - running without defender absences)")
        CFG["def_inj"] = False
    for c in cols:
        if c not in d:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0)
    return d, info


def report_defense_missing(players, up, season, week):
    """Defenders (CB / S / LB / DL) expected to miss this week's games, for the defenses facing projected players."""
    if players is None or not len(up):
        return
    teams = set(up.opp)
    p = players[(players.season == season) & (players.week == week) & players.team.isin(teams) & (players.miss >= 0.15)]
    if p.empty:
        return
    print(f"\n=== DEFENDERS EXPECTED TO MISS WEEK {week} (miss = chance he sits x his recent snap share; 1.0 = a full-time starter) ===")
    show = p.sort_values(["team", "miss"], ascending=[True, False])[["team", "full_name", "group", "status", "practice", "share", "miss"]]
    show = show.rename(columns={"full_name": "defender", "share": "snap_share"})
    print(show.round(2).to_string(index=False))
    print("(Only cornerbacks feed the projections: a missing CB raises the opposing QB's and WRs' passing numbers, "
          "about +18 pass yds / +0.35 pass TD per full-time CB for the QB and +4 yds per WR in the 2022-2026 test.)")


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
def stat_features(s, ctx=True, inj=False):
    f = ["e_" + s]
    if s in USAGE_DRIVER:
        f.append("e_" + USAGE_DRIVER[s])
    if ctx:
        f += ["def_" + s, "team_implied", "team_spread", "is_home"]
    if inj:
        f += INJ_OWN + MISS_FOR[s]
        if CFG.get("def_inj") and s in DEF_CB_STATS:
            f.append("opp_dm_cb")          # cornerback snaps missing from the opposing defense
    return f


def fit_stat_models(train, ctx=True, inj=False):
    """One regression per (position, stat). Counts (TDs, INTs) are Poisson, the rest OLS."""
    fits = {}
    for pos in POS:
        t0 = train[train.position == pos]
        for s in MODEL_STATS[pos]:
            feats = stat_features(s, ctx, inj)
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


def stat_points(df, pos, prefix="", w=None):
    """Points implied by the stat columns (prefix '' = actual stats, 'p_' = projected), by source.
    w = scoring weights (default: your league's; pass DK_W for DraftKings)."""
    ms = MODEL_STATS[pos]
    w = w or score_weights(pos)
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


def dk_other_means(train):
    """Average DK points per game NOT explained by the modeled stats or the yardage bonuses."""
    out = {}
    for pos in POS:
        t = train[train.position == pos]
        out[pos] = float((t.dk_pts - t.dk_bonus - stat_points(t, pos, "", DK_W).sum(axis=1)).mean())
    return out


def expected_bonus(df, pos, prefix):
    """Expected DK yardage-bonus points: 3 x P(reaching 100 rush / 100 rec / 300 pass yds)."""
    from scipy.stats import gamma
    tot = pd.Series(0.0, index=df.index)
    for stat, thr in DK_BONUS.items():
        if stat in MODEL_STATS[pos]:
            mu = df[prefix + stat].clip(lower=0.5)
            k = BONUS_SHAPE.get((pos, stat), BONUS_SHAPE_DEFAULT[stat])
            tot = tot + 3.0 * pd.Series(gamma.sf(thr, a=k, scale=(mu / k).to_numpy()), index=df.index)
    return tot


def project_dk(df, prefix, dk_other):
    """Projected DraftKings points (if he plays) = DK weights x projected stats + expected bonuses + 'other'."""
    parts = []
    for pos in POS:
        sub = df[df.position == pos]
        sp = stat_points(sub, pos, prefix, DK_W).sum(axis=1, skipna=False)
        bonus = expected_bonus(sub, pos, prefix)
        parts.append(pd.DataFrame({"dk_proj": sp + bonus + dk_other[pos], "dk_bonus_ev": bonus}, index=sub.index))
    return pd.concat(parts).reindex(df.index)


# ----------------------------------------------------------------------------- DraftKings salaries
DK_TEAM_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE",
               "HST": "HOU", "LVR": "LV", "SD": "LAC", "STL": "LA"}


def load_dk_salaries(path):
    """DraftKings 'Export to CSV' (DKSalaries.csv) or the lineup-upload template, whose player list sits to the
    right of the entry columns. A plain CSV with name, position, team, salary columns also works."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f)]
    hdr = next((i for i, r in enumerate(rows) if "Salary" in r and "TeamAbbrev" in r and "Name" in r), None)
    recs = []
    if hdr is not None:
        h = rows[hdr]
        ix = {c: h.index(c) for c in ("Position", "Name", "Salary", "TeamAbbrev")}
        ia = h.index("AvgPointsPerGame") if "AvgPointsPerGame" in h else None
        ir = h.index("Roster Position") if "Roster Position" in h else None
        for r in rows[hdr + 1:]:
            if len(r) <= max(ix.values()) or not r[ix["Name"]] or not r[ix["Salary"]].replace(".", "").isdigit():
                continue
            if ir is not None and len(r) > ir and "CPT" in r[ir]:
                raise ValueError("this looks like a Showdown file (CPT slots) - export the Classic slate instead")
            try:
                avg = float(r[ia]) if ia is not None and len(r) > ia and r[ia] else np.nan
            except ValueError:
                avg = np.nan
            recs.append({"name": r[ix["Name"]], "pos": r[ix["Position"]], "team": r[ix["TeamAbbrev"]],
                         "salary": float(r[ix["Salary"]]), "dk_avg": avg})
        dk = pd.DataFrame(recs)
    else:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        ren = {"teamabbrev": "team", "position": "pos", "player": "name"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if not {"name", "salary"} <= set(df.columns):
            raise ValueError("couldn't find Name / Salary columns in that file")
        for c in ("pos", "team", "dk_avg"):
            if c not in df.columns:
                df[c] = np.nan
        dk = df[["name", "pos", "team", "salary", "dk_avg"]]
    if dk.empty:
        raise ValueError("no players found in that file")
    dk = dk[dk.pos.isin(POS) | dk.pos.isna()].copy()
    dk["team"] = dk.team.replace(DK_TEAM_FIX)
    dk["key"] = dk.name.map(name_key)
    return dk.drop_duplicates(["key", "pos", "team"]).reset_index(drop=True)


def attach_salary(up, dk):
    """Adds salary and DK's own season average to `up` (match on name + position, team breaks ties)."""
    up = up.copy()
    up["salary"], up["dk_avg"] = np.nan, np.nan
    idx = {}
    for r in dk.itertuples():
        idx.setdefault((r.key, r.pos), []).append(r)
        idx.setdefault((r.key, None), []).append(r)
    matched = set()
    for i in up.index:
        k, pos, team = name_key(up.at[i, "player_display_name"]), up.at[i, "position"], up.at[i, "team"]
        cands = idx.get((k, pos)) or idx.get((k, None)) or []
        if len(cands) > 1:
            cands = [c for c in cands if c.team == team]
        if len(cands) == 1:
            up.at[i, "salary"], up.at[i, "dk_avg"] = cands[0].salary, cands[0].dk_avg
            matched.add(cands[0].Index)
    return up, matched


def add_dk_value(up):
    """pts_per_k = dk_ev per $1,000. value_pts = projected points ABOVE what that salary 'normally buys' at the
    position on this slate (a line fitted of dk_ev on salary), so cheap-and-good AND mid-priced-and-great show up."""
    up = up.copy()
    k = up.salary / 1000.0
    up["pts_per_k"] = up.dk_ev / k
    up["value_pts"] = np.nan
    lines = {}
    for pos in POS:
        m = (up.position == pos) & up.salary.notna()
        fit_rows = m & (up.p_play >= 0.5)
        if fit_rows.sum() >= 8:
            X = sm.add_constant(k[fit_rows], has_constant="add")
            fit = sm.OLS(up.loc[fit_rows, "dk_ev"], X).fit()
            up.loc[m, "value_pts"] = up.loc[m, "dk_ev"] - (fit.params["const"] + fit.params.iloc[1] * k[m])
            lines[pos] = (float(fit.params["const"]), float(fit.params.iloc[1]), float(fit.rsquared), int(fit_rows.sum()))
    return up, lines


def report_dk(out, lines, dk, matched, panel, season, week, path):
    print("\n=== DRAFTKINGS VALUE (DK scoring: full PPR with NO TE premium, INT -1, +3 at 100 rush/rec or 300 pass yds) ===")
    print("dk_proj = projected DK points if he plays | dk_ev = p_play x dk_proj | pts_per_k = dk_ev per $1,000")
    print("value_pts = dk_ev minus the points that salary normally buys at that position on this slate\n")
    for pos, (a, b, r2, n) in lines.items():
        print(f"  {pos}: each extra $1,000 of salary buys about {b:+.2f} projected points "
              f"(R2 {r2:.2f}, n={n} players)")
    v = out[out.salary.notna()].copy()
    show = ["player", "team", "opp", "salary", "dk_proj", "p_play", "dk_ev", "pts_per_k", "value_pts", "dk_avg", "injury"]
    for pos, n in (("QB", 8), ("RB", 12), ("WR", 15), ("TE", 10)):
        t = v[(v.pos == pos) & (v.p_play >= 0.5)].sort_values("value_pts", ascending=False).head(n)
        print(f"\n--- {pos}: best values (sorted by value_pts) ---")
        print(t[show].to_string())
    print("\n--- Best points per $1,000 overall (players projected for 8+ DK points) ---")
    t = v[(v.dk_ev >= 8) & (v.p_play >= 0.5)].sort_values("pts_per_k", ascending=False).head(15)
    print(t[["player", "pos", "team", "opp", "salary", "dk_ev", "pts_per_k", "value_pts", "injury"]].to_string())

    out_keys = set()
    if panel is not None:
        cur = panel[(panel.season == season) & (panel.week == week) & (panel.report_status == "Out")]
        out_keys = {(name_key(n), t) for n, t in zip(cur.full_name, cur.team)}
    un = dk[~dk.index.isin(matched) & (dk.salary >= 4500)].sort_values("salary", ascending=False).head(25)
    if len(un):
        un = un.assign(reason=[("listed Out" if (r.key, r.team) in out_keys else "no projection (<4 games of history, "
                                "low usage, or not the listed starting QB)") for r in un.itertuples()])
        print("\n--- DraftKings players priced $4,500+ that I could NOT project (check these by hand) ---")
        print(un[["name", "pos", "team", "salary", "reason"]].to_string(index=False))
    cols = (["player", "pos", "team", "opp", "salary", "dk_proj", "p_play", "dk_ev", "pts_per_k", "value_pts", "dk_avg",
             "proj"] + [SHORT[s] for s in STAT_COLS] + ["injury"])
    v.sort_values(["pos", "value_pts"], ascending=[True, False])[cols].to_csv(path, index=False)
    print(f"\nSaved {len(v)} priced players to {path}")


def proj_with(fits, df, om, zero=()):
    """Projected points with some feature columns forced to 0 (to isolate what injuries contribute)."""
    u = df[["position"] + sorted({f for (_, _), (_, feats, _) in fits.items() for f in feats})].copy()
    for c in zero:
        u[c] = 0.0
    ps = predict_stats(fits, u, "x_")
    return project_points(ps.assign(position=u.position), "x_", om).proj


# ----------------------------------------------------------------------------- weekly backtest
def backtest_weekly(d, first_test=2022):
    """Walk forward one week at a time: fit on every earlier week, predict that week."""
    pool = d[d.elig & ~d.upcoming].copy()
    pool["tkey"] = pool.season * 100 + pool.week
    keys = [k for k in sorted(pool.tkey.unique()) if k // 100 >= first_test]
    parts, seen = [], set()
    for k in keys:
        tr, te = pool[pool.tkey < k], pool[pool.tkey == k].copy()
        base, ctxm, full = fit_stat_models(tr, False), fit_stat_models(tr, True), fit_stat_models(tr, True, True)
        om = other_means(tr)
        te = (te.join(predict_stats(full, te, "p_")).join(predict_stats(base, te, "b_"))
                .join(predict_stats(ctxm, te, "c_")))
        te["proj_stats"] = project_points(te, "p_", om).proj      # full model incl. injuries
        te["ctx_stats"] = project_points(te, "c_", om).proj       # matchup + game environment, no injuries
        te["base_stats"] = project_points(te, "b_", om).proj      # form + usage only
        te["proj_direct"] = predict(fit_models(tr, True), te, True)
        te["base_direct"] = predict(fit_models(tr, False), te, False)
        dkp = project_dk(te, "p_", dk_other_means(tr))
        te["dk_proj"], te["dk_bonus_ev"] = dkp.dk_proj, dkp.dk_bonus_ev
        parts.append(te)
        if k // 100 not in seen:
            seen.add(k // 100)
            print(f"  backtesting {k // 100} ...", flush=True)
    bt = pd.concat(parts)
    return bt.dropna(subset=["proj_stats", "ctx_stats", "base_stats", "proj_direct", "base_direct", "dk_proj"])


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def weekly_table(bt):
    rows = []
    for (s, w), g in bt.groupby(["season", "week"]):
        r = {"season": int(s), "week": int(w), "n": len(g), "rmse": _rmse(g.pts, g.proj_stats),
             "bias": float((g.proj_stats - g.pts).mean())}
        for k, col in (("final", "proj_stats"), ("ctx", "ctx_stats"), ("base", "base_stats"),
                       ("direct", "proj_direct")):
            r[f"rho_{k}"] = g[col].corr(g.pts, method="spearman") if len(g) >= 30 else np.nan
        for pos in POS:
            gp = g[g.position == pos]
            r[f"rho_{pos}"] = gp.proj_stats.corr(gp.pts, method="spearman") if len(gp) >= 10 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def injury_backtest_sections(bt):
    b = bt.copy()
    print("\n--- 7. Hurt but playing: did the discount help? (players who DID play; bias = projection - actual) ---")
    b["group"] = np.select(
        [b.inj_dbt == 1, (b.inj_q == 1) & (b.pr_dnp == 1), (b.inj_q == 1) & (b.pr_lim == 1), b.inj_q == 1,
         b.pr_dnp == 1, b.pr_lim == 1],
        ["Doubtful", "Questionable, DNP in practice", "Questionable, limited practice",
         "Questionable, full practice", "no tag, DNP in practice", "no tag, limited practice"],
        default="no injury flags")
    rows = []
    for gname, g in b.groupby("group"):
        rows.append({"group": gname, "n": len(g), "avg_actual": g.pts.mean(), "avg_proj_before": g.ctx_stats.mean(),
                     "avg_proj_after": g.proj_stats.mean(), "bias_before": (g.ctx_stats - g.pts).mean(),
                     "bias_after": (g.proj_stats - g.pts).mean(), "rmse_before": _rmse(g.pts, g.ctx_stats),
                     "rmse_after": _rmse(g.pts, g.proj_stats)})
    t = pd.DataFrame(rows).set_index("group")
    print(t.round(2).to_string())
    print("(before = matchup+environment model without injury info; after = with it)")

    print("\n--- 8. Teammates absorbing work (pass catchers: actual targets minus projected targets) ---")
    rc = b[b.position.isin(["RB", "WR", "TE"])].copy()
    rc["miss_tgt"] = rc.m_tgt_WR + rc.m_tgt_TE + rc.m_tgt_RB
    rc["bin"] = pd.cut(rc.miss_tgt, [-0.01, 0.5, 3, 6, 100], labels=["0 (nobody missing)", "0.5-3", "3-6", "6+"])
    rows = []
    for k, g in rc.groupby("bin", observed=True):
        rows.append({"expected targets missing from team": k, "n": len(g), "avg_targets": g.targets.mean(),
                     "bias_before": (g.c_targets - g.targets).mean(), "bias_after": (g.p_targets - g.targets).mean(),
                     "rmse_before": _rmse(g.targets, g.c_targets), "rmse_after": _rmse(g.targets, g.p_targets)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- 9. RBs when a teammate RB is missing (actual carries minus projected carries) ---")
    rb = b[b.position == "RB"].copy()
    rb["bin"] = pd.cut(rb.m_car_RB, [-0.01, 0.5, 4, 8, 100], labels=["0", "0.5-4", "4-8", "8+"])
    rows = []
    for k, g in rb.groupby("bin", observed=True):
        rows.append({"expected RB carries missing": k, "n": len(g), "avg_carries": g.carries.mean(),
                     "bias_before": (g.c_carries - g.carries).mean(), "bias_after": (g.p_carries - g.carries).mean(),
                     "rmse_before": _rmse(g.carries, g.c_carries), "rmse_after": _rmse(g.carries, g.p_carries)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- 10. Starting QB missing (backup playing): pass catchers' receiving yards, actual vs projected ---")
    qbo = rc[rc.m_att_QB >= 10]
    rest = rc[rc.m_att_QB < 10]
    rows = []
    for name, g in (("QB1 out/doubtful", qbo), ("QB healthy", rest)):
        rows.append({"situation": name, "n": len(g), "avg_rec_yds": g.receiving_yards.mean(),
                     "bias_before": (g.c_receiving_yards - g.receiving_yards).mean(),
                     "bias_after": (g.p_receiving_yards - g.receiving_yards).mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def dk_backtest_section(bt):
    print("\n--- 11. DraftKings-scoring projection (dk_proj) vs actual DK points ---")
    rows = []
    for pos in POS + ["ALL"]:
        b = bt if pos == "ALL" else bt[bt.position == pos]
        rows.append({"pos": pos, "n": len(b), "avg_actual_dk": b.dk_pts.mean(), "avg_dk_proj": b.dk_proj.mean(),
                     "bias": (b.dk_proj - b.dk_pts).mean(), "rmse": _rmse(b.dk_pts, b.dk_proj),
                     "r2": _r2(b.dk_pts, b.dk_proj), "avg_bonus_actual": b.dk_bonus.mean(),
                     "avg_bonus_expected": b.dk_bonus_ev.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    c = bt.assign(decile=pd.qcut(bt.dk_proj, 10, labels=False, duplicates="drop") + 1)
    cal = c.groupby("decile").agg(n=("dk_pts", "size"), avg_dk_proj=("dk_proj", "mean"), avg_actual_dk=("dk_pts", "mean"))
    cal["diff"] = cal.avg_dk_proj - cal.avg_actual_dk
    print("\nCalibration by decile (projected DK points vs actual):")
    print(cal.round(2).to_string())


def report_weekly_backtest(bt, season):
    tag = f"{int(bt.season.min())}-{int(bt.season.max())}"
    wk_all = weekly_table(bt)
    wk = wk_all[wk_all.n >= 30]            # skip partial weeks (e.g. only a Thursday game played so far)
    print(f"\n=== WEEK-BY-WEEK BACKTEST ({tag}: {len(wk)} complete weeks, {len(bt):,} player-games) ===")
    print("Each week is predicted using ONLY earlier weeks (models refit weekly).")
    print("final = stat-line projection incl. injuries | ctx = same without injury info | "
          "base = form+usage only | direct = one regression on points\n")

    print("--- 1. Fantasy-points accuracy ---")
    rows = []
    for pos in POS + ["ALL"]:
        b = bt if pos == "ALL" else bt[bt.position == pos]
        r = {"pos": pos, "n": len(b), "avg_actual": b.pts.mean(), "bias_final": (b.proj_stats - b.pts).mean()}
        for k, col in (("final", "proj_stats"), ("ctx", "ctx_stats"), ("base", "base_stats"), ("direct", "proj_direct")):
            r[f"rmse_{k}"], r[f"r2_{k}"] = _rmse(b.pts, b[col]), _r2(b.pts, b[col])
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- 2. Accuracy of each projected stat (final model vs without injuries vs form+usage only) ---")
    rows = []
    for pos in POS:
        b = bt[bt.position == pos]
        for s in MODEL_STATS[pos]:
            y, p, c, q = b[s], b["p_" + s], b["c_" + s], b["b_" + s]
            rows.append({"pos": pos, "stat": SHORT[s], "avg_actual": y.mean(), "avg_pred": p.mean(),
                         "rmse_base": _rmse(y, q), "rmse_ctx": _rmse(y, c), "rmse_final": _rmse(y, p),
                         "r2_base": _r2(y, q), "r2_ctx": _r2(y, c), "r2_final": _r2(y, p)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n--- 3. Ranking skill by season: average weekly Spearman rank correlation with actual points ---")
    by = wk.groupby("season").agg(weeks=("week", "size"), rmse=("rmse", "mean"), bias=("bias", "mean"),
                                  rho_final=("rho_final", "mean"), rho_ctx=("rho_ctx", "mean"),
                                  rho_base=("rho_base", "mean"), rho_direct=("rho_direct", "mean"),
                                  rho_QB=("rho_QB", "mean"), rho_RB=("rho_RB", "mean"),
                                  rho_WR=("rho_WR", "mean"), rho_TE=("rho_TE", "mean"))
    print(by.round(3).to_string())

    print(f"\n--- 4. Week by week, {season} (a week with n < 30 is only partly played, so it has no rank score) ---")
    cur = wk_all[wk_all.season == season].drop(columns="season")
    print(cur.round(3).to_string(index=False))

    print("\n--- 5. Start/sit value: actual points of the top-K players each model would start (per week) ---")
    out = []
    for (s, w, pos), g in bt.groupby(["season", "week", "position"]):
        k = TOPK[pos]
        if len(g) >= 2 * k:
            out.append((pos, g.nlargest(k, "proj_stats").pts.mean(), g.nlargest(k, "ctx_stats").pts.mean(),
                        g.nlargest(k, "base_stats").pts.mean(), g.nlargest(k, "pts").pts.mean()))
    o = pd.DataFrame(out, columns=["pos", "final", "ctx", "base", "hindsight_best"])
    o["gain_inj"] = o.final - o.ctx
    o["gain_vs_base"] = o.final - o.base
    t = o.groupby("pos").agg(weeks=("gain_inj", "size"), final=("final", "mean"), ctx=("ctx", "mean"),
                             base=("base", "mean"), gain_inj=("gain_inj", "mean"),
                             se_inj=("gain_inj", lambda x: x.std() / np.sqrt(len(x))),
                             gain_vs_base=("gain_vs_base", "mean"), hindsight_best=("hindsight_best", "mean"))
    t["t_inj"] = t.gain_inj / t.se_inj
    t["K"] = t.index.map(TOPK)
    print(t.round(3).to_string())

    print("\n--- 6. Calibration: projected vs actual points by projection decile (all positions) ---")
    c = bt.assign(decile=pd.qcut(bt.proj_stats, 10, labels=False, duplicates="drop") + 1)
    cal = c.groupby("decile").agg(n=("pts", "size"), avg_proj=("proj_stats", "mean"), avg_actual=("pts", "mean"))
    cal["diff"] = cal.avg_proj - cal.avg_actual
    print(cal.round(2).to_string())

    injury_backtest_sections(bt)
    dk_backtest_section(bt)
    return wk_all


def save_backtest(bt, wk, season):
    keep = (["season", "week", "player_display_name", "position", "team", "opp", "pts", "proj_stats", "ctx_stats",
             "base_stats", "proj_direct", "dk_pts", "dk_proj"] + INJ_OWN + MISS_ALL + STAT_COLS
            + ["p_" + s for s in STAT_COLS])
    keep = [c for c in keep if c in bt.columns]
    bt[keep].rename(columns={"player_display_name": "player", "pts": "actual_pts", "dk_pts": "actual_dk_pts",
                             "proj_stats": "proj_final", "ctx_stats": "proj_no_injury",
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
    skip = set(INJ_OWN + MISS_ALL)
    rows = []
    for s in MODEL_STATS["TE"]:
        res, feats, _ = full[("TE", s)]
        for v in feats:
            if v not in skip:
                rows.append({"stat": SHORT[s], "feature": v, "coef": res.params[v], "p": res.pvalues[v]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def report_injury_coefficients(full):
    print("\n=== WHAT INJURIES DO (fit on all history): effect on projected stat, coefficient (p-value) ===")
    print("inj_*/pr_* = the player's own tag/practice (he plays); m_* = expected usage missing from his team "
          "(per 1 missing target / carry / pass attempt)\n")
    combos = [("WR", "targets"), ("TE", "targets"), ("RB", "targets"), ("RB", "carries"),
              ("WR", "receiving_yards"), ("TE", "receiving_yards"), ("QB", "passing_yards")]
    feats = INJ_OWN + MISS_ALL
    tab = {}
    for pos, s in combos:
        res, fl, _ = full[(pos, s)]
        col = f"{pos} {SHORT[s]}"
        tab[col] = {f: (f"{res.params[f]:+.3f} ({res.pvalues[f]:.2f})" if f in fl else "") for f in feats}
    print(pd.DataFrame(tab).to_string())


def report_injury_week(out, panel, season, week, up_teams):
    print("\n=== INJURY ADJUSTMENTS THIS WEEK ===")
    if panel is not None:
        cur = panel[(panel.season == season) & (panel.week == week) & panel.team.isin(up_teams)
                    & (panel.report_status == "Out")].copy()
        cur["usage"] = cur.ea_targets.fillna(0) + cur.ea_carries.fillna(0) + 0.3 * cur.ea_attempts.fillna(0)
        cur = cur[(cur.ea_targets.fillna(0) >= 3) | (cur.ea_carries.fillna(0) >= 6) | (cur.ea_attempts.fillna(0) >= 15)]
        if len(cur):
            cur = cur.sort_values(["team", "usage"], ascending=[True, False])
            cur["new_absence"] = cur.presence.round(2)
            t = cur[["team", "full_name", "position", "ea_targets", "ea_carries", "ea_attempts", "new_absence"]].rename(
                columns={"full_name": "player", "position": "pos", "ea_targets": "tgt/g", "ea_carries": "car/g",
                         "ea_attempts": "att/g", "new_absence": "in_lineup_recently"})
            print("Regulars listed OUT (their work is redistributed; 'in_lineup_recently' near 0 means the team "
                  "has already adjusted to life without him, so little changes):")
            print(t.round(1).to_string(index=False))
    show = ["player", "pos", "team", "injury", "p_play", "proj", "inj_own", "inj_team", "proj_ev"]
    t = out[out.inj_team > 0].sort_values("inj_team", ascending=False).head(12)
    print("\nBiggest bumps from teammates being out (inj_team = points added):")
    print(t[show].to_string())
    t = out[out.inj_team < -0.05].sort_values("inj_team").head(6)
    if len(t):
        print("\nBiggest drags from teammates being out (e.g. a backup QB throwing to him):")
        print(t[show].to_string())
    t = out[out.inj_own < -0.05].sort_values("inj_own").head(12)
    print("\nHurt but expected to play (inj_own = points removed; p_play = chance he's active):")
    print(t[show].to_string())


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


# ----------------------------------------------------------------------------- projected game scores
def team_stat_sums(df, td_col, rtd_col, pyd_col, ryd_col, weight=None):
    """Per team: projected QB passing TDs, all rushing TDs, passing yards, rushing yards (optionally x p_play)."""
    w = 1.0 if weight is None else df[weight]
    x = pd.DataFrame({"team": df.team, "ptd": df[td_col].fillna(0) * w * (df.position == "QB"),
                      "rtd": df[rtd_col].fillna(0) * w, "pyd": df[pyd_col].fillna(0) * w * (df.position == "QB"),
                      "ryd": df[ryd_col].fillna(0) * w})
    return x.groupby("team", as_index=False).sum()


def team_points(t, coef=None):
    c = coef or TEAM_PTS
    return c["const"] + sum(c[k] * t[k] for k in TEAM_PTS_X)


def fit_team_points(bt, games):
    """Fit team points on the backtest's projected stats and print how it compares with Vegas (leave-one-season-out)."""
    bt = bt.copy()
    for c in ["proj_pass_td", "proj_rush_td", "proj_pass_yds", "proj_rush_yds"]:
        bt[c] = bt[c].fillna(0)
    isqb = bt.position == "QB"
    bt["ptd"], bt["pyd"] = bt.proj_pass_td * isqb, bt.proj_pass_yds * isqb
    t = bt.groupby(["season", "week", "team"]).agg(
        n=("player", "size"), nqb=("position", lambda x: (x == "QB").sum()), ptd=("ptd", "sum"),
        rtd=("proj_rush_td", "sum"), pyd=("pyd", "sum"), ryd=("proj_rush_yds", "sum")).reset_index()
    g = games.dropna(subset=["team_score"]).assign(impl=lambda x: x.total_line / 2 + x.team_spread / 2)
    t = t.merge(g[["season", "week", "team", "team_score", "impl"]], on=["season", "week", "team"])
    t = t[(t.nqb >= 1) & (t.n >= 6)].dropna(subset=["impl"]).reset_index(drop=True)
    if len(t) < 200:
        print("(not enough backtest team-games to fit the game-score model; keeping defaults)")
        return TEAM_PTS
    ours = pd.Series(index=t.index, dtype=float)
    for s_ in sorted(t.season.unique()):
        m = sm.OLS(t.team_score[t.season != s_], sm.add_constant(t.loc[t.season != s_, TEAM_PTS_X])).fit()
        ours[t.season == s_] = m.predict(sm.add_constant(t.loc[t.season == s_, TEAM_PTS_X], has_constant="add"))
    fit = sm.OLS(t.team_score, sm.add_constant(t[TEAM_PTS_X])).fit()
    coef = {"const": float(fit.params["const"]), **{k: float(fit.params[k]) for k in TEAM_PTS_X}}
    print(f"\n=== TEAM POINTS FROM OUR STAT LINES vs VEGAS ({len(t)} team-games; each season scored by a model fit on the others) ===")
    rows = []
    for nm, pr in (("our stat lines", ours), ("Vegas implied", t.impl), ("average of both", (ours + t.impl) / 2)):
        e = t.team_score - pr
        rows.append({"source": nm, "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "bias": e.mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    both = sm.OLS(t.team_score, sm.add_constant(pd.DataFrame({"vegas": t.impl, "ours": ours}))).fit()
    print(f"Actual points on both: Vegas weight {both.params['vegas']:.2f} (se {both.bse['vegas']:.2f}), "
          f"our-stat-lines weight {both.params['ours']:.2f} (se {both.bse['ours']:.2f})  <- does our build add to Vegas?")
    print("fitted: points = " + f"{coef['const']:.2f} " + " ".join(f"{coef[k]:+.4f}*{k}" for k in TEAM_PTS_X))
    return coef


def report_game_scores(up, games, season, week, coef, path):
    """Projected score for every game of the week from the players' projected stat lines."""
    t = team_stat_sums(up, "pass_td", "rush_td", "pass_yds", "rush_yds", weight="p_play")
    t["ours"] = team_points(t, coef)
    g = games[(games.season == season) & (games.week == week) & (~games.played) & (games.is_home == 1)].copy()
    g = g.merge(t[["team", "ours", "ptd", "rtd"]], on="team", how="left").merge(
        t[["team", "ours", "ptd", "rtd"]].rename(columns={"team": "opp", "ours": "ours_opp", "ptd": "ptd_opp", "rtd": "rtd_opp"}),
        on="opp", how="left")
    g["vegas_home"] = g.total_line / 2 + g.team_spread / 2
    g["vegas_away"] = g.total_line / 2 - g.team_spread / 2
    g["our_total"] = g.ours + g.ours_opp
    g["our_spread"] = g.ours - g.ours_opp                      # positive = home team ahead
    g["total_diff"] = g.our_total - g.total_line
    g["spread_diff"] = g.our_spread - g.team_spread
    out = g.rename(columns={"team": "home", "opp": "away", "ours": "home_pts", "ours_opp": "away_pts",
                            "team_spread": "vegas_spread", "total_line": "vegas_total"})[
        ["home", "away", "home_pts", "away_pts", "our_total", "vegas_total", "total_diff",
         "our_spread", "vegas_spread", "spread_diff", "vegas_home", "vegas_away"]]
    out = out.dropna(subset=["our_total"]).sort_values("total_diff", ascending=False).round(1)
    if out.empty:
        return None
    out.to_csv(path, index=False)
    print(f"\n=== PROJECTED SCORES - week {week} (built from the players' projected stat lines, p_play-weighted) ===")
    print("home_pts / away_pts = our score; our_spread > 0 means the HOME team wins by that much; "
          "total_diff / spread_diff = ours minus Vegas")
    print(out.to_string(index=False))
    print(f"(saved to {path}. Caution: the stat lines are anchored to the Vegas total, so big gaps mostly come from "
          "injuries/usage that the line may not reflect; in backtests this build matched Vegas but did not beat it.)")
    return out


def _stat_note(r):
    f = lambda c, n=1: f"{r[c]:.{n}f}"
    if r.pos == "QB":
        return f"{f('pass_yds', 0)} pass yds, {f('pass_td')} pass TD, {f('rush_yds', 0)} rush yds"
    if r.pos == "RB":
        return f"{f('car', 0)} car, {f('rush_yds', 0)} rush yds, {f('rec')} rec, {f('rec_yds', 0)} rec yds, {f('rush_td')}+{f('rec_td')} TD"
    return f"{f('tgt')} tgt, {f('rec')} rec, {f('rec_yds', 0)} rec yds, {f('rec_td')} TD"


def _hit_rate(a):
    return next(h for lo, h in ADJ_HIT if a >= lo)


def load_props(path):
    p = pd.read_csv(path)
    p.columns = [c.strip().lower() for c in p.columns]
    need = {"player", "stat", "line"} - set(p.columns)
    if need:
        raise ValueError(f"props file needs columns player, stat, line (missing {sorted(need)})")
    p["stat"] = p.stat.str.strip().str.lower()
    return p


def report_opportunities(out, games_out, season, week, props=None, top_n=5):
    """Top selections per game: our game total / spread vs Vegas, and players whose projection moved most
    away from what their recent form alone says (matchup, injuries, teammates' absences, game script)."""
    if games_out is None or games_out.empty:
        return
    rows = []
    for g in games_out.itertuples():
        game = f"{g.away} @ {g.home}"
        if abs(g.total_diff) > 0:
            side = "OVER" if g.total_diff > 0 else "UNDER"
            rows.append(dict(game=game, type="TOTAL", selection=f"{side} {g.vegas_total:g}", ours=g.our_total,
                             baseline=g.vegas_total, edge=g.total_diff, z=abs(g.total_diff) / TOTAL_DIFF_SD,
                             confidence="low: no proven edge vs Vegas in backtest",
                             note=f"our score {g.away} {g.away_pts:.1f}, {g.home} {g.home_pts:.1f}"))
        if abs(g.spread_diff) > 0:
            fav = g.home if g.spread_diff > 0 else g.away       # we like the home team more when our spread is higher
            line = g.vegas_spread if fav == g.home else -g.vegas_spread
            rows.append(dict(game=game, type="SPREAD", selection=f"{fav} {-line:+g}", ours=g.our_spread,
                             baseline=g.vegas_spread, edge=g.spread_diff, z=abs(g.spread_diff) / MARGIN_DIFF_SD,
                             confidence="low: no proven edge vs Vegas in backtest",
                             note=f"our margin {g.home} by {g.our_spread:+.1f} vs Vegas {g.vegas_spread:+.1f}"))
        for team in (g.home, g.away):
            pl = out[(out.team == team) & (out.p_play >= 0.6)]
            for r in pl.itertuples(index=False):
                r = pd.Series(r._asdict())
                adj = r.proj - r.form_only
                if not np.isfinite(adj) or abs(adj) < 0.5:
                    continue
                rows.append(dict(game=game, type="PLAYER", selection=f"{r.player} ({r.pos}, {team}) "
                                 f"{'UP' if adj > 0 else 'DOWN'}", ours=r.proj, baseline=r.form_only, edge=adj,
                                 z=abs(adj) / ADJ_SD, confidence=f"backtest: moved our way {_hit_rate(abs(adj)):.0%} of the time",
                                 note=f"proj {_stat_note(r)} | p_play {r.p_play:.2f}"
                                      + (f" | {r.injury}" if isinstance(r.injury, str) and r.injury else "")))
    if not rows:
        return
    o = pd.DataFrame(rows).sort_values(["game", "z"], ascending=[True, False])
    o["rank"] = o.groupby("game").cumcount() + 1
    o = o[o["rank"] <= top_n]
    order = games_out.assign(g=games_out.away + " @ " + games_out.home).set_index("g").total_diff.abs()
    o["_k"] = o.game.map(order)
    o = o.sort_values(["_k", "game", "rank"], ascending=[False, True, True]).drop(columns="_k")
    o = o[["game", "rank", "type", "selection", "ours", "baseline", "edge", "z", "confidence", "note"]].round(2)
    path = f"opportunities_{season}_wk{week:02d}.csv"
    o.to_csv(path, index=False)
    print(f"\n=== BEST OPPORTUNITIES BY GAME - week {week} (top {top_n} per game; z = size of the gap in standard deviations) ===")
    print("PLAYER UP/DOWN = our projection vs what his recent form alone implies (edge in fantasy points). "
          "TOTAL/SPREAD = our game score vs Vegas.")
    for game, g in o.groupby("game", sort=False):
        print(f"\n{game}")
        print(g[["rank", "type", "selection", "ours", "baseline", "edge", "z"]].to_string(index=False, header=True))
    print(f"\n(saved to {path}. In the backtest, players with |adjustment| >= 2 pts moved the way we said about 65% of the time; "
          "our game totals/spreads did not reliably beat Vegas, so treat those two as flags only.)")

    if props is not None:
        pr = []
        key = {(name_key(r.player), r.team): r for r in out.itertuples(index=False)}
        by_name = {}
        for r in out.itertuples(index=False):
            by_name.setdefault(name_key(r.player), []).append(r)
        gm = {}
        for g in games_out.itertuples():
            gm[g.home] = gm[g.away] = f"{g.away} @ {g.home}"
        miss = 0
        for q in props.itertuples():
            c = by_name.get(name_key(q.player), [])
            if hasattr(q, "team") and isinstance(q.team, str):
                c = [x for x in c if x.team == q.team.upper()] or c
            if len(c) != 1 or q.stat not in SHORT.values() or not hasattr(c[0], q.stat):
                miss += 1
                continue
            r = c[0]
            ours, sd = getattr(r, q.stat), PROP_SD.get((r.pos, q.stat))
            if sd is None or not np.isfinite(ours):
                miss += 1
                continue
            pr.append(dict(game=gm.get(r.team, ""), player=f"{r.player} ({r.pos}, {r.team})", stat=q.stat,
                           line=q.line, ours=ours, edge=ours - q.line, pick="OVER" if ours > q.line else "UNDER",
                           z=abs(ours - q.line) / sd, p_play=r.p_play))
        if pr:
            pdf = pd.DataFrame(pr).sort_values(["game", "z"], ascending=[True, False])
            pdf["rank"] = pdf.groupby("game").cumcount() + 1
            pdf = pdf[pdf["rank"] <= top_n].round(2)
            pp = f"prop_edges_{season}_wk{week:02d}.csv"
            pdf.to_csv(pp, index=False)
            print(f"\n=== PROP LINES vs OUR STAT LINES (top {top_n} per game; matched {len(pr)} of {len(props)} lines) ===")
            print(pdf[["game", "rank", "player", "stat", "line", "ours", "edge", "pick", "z", "p_play"]].to_string(index=False))
            print(f"(saved to {pp}. z = edge / typical projection miss for that stat; most edges are well under 0.5, "
                  "and nothing here has been tested against historical lines.)")
        elif miss:
            print("(no prop lines could be matched; the file needs player, stat, line with stat like rec, rec_yds, pass_yds)")


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
                    help="run the week-by-week walk-forward backtest (about 3 minutes)")
    ap.add_argument("--first-test", type=int, default=2022, help="first season to backtest")
    ap.add_argument("--no-backtest", action="store_true", help=argparse.SUPPRESS)   # old flag, now the default
    ap.add_argument("--sleeper", action="store_true", help="add Sleeper's projections as a column")
    ap.add_argument("--compare-csv", default=None, help="CSV (player, proj[, pos, team]) from ESPN etc.")
    ap.add_argument("--compare-name", default="other", help="label for --compare-csv, e.g. ESPN")
    ap.add_argument("--dk-salaries", default=None,
                    help="DraftKings salary CSV (the 'Export to CSV' / DKSalaries.csv file for your slate)")
    ap.add_argument("--props-csv", default=None,
                    help="optional player prop lines (columns: player, stat, line[, team]; stat = rec, rec_yds, "
                         "pass_yds, rush_yds, ...) to compare with our stat lines")
    ap.add_argument("--no-def-injuries", action="store_true",
                    help="ignore defender absences (e.g. a missing cornerback)")
    ap.add_argument("--no-roster-filter", action="store_true",
                    help="skip the roster-status / missed-games availability filter")
    ap.add_argument("--drop-missed", action="store_true",
                    help="drop (instead of discounting) regulars who missed their team's last 2+ games unreported")
    args = ap.parse_args()

    CFG.update(te_premium=args.te_premium, pass_td=args.pass_td, def_inj=not args.no_def_injuries)
    season = args.season or nfl.get_current_season()
    stats, games = load_data(season, args.te_premium, args.pass_td)

    cur = games[games.season == season]
    week = args.week or int(cur[~cur.played].week.min())
    print(f"Season {season}, projecting week {week}  |  TE premium: +{args.te_premium}/rec "
          f"(TE = {1 + args.te_premium:g} PPR), passing TD = {args.pass_td:g}")

    rosters = None if args.no_roster_filter else load_rosters(season)
    d = build_frame(stats, games, season, week, args.max_gap_days, rosters)
    d = add_features(d, games, season, week)
    inj = load_injuries_all(range(FIRST_SEASON, season + 1))
    if inj is None:
        print("(injury reports unavailable - running WITHOUT injury adjustments; check inactives yourself)")
    elif not (inj.season == season).any() or not ((inj.season == season) & (inj.week == week)).any():
        print(f"(no injury report posted yet for {season} week {week} - injury adjustments will be minimal)")
    d, panel = add_injury_features(d, inj, stats, games)
    d, def_players = attach_defense_missing(d, season)
    if panel is not None:
        report_play_rates(panel)

    if args.backtest:
        bt = backtest_weekly(d, args.first_test)
        wk = report_weekly_backtest(bt, season)
        save_backtest(bt, wk, season)
        TEAM_PTS.update(fit_team_points(pd.read_csv("backtest_weekly_players.csv"), games))

    pool = d[d.elig & ~d.upcoming]
    base = fit_stat_models(pool, False)
    full = fit_stat_models(pool, True, True)
    om = other_means(pool)
    dk_om = dk_other_means(pool)
    direct = fit_models(pool, True, robust=True)
    report_coefficients(direct)
    report_te_stat_coefficients(full)
    report_injury_coefficients(full)

    up = d[d.upcoming & d.elig].copy()
    n_out = int((up.inj_status == "Out").sum())
    up = up[up.inj_status != "Out"]                       # players listed Out are not projected
    print(f"\nInjury report: {n_out} projectable players listed Out for week {week} (dropped; "
          f"their teammates' work is adjusted below).")
    if not args.no_roster_filter:
        up, _ = roster_filter(up, rosters, stats, games, inj, season, week, args.drop_missed)
    else:
        up["roster_note"] = ""
    up = choose_starting_qbs(up)
    up = up.join(predict_stats(full, up, "p_"))
    up = up.join(predict_stats(base, up, "b_"))
    pp = project_points(up, "p_", om)
    up["proj"] = pp.proj
    up["pts_rec"], up["pts_yds"], up["pts_td"] = pp.pts_rec, pp.pts_yds, pp.pts_td
    up["pts_other"] = pp.pts_int + pp.pts_other           # interceptions + fumbles/2pt average
    up["form_only"] = project_points(up, "b_", om).proj
    # what injuries contribute: re-project with own-injury flags / teammates' missing work switched off
    up["inj_own"] = up.proj - proj_with(full, up, om, INJ_OWN)
    up["inj_team"] = up.proj - proj_with(full, up, om, MISS_ALL)
    up["matchup_boost"] = proj_with(full, up, om, INJ_OWN + MISS_ALL) - up.form_only
    up["proj_ev"] = up.p_play * up.proj                   # expected points incl. the chance he sits
    dkp = project_dk(up, "p_", dk_om)                     # same stat line, scored the DraftKings way
    up["dk_proj"], up["dk_bonus_ev"] = dkp.dk_proj, dkp.dk_bonus_ev
    up["dk_ev"] = up.p_play * up.dk_proj
    up["proj_direct"] = predict(direct, up, True)         # one-regression-on-points model, for reference
    up = up.dropna(subset=["proj"])
    up["injury"] = (up.inj_status.fillna("") + " " + up.inj_practice.map(
        {"full": "full practice", "limited": "limited practice", "dnp": "DNP"}).fillna("")).str.strip()
    up["injury"] = (up.injury + "; " + up.roster_note).str.strip("; ")

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

    have_salary, dk_lines, dk_matched, dk_pool = False, {}, set(), None
    if args.dk_salaries:
        try:
            dk_pool = load_dk_salaries(args.dk_salaries)
            up, dk_matched = attach_salary(up, dk_pool)
            up, dk_lines = add_dk_value(up)
            have_salary = True
            print(f"DraftKings salaries: {len(dk_pool)} players in the file, "
                  f"{int(up.salary.notna().sum())} of my {len(up)} projections matched to a salary.")
        except Exception as e:
            print(f"(could not use the DraftKings salary file: {e})")
    dk_cols = ["dk_proj", "dk_ev"] + (["salary", "pts_per_k", "value_pts", "dk_avg"] if have_salary else [])

    up["side"] = np.where(up.is_home == 1, "vs", "@")
    up["opp_str"] = up.side + " " + up.opp
    up["line"] = up.team_spread.map(lambda x: f"{x:+.1f}")
    up = up.rename(columns={"p_" + s: SHORT[s] for s in STAT_COLS})
    stat_cols = [SHORT[s] for s in STAT_COLS]
    cols = (["player_display_name", "position", "team", "opp_str", "line", "total_line", "def_rel",
             "pts_ewma", "form_only", "proj", "p_play", "proj_ev"] + dk_cols
            + ["inj_own", "inj_team", "matchup_boost"]
            + stat_cols + ["pts_rec", "pts_yds", "pts_td", "pts_other", "proj_direct"] + extra + ["injury"])
    out = up.sort_values("proj_ev", ascending=False)[cols].rename(columns={
        "player_display_name": "player", "position": "pos", "opp_str": "opp",
        "total_line": "total", "pts_ewma": "recent_avg"})
    out = out.round(2).reset_index(drop=True)
    out.index += 1

    path = f"projections_{season}_wk{week:02d}.csv"
    out.to_csv(path, index_label="rank")

    head = (["player", "pos", "team", "opp", "line", "total", "recent_avg", "proj", "p_play", "proj_ev",
             "inj_own", "inj_team", "matchup_boost"] + extra + ["injury"])
    print(f"\n=== TOP 30 OVERALL - week {week} (TE-premium PPR; sorted by proj_ev = p_play x proj) ===")
    print(out.head(30)[head].to_string())
    report_injury_week(out, panel, season, week, set(up.team))
    if CFG.get("def_inj"):
        report_defense_missing(def_players, up, season, week)

    recv = ["player", "team", "opp", "line", "total", "proj", "proj_ev", "tgt", "rec", "rec_yds", "rec_td",
            "car", "rush_yds", "rush_td", "pts_rec", "pts_yds", "pts_td", "pts_other"] + extra + ["injury"]
    qbc = ["player", "team", "opp", "line", "total", "proj", "proj_ev", "pass_att", "pass_yds", "pass_td", "int",
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
    gout = report_game_scores(up, games, season, week, TEAM_PTS, f"game_projections_{season}_wk{week:02d}.csv")
    props = None
    if args.props_csv:
        try:
            props = load_props(args.props_csv)
        except Exception as e:
            print(f"(could not read the props file: {e})")
    report_opportunities(out, gout, season, week, props)
    print("\n=== BIGGEST MATCHUP BOOSTS / DRAGS (vs. the same player's form-only projection) ===")
    mb = ["player", "pos", "team", "opp", "line", "total", "recent_avg", "proj", "matchup_boost", "injury"]
    print(out.sort_values("matchup_boost", ascending=False).head(10)[mb].to_string())
    print(out.sort_values("matchup_boost").head(10)[mb].to_string())
    for label, _ in sources:
        report_comparison(out, label)
    if have_salary:
        report_dk(out, dk_lines, dk_pool, dk_matched, panel, season, week, f"dk_value_{season}_wk{week:02d}.csv")
    else:
        print("\n(DraftKings: dk_proj / dk_ev are in the CSV. Add --dk-salaries DKSalaries.csv for points per $1,000.)")
    print(f"\nSaved {len(out)} projections to {path}  (keep each week's file: it's your snapshot "
          "for scoring us vs. them after the games)")


if __name__ == "__main__":
    main()