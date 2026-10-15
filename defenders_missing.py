"""
Defender absences: how much defensive playing time is missing from each team's defense in a given week.

For every defender on an injury report we estimate how much of his team's defense is missing this week:

    missing = (1 - P(plays | report tag, practice)) x his recent share of defensive snaps

summed by group: CB, S (safeties), LB (incl. edge rushers listed as LB) and DL (DE / DT / NT).
A defender who has already been out for weeks has a recent snap share near 0, so his absence counts for little:
it is already in how the defense has been playing. A NEW absence of a full-time starter counts about 1.0.

Free data only: nflverse snap counts + injury reports. Players are matched by first initial + last name + team + week.
"""
import re
import time

import numpy as np
import pandas as pd
import nflreadpy as nfl

TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
GROUP = {"CB": "cb", "DB": "cb", "S": "s", "FS": "s", "SS": "s",
         "LB": "lb", "OLB": "lb", "ILB": "lb", "MLB": "lb", "DE": "dl", "DT": "dl", "DL": "dl", "NT": "dl"}
GROUPS = ["cb", "s", "lb", "dl"]
HALFLIFE = 4
TRAIN_END = 2021          # play-probability table learned from seasons <= this


def _retry(f, tries=4):
    for i in range(tries):
        try:
            return f()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3)


def name_key(s):
    """First initial + last name ('Pat Surtain II' and 'Patrick Surtain II' both -> 'psurtain'): the two data
    sources disagree on nicknames, and same-team collisions on initial + surname are very rare."""
    s = str(s).lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    parts = [p for p in re.sub(r"[^a-z ]", "", s).split() if p]
    if not parts:
        return ""
    return parts[0][0] + parts[-1] if len(parts) > 1 else parts[0]


def load_defense_data(seasons, snaps=None, inj=None):
    sn = snaps if snaps is not None else _retry(lambda: nfl.load_snap_counts(seasons=list(seasons)).to_pandas())
    ij = inj if inj is not None else _retry(lambda: nfl.load_injuries(seasons=list(seasons)).to_pandas())
    return sn, ij


def build_missing(sn, ij, first_season=2018):
    """Returns (table of missing defensive snap share per defense-week, defender table with p_play)."""
    sn = sn[(sn.game_type == "REG") & sn.position.isin(GROUP)].copy()
    sn["team"] = sn.team.replace(TEAM_FIX)
    sn["nk"] = sn.player.map(name_key)
    sn["pct"] = sn.defense_pct.fillna(0.0)
    sn = sn.drop_duplicates(["season", "week", "team", "nk"])
    sn["tkey"] = sn.season * 100 + sn.week

    ij = ij[(ij.game_type == "REG") & ij.position.isin(GROUP)].dropna(subset=["season", "week"]).copy()
    ij["season"], ij["week"] = ij.season.astype(int), ij.week.astype(int)
    ij["team"] = ij.team.replace(TEAM_FIX)
    ij["nk"] = ij.full_name.map(name_key)
    ij["group"] = ij.position.map(GROUP)
    ij["status"] = ij.report_status.where(ij.report_status.isin(["Out", "Doubtful", "Questionable"]), "none")
    ps = ij.practice_status.fillna("")
    ij["practice"] = np.select([ps.str.startswith("Full"), ps.str.startswith("Limited"), ps.str.startswith("Did Not")],
                               ["full", "limited", "dnp"], default="")
    ij = ij.drop_duplicates(["season", "week", "team", "nk"], keep="last").reset_index(drop=True)
    ij["rid"] = np.arange(len(ij))

    # the team's game sequence (across seasons) from the snap data
    tg = sn[["team", "season", "week", "tkey"]].drop_duplicates().sort_values(["team", "tkey"]).reset_index(drop=True)
    tg["gidx"] = tg.groupby("team").cumcount()
    # game number of the target week = number of team games before it
    ij["tkey"] = ij.season * 100 + ij.week
    gi = []
    for team, g in ij.groupby("team"):
        keys = tg.loc[tg.team == team, "tkey"].to_numpy()
        gi.append(pd.Series(np.searchsorted(keys, g.tkey.to_numpy(), side="left"), index=g.index))
    ij["tgidx"] = pd.concat(gi)

    # as-of snap share: recency-weighted share of defensive snaps over his team's previous 4 games (0 if absent)
    num = np.zeros(len(ij))
    den = np.zeros(len(ij))
    for lag in range(1, 5):
        prev = tg.assign(gidx=tg.gidx + lag)[["team", "gidx", "season", "week"]].rename(
            columns={"season": "ps", "week": "pw"})
        j = ij[["rid", "team", "tgidx", "nk"]].merge(prev, left_on=["team", "tgidx"], right_on=["team", "gidx"], how="left")
        j = j.merge(sn[["season", "week", "team", "nk", "pct"]].rename(columns={"season": "ps", "week": "pw"}),
                    on=["ps", "pw", "team", "nk"], how="left")
        j = j.set_index("rid").reindex(ij.rid)
        w = 0.5 ** ((lag - 1) / HALFLIFE)
        ex = j.ps.notna().to_numpy()
        num += np.where(ex, w * j.pct.fillna(0.0).to_numpy(), 0.0)
        den += np.where(ex, w, 0.0)
    ij["share"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)

    # did he play that week? (appears in the snap data with snaps)
    ij = ij.merge(sn[["season", "week", "team", "nk", "defense_snaps"]], on=["season", "week", "team", "nk"], how="left")
    ij["played"] = (ij.defense_snaps.fillna(0) > 0).astype(float)

    # P(plays | tag, practice) among regular defenders (as-of share >= 0.3), seasons <= TRAIN_END
    t = ij[(ij.season <= TRAIN_END) & (ij["share"] >= 0.3)]
    fine = t.groupby(["status", "practice"]).played.agg(["mean", "size"])
    coarse = t.groupby("status").played.mean()
    fd = {k: v["mean"] for k, v in fine.iterrows() if v["size"] >= 30}
    ij["p_play"] = [fd.get(k, coarse.get(k[0], 1.0)) for k in zip(ij.status, ij.practice)]
    ij["miss"] = (1.0 - ij.p_play) * ij["share"].fillna(0.0)

    m = ij.pivot_table(index=["season", "week", "team"], columns="group", values="miss", aggfunc="sum", fill_value=0.0)
    for g in GROUPS:
        if g not in m:
            m[g] = 0.0
    m = m[GROUPS].add_prefix("dm_").reset_index()
    return m, ij, fine


if __name__ == "__main__":
    sn, ij = load_defense_data(range(2018, 2027))
    m, d, fine = build_missing(sn, ij)
    print(fine.round(2).to_string())
    print(m.describe().round(3).T)