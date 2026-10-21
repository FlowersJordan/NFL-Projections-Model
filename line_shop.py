#!/usr/bin/env python3
"""
Line shop: compare DraftKings and FanDuel on the same player props and find where the difference is worth money.

Usage (repo root, after fetch_props.py has written props_books.csv):
    python line_shop.py                                       # book vs book only
    python line_shop.py --edges prop_edges_2026_wk07.csv      # also pick the best book for each of OUR model picks
    python line_shop.py --min-ev 3 --min-gain 4

Needs: pip install pandas numpy

What it looks for (line_shop.csv, one row per opportunity, 'kind' says which):
  arb     Over at one book + Under at the other, same line, and the two prices together cost less than 100%.
          Guaranteed profit in theory. Free-plan lines are ~60 seconds old, so verify on both apps before betting.
  middle  The books' main lines differ (say FanDuel Over 83.5, DraftKings Under 85.5). Bet both: if the player lands in
          the gap, BOTH win; otherwise you usually lose a little on one leg. value_pct is the expected return on the two
          stakes using each book's own no-vig odds, win_both_pct the chance of landing in the gap.
  ev      Same main line at both books: one book's price is better than the other book's no-vig fair price.
  shop    Same line and side at both books but different prices: take the better one (value_pct = extra payout).
shop_best_bets.csv (needs --edges): for each model pick, which book/line gives the best expected value if our projection
is right (normal approximation around our number, using the spread our edge file implies). model_p and ev_pct are only
as good as the projection; the book comparison (which book is better, and by how much) is the dependable part.
"""
import argparse
import math
import re
import sys

import numpy as np
import pandas as pd

BOOKS = ("draftkings", "fanduel")


def norm(s):
    s = str(s).lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def imp(o):
    if pd.isna(o) or o == 0:
        return np.nan
    return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)


def dec(o):
    if pd.isna(o) or o == 0:
        return np.nan
    return 1 + (o / 100 if o > 0 else 100 / abs(o))


def nov(po, pu):
    """No-vig probability of the Over from both sides' implied probabilities."""
    return po / (po + pu)


def cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def row(kind, g, **kw):
    base = dict(kind=kind, game=g[0], player=g[1], stat=g[2])
    base.update(kw)
    return base


def book_vs_book(bt, min_ev, min_gain, min_middle_ev=0.0):
    out = []
    diffs = []
    for g, x in bt.groupby(["game", "player", "stat"]):
        by = {b: x[x.book == b] for b in BOOKS}
        if any(len(v) == 0 for v in by.values()):
            continue
        dk, fd = by["draftkings"], by["fanduel"]
        # ---------------- shop + arb on every line both books quote
        for ln in sorted(set(dk.line) & set(fd.line)):
            a, b = dk[dk.line == ln].iloc[0], fd[fd.line == ln].iloc[0]
            for side, col in (("OVER", "over_odds"), ("UNDER", "under_odds")):
                oa, ob = a[col], b[col]
                if pd.isna(oa) or pd.isna(ob):
                    continue
                gain = 100 * (max(dec(oa), dec(ob)) / min(dec(oa), dec(ob)) - 1)
                if gain >= min_gain and oa != ob:
                    best = ("draftkings", oa, "fanduel", ob) if dec(oa) > dec(ob) else ("fanduel", ob, "draftkings", oa)
                    out.append(row("shop", g, pick1_book=best[0], pick1_side=side, pick1_line=ln, pick1_odds=best[1],
                                   pick2_book=best[2], pick2_side=side, pick2_line=ln, pick2_odds=best[3],
                                   value_pct=round(gain, 1), note="same line and side, better price at pick1"))
            # arb: Over at one book, Under at the other
            for bo, bu in ((a, b), (b, a)):
                po, pu = imp(bo.over_odds), imp(bu.under_odds)
                if pd.notna(po) and pd.notna(pu) and po + pu < 1:
                    out.append(row("arb", g, pick1_book=bo.book, pick1_side="OVER", pick1_line=ln, pick1_odds=bo.over_odds,
                                   pick2_book=bu.book, pick2_side="UNDER", pick2_line=ln, pick2_odds=bu.under_odds,
                                   value_pct=round(100 * (1 / (po + pu) - 1), 2), note="guaranteed profit if both fill; verify on the apps"))
        # ---------------- main lines
        ma, mb = dk[dk.is_main], fd[fd.is_main]
        if len(ma) and len(mb):
            ma, mb = ma.iloc[0], mb.iloc[0]
            ok = all(pd.notna(v) for v in (ma.over_odds, ma.under_odds, mb.over_odds, mb.under_odds))
            diffs.append(dict(game=g[0], player=g[1], stat=g[2], dk_line=ma.line, fd_line=mb.line, gap=abs(ma.line - mb.line)))
            if not ok:
                continue
            pa = nov(imp(ma.over_odds), imp(ma.under_odds))
            pb = nov(imp(mb.over_odds), imp(mb.under_odds))
            if ma.line == mb.line:
                fair = (pa + pb) / 2                                    # consensus no-vig P(over)
                for book, r in (("draftkings", ma), ("fanduel", mb)):
                    for side, col, p in (("OVER", "over_odds", fair), ("UNDER", "under_odds", 1 - fair)):
                        ev = 100 * (p * dec(r[col]) - 1)
                        if ev >= min_ev:
                            out.append(row("ev", g, pick1_book=book, pick1_side=side, pick1_line=r.line, pick1_odds=r[col],
                                           value_pct=round(ev, 1), note=f"fair P({side.lower()}) {100 * p:.1f}% from both books' no-vig prices"))
            else:
                lo, hi = (ma, mb) if ma.line < mb.line else (mb, ma)
                p_lo = nov(imp(lo.over_odds), imp(lo.under_odds))       # P(X > lo.line) at the low book
                p_hi = nov(imp(hi.over_odds), imp(hi.under_odds))       # P(X > hi.line) at the high book
                d_o, d_u = dec(lo.over_odds), dec(hi.under_odds)        # over at the LOW line, under at the HIGH line
                # outcomes: X < lo (under wins, over loses), lo < X < hi (both win), X > hi (over wins, under loses)
                p_lt = 1 - p_lo
                p_gt = p_hi
                p_mid = max(0.0, 1 - p_lt - p_gt)
                ev2 = p_lt * (d_u - 1 - 1) + p_mid * ((d_o - 1) + (d_u - 1)) + p_gt * (d_o - 1 - 1)
                if 100 * ev2 / 2 < min_middle_ev:
                    continue                                              # gaps that don't beat the vig aren't worth listing
                out.append(row("middle", g, pick1_book=lo.book, pick1_side="OVER", pick1_line=lo.line, pick1_odds=lo.over_odds,
                               pick2_book=hi.book, pick2_side="UNDER", pick2_line=hi.line, pick2_odds=hi.under_odds,
                               value_pct=round(100 * ev2 / 2, 1), win_both_pct=round(100 * p_mid, 1),
                               note=f"{hi.line - lo.line:g}-point gap; wins both if the result lands between {lo.line:g} and {hi.line:g}"))
    return pd.DataFrame(out), pd.DataFrame(diffs)


def best_bets(bt, edges):
    e = edges.copy()
    parts = e.player.astype(str).str.extract(r"^(.*?)\s*\(([^,()]*),\s*([^()]*)\)\s*$")
    e["name"], e["pos"], e["team"] = parts[0].fillna(e.player), parts[1], parts[2]
    e["key"] = e.name.map(norm)
    if "line_type" in e:
        e = e[e.line_type == "standard"]
    e = e[e.z.abs() > 0.05]
    sd = (e.edge.abs() / e.z.abs()).groupby(e.stat).median()          # typical per-stat spread implied by the edge file
    bt = bt.copy()
    bt["key"] = bt.player.map(norm)
    out = []
    for r in e.itertuples():
        x = bt[(bt.key == r.key) & (bt.stat == r.stat) & bt.is_main]
        if x.empty or pd.isna(sd.get(r.stat)):
            continue
        s = sd[r.stat]
        opts = []
        for o in x.itertuples():
            odds = o.over_odds if r.pick == "OVER" else o.under_odds
            if pd.isna(odds):
                continue
            pw = 1 - cdf((o.line - r.ours) / s) if r.pick == "OVER" else cdf((o.line - r.ours) / s)
            opts.append(dict(book=o.book, line=o.line, odds=odds, model_p=pw, ev=100 * (pw * dec(odds) - 1),
                             z=abs(r.ours - o.line) / s, game=o.game))
        if not opts:
            continue
        opts.sort(key=lambda d: -d["ev"])
        b = opts[0]
        alt = opts[1] if len(opts) > 1 else None
        out.append(dict(game=b["game"], player=r.name, pos=r.pos, team=r.team, stat=r.stat, pick=r.pick, ours=r.ours,
                        p_play=getattr(r, "p_play", np.nan),
                        best_book=b["book"], best_line=b["line"], best_odds=b["odds"], model_p_pct=round(100 * b["model_p"], 1),
                        ev_pct=round(b["ev"], 1), z=round(b["z"], 2),
                        other_book=alt["book"] if alt else None, other_line=alt["line"] if alt else None,
                        other_odds=alt["odds"] if alt else None, other_ev_pct=round(alt["ev"], 1) if alt else np.nan,
                        shop_gain_pts=round(b["ev"] - alt["ev"], 1) if alt else np.nan))
    return pd.DataFrame(out).sort_values("ev_pct", ascending=False).reset_index(drop=True) if out else pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("books", nargs="?", default="output/props_books.csv")
    ap.add_argument("--edges", help="prop_edges CSV from weekly_projections.py (adds the best-book column for our picks)")
    ap.add_argument("--min-ev", type=float, default=2.0, help="min expected value %% to list an 'ev' row")
    ap.add_argument("--min-gain", type=float, default=3.0, help="min price gain %% to list a 'shop' row")
    ap.add_argument("--min-middle-ev", type=float, default=0.0, help="min expected value %% for a 'middle' row (default 0)")
    a = ap.parse_args()

    bt = pd.read_csv(a.books)
    bt["book"] = bt.book.str.lower()
    if set(bt.book) < set(BOOKS) or bt.book.nunique() < 2:
        sys.exit(f"Need both {BOOKS} in {a.books}; found {sorted(bt.book.unique())}. Re-run fetch_props.py.")
    print(f"{len(bt)} rungs; players with lines at both books: "
          f"{bt.groupby(['game', 'player', 'stat']).book.nunique().eq(2).sum()} player-stats")

    res, diffs = book_vs_book(bt, a.min_ev, a.min_gain, a.min_middle_ev)
    if not diffs.empty:
        d = diffs
        print(f"\nMain lines: {len(d)} player-stats priced at both books; the lines differ on {(d.gap > 0).sum()} "
              f"({100 * (d.gap > 0).mean():.0f}%), average gap when they differ {d[d.gap > 0].gap.mean():.1f}.")
        print(d.assign(differs=d.gap > 0).groupby("stat").agg(n=("gap", "size"), differ=("differs", "sum"),
                                                              avg_gap=("gap", lambda s: round(s[s > 0].mean(), 1) if (s > 0).any() else 0))
              .to_string())
    cols = ["kind", "game", "player", "stat", "pick1_book", "pick1_side", "pick1_line", "pick1_odds",
            "pick2_book", "pick2_side", "pick2_line", "pick2_odds", "value_pct", "win_both_pct", "note"]
    if res.empty:
        print("\nNo opportunities above the thresholds.")
        pd.DataFrame(columns=cols).to_csv("line_shop.csv", index=False)
    else:
        for c in cols:
            if c not in res:
                res[c] = np.nan
        res = res[cols].sort_values(["kind", "value_pct"], ascending=[True, False])
        res.to_csv("line_shop.csv", index=False)
        print("\n" + res.kind.value_counts().to_string())
        for k in ("arb", "middle", "ev", "shop"):
            t = res[res.kind == k].head(12)
            if len(t):
                print(f"\n=== {k.upper()} (top {len(t)}) ===")
                print(t.drop(columns=["kind", "game"]).dropna(axis=1, how="all").to_string(index=False))
        print("\nWrote line_shop.csv")

    if a.edges:
        edges = pd.read_csv(a.edges)
        bb = best_bets(bt, edges)
        bb.to_csv("shop_best_bets.csv", index=False)
        print(f"\n=== MODEL PICKS, BEST BOOK ({len(bb)} picks) -> shop_best_bets.csv ===")
        if len(bb):
            print(bb.head(25).drop(columns=["game", "pos"]).to_string(index=False))
            print("\nshop_gain_pts = how many expected-value points the better book adds over the other for the same pick.")
    print("\nReminder: lines are delayed ~60s on the free plan and move fast; confirm the price on the app before betting.")


if __name__ == "__main__":
    main()