"""Smoke tests: run with `python tests/test_engine.py` (or pytest)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine as E

DATA = Path(__file__).resolve().parents[1] / "data"

def prep():
    players = E.load_players(pd.read_csv(DATA / "nfl_fanteam_week6.csv"))
    games = E.load_games(pd.read_csv(DATA / "week6_games.csv"))
    d = E.add_features(players, E.implied_map(games))
    rules = E.Rules()
    d["own"] = E.project_ownership(d, E.OwnParams(), rules)
    return d, rules

def check_lineup(d, rules, L):
    sub = d.loc[L]
    assert len(L) == rules.roster_size and len(set(L)) == len(L)
    assert sub.salary.sum() <= rules.budget + 1e-6
    for pos in E.POS_ORDER:
        c = (sub.pos == pos).sum()
        assert rules.slot_min[pos] <= c <= rules.slot_max(pos), (pos, c)
    assert sub.groupby("team").size().max() <= rules.max_per_team

def test_all():
    d, rules = prep()
    assert abs(d.groupby("pos").own.sum() / 100 - pd.Series({"QB":1,"RB":2.4,"WR":3.45,"TE":1.15,"DST":1})).abs().max() < 1e-2
    for mode in ["Cash", "GPP", "Contrarian", "Pure projection"]:
        cfg = E.default_cfg(mode)
        if cfg["use_sim"]:
            cfg["pool_size"] = 40; cfg["n"] = 8
        t = time.time()
        res = E.build_lineups(d[d.status != "unexpected"], rules, cfg)
        for L in res["lineups"]:
            check_lineup(res["pool"], rules, L)
        s = res["stats"]
        print(f"\n== {mode}: {len(res['lineups'])} lineups in {time.time()-t:.1f}s")
        print(s[["lineup","salary","left","proj","own_sum","chalk","p_cash","p_top1","stack"]].round(3).head(6).to_string(index=False))
        print(res["notes"])

if __name__ == "__main__":
    test_all(); print("\nOK")
