"""Parses a synthetic page shaped like dailyfantasy.tips/nfl (tables per site, games table, week banner)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scraper as S, engine as E

DATA = Path(__file__).resolve().parents[1] / "data"

def html_table(df, strong_cols=()):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for r in df.itertuples(index=False):
        cells = "".join(f"<td>{'<strong>%s</strong>' % v if c in strong_cols else v}</td>" for c, v in zip(df.columns, r))
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

def make_page():
    fan = pd.read_csv(DATA / "nfl_fanteam_week6.csv")
    dk = pd.DataFrame({"Player": ["Josh Allen", "Bijan Robinson"], "Pos": ["QB", "RB"], "Team": ["BUF", "ATL"], "Opp": ["LV", "CHI"],
                       "Salary": ["$7,500", "$8,700"], "Proj": [25.6, 25.6], "Value": [3.41, 2.94], "DK ID": [1, 2]})
    games = pd.DataFrame({"Game": ["BUF @ LV", "ARI @ LAR", "CHI @ ATL"], "Total": ["**53.5**", "**51.0**", ""],
                          "Favourite": ["BUF -4.5", "LAR -13", "Pick'em"], "Away implied": ["BUF 29.0", "ARI 19.0", "CHI"],
                          "Home implied": ["LV 24.5", "LAR 32.0", "ATL"]})
    vals = pd.DataFrame({"Player": ["LAR Matthew Stafford"], "Salary": ["$6,400"], "Proj": [22.5], "Value": [3.51]})
    return f"""<html><body><h1>NFL DFS Projections</h1>
    <h2>This week's NFL games and implied totals</h2><p>WEEK 6<span>Last updated: 18/10/2026, 19:23</span></p>{html_table(games)}
    <h2>DraftKings</h2><h3>Value plays</h3>{html_table(vals)}<h3>All projections</h3>{html_table(dk, ('Proj',))}
    <h2>Fanteam</h2><h3>Value plays</h3>{html_table(vals)}<h3>All projections</h3>{html_table(fan, ('Proj',))}
    </body></html>"""

def test_all():
    res = S.scrape(make_page())
    assert res["week"] == 6 and res["updated"] == "18/10/2026, 19:23", (res["week"], res["updated"])
    p = res["players"]
    assert len(p) == 204 and "Fanteam ID" in p.columns, (len(p), p.columns.tolist())
    orig = pd.read_csv(DATA / "nfl_fanteam_week6.csv")
    assert np.allclose(p["Proj"], orig["Proj"]) and np.allclose(p["Salary"], orig["Salary"])
    g = res["games"]
    assert g.iloc[0].to_dict() == {"away": "BUF", "home": "LV", "total": 53.5, "away_implied": 29.0, "home_implied": 24.5}
    assert np.isnan(g.iloc[2].total) and np.isnan(g.iloc[2].home_implied)
    E.load_players(p)   # the engine accepts the scraped frame unchanged
    print(S.describe_tables(res["tables"]).splitlines()[0][:110])
    out = S.save(res, "/tmp/scrape_out"); print([str(x) for x in out])

if __name__ == "__main__":
    test_all(); print("OK")
