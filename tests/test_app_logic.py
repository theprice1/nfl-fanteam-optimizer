"""Executes app.py with a fake streamlit: catches NameErrors / logic errors (not widget-API mistakes)."""
import runpy, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import fake_streamlit

def run(press=(), state=None):
    f = fake_streamlit.install(press)
    f.session_state.update(state or {})
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except SystemExit:
        pass
    return f

def test_all():
    f = run()
    assert "_ctx" in f.session_state and len(f.session_state["_ctx"]["feat"]) == 204
    print("default run ok; ownership sum:", round(f.session_state["_ctx"]["feat"].own.sum()))
    # build every mode
    for mode in ("Cash", "GPP", "Contrarian", "Pure projection"):
        k = mode.lower().replace(" ", "_")
        f = run(press={f"{k}_run"}, state={"mode": mode, f"{k}_n": 3, f"{k}_pool": 30, f"{k}_sim": True} if mode in ("GPP", "Contrarian") else {"mode": mode})
        res = f.session_state["results"].get(mode)
        assert res and len(res["lineups"]) >= 1, mode
        print(mode, "lineups:", len(res["lineups"]))
    # post-lock: paste the saved Week 6 table text and fit
    import pandas as pd
    r = pd.read_csv(ROOT / "data/history/week06_results.csv")
    txt = "\n".join(f"{x.name}  {x.pos}  {x.team}  {x.price}M  " + (f"{x.own_pct}%" if pd.notna(x.own_pct) else "N/A") + "  "
                    + (f"{x.form}" if pd.notna(x.form) else "N/A") + f"  {x.score}" for x in r.itertuples())
    f = run(press={"Apply results", "Fit ownership model"}, state={"paste_text": txt})
    st = f.session_state
    print(st["paste_msg"], "| unmatched:", st["paste_unmatched"][:4], "| fit:", {k: round(v, 2) for k, v in st["fit"]["after"].items()})
    assert st["results_table"] is not None and len(st["results_table"]) == 190
    assert st["fit"]["weeks"] >= 2   # saved week06_training.csv + the pasted table
    # name,% fallback format
    f = run(press={"Apply results"}, state={"paste_text": "Josh Allen, 20.1%\nPuka Nacua, 41%"})
    assert f.session_state["own_paste"]["Josh Allen"] == 20.1
    # price list adds a missing player
    plist = "Dameon Pierce  RB  PHI  6.7M  3.20  \nDameon Pierce2  RB  PHI  6.7M"
    f = run(state={"price_text": "Dameon Pierce  RB  PHI  6.7M  13.06%  3.20  15.00"})
    names = set(f.session_state["_ctx"]["feat"].name)
    assert "Dameon Pierce" in names, "price-list player not added"
    print("price list ok")

if __name__ == "__main__":
    test_all(); print("OK")
