"""
app.py - Fanteam NFL lineup optimizer (Streamlit).

Run locally:   streamlit run app.py
Deploy:        push to GitHub, then create the app on share.streamlit.io (main file: app.py)
"""
from __future__ import annotations

import io
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import engine as E
import scraper
from theme import hero, inject_css, lineup_card

st.set_page_config(page_title="Gridiron Lab - Fanteam NFL", page_icon="🏈", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

DATA = Path(__file__).parent / "data"
POS_RANK = {p: i for i, p in enumerate(E.POS_ORDER)}
MODE_HELP = {
    "Cash": "Floor-weighted projection, no ownership penalty. For head-to-heads, 50/50s and double-ups.",
    "GPP": "Ceiling-weighted with a QB stack and bring-back, a mild ownership penalty, and lineups ranked by "
           "simulated top-1% odds against a field built from the ownership projection.",
    "Contrarian": "GPP with a heavier ownership penalty, a cap on chalk players and total ownership, and forced "
                  "low-owned pieces. Fewer cashes, more upside when the chalk misses.",
    "Pure projection": "Straight projected points with no risk or ownership adjustments.",
}
OWN_KEYS = asdict(E.OwnParams())
for _k, _v in OWN_KEYS.items():
    st.session_state.setdefault(f"op_{_k}", _v)
st.session_state.setdefault("own_ver", 0)
st.session_state.setdefault("own_manual", {})
st.session_state.setdefault("own_paste", {})
st.session_state.setdefault("calib_log", [])
st.session_state.setdefault("results", {})


@st.cache_data(show_spinner=False)
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))


# ----------------------------------------------------------------------------
# Callbacks (run before the script reruns, so state is ready for every tab)
# ----------------------------------------------------------------------------
def cb_fetch() -> None:
    try:
        res = scraper.scrape()
        st.session_state["fetched"] = {k: res[k] for k in ("players", "games", "week", "updated")}
        st.session_state["fetch_msg"] = (f"Fetched week {res['week']} (page updated {res['updated']}): "
                                         f"{len(res['players'])} players, {len(res['games'])} games.")
    except Exception as exc:  # network or page-layout problem: show it, keep the app running
        st.session_state["fetch_msg"] = f"Fetch failed: {exc}"


def cb_clear_fetch() -> None:
    st.session_state.pop("fetched", None)
    st.session_state["fetch_msg"] = ""


def cb_reset_params() -> None:
    for k, v in asdict(E.OwnParams()).items():
        st.session_state[f"op_{k}"] = float(v)


def cb_apply_results() -> None:
    """Read a pasted Fanteam table (or simple 'Name, 23.4%' lines) and store real ownership."""
    ctx, text = st.session_state.get("_ctx"), st.session_state.get("paste_text", "")
    if not ctx:
        return
    feat = ctx["feat"]
    table = E.parse_fanteam_table(text)
    if len(table):
        pct, unmatched = {}, []
        for r in table.itertuples(index=False):
            hit = E._match_row(feat, r.name, r.pos, r.team)
            if hit is None:
                unmatched.append(r.name)
            else:
                pct[feat.at[hit, "name"]] = 0.0 if pd.isna(r.own_pct) else float(r.own_pct)
        played = set(pct)
        if len(table) >= 60:  # a full results table: anyone not in it did not play (safety net replaced them)
            for n in feat["name"]:
                pct.setdefault(n, 0.0)
        st.session_state.update(results_table=table, own_paste=pct, played_names=played, paste_unmatched=unmatched,
                                paste_msg=f"Read {len(table)} players, matched {len(played)}.")
        return
    rows = E.parse_ownership_text(text)
    if not rows:
        st.session_state["paste_msg"] = "Nothing recognised. Paste the Fanteam table or 'Player, 23.4%' lines."
        return
    matched, unmatched = E.match_ownership(rows, feat)
    pct = dict(zip(matched["name"], matched["actual"]))
    st.session_state.update(results_table=None, own_paste=pct, played_names=set(pct), paste_unmatched=unmatched,
                            paste_msg=f"Matched {len(matched)} of {len(rows)} lines.")


def cb_clear_overrides() -> None:
    st.session_state["own_paste"] = {}
    st.session_state["own_manual"] = {}
    st.session_state["results_table"] = None
    st.session_state["played_names"] = set()
    st.session_state["own_ver"] += 1
    st.session_state["paste_unmatched"] = []
    st.session_state["paste_msg"] = ""


def cb_fit() -> None:
    ctx, table = st.session_state.get("_ctx"), st.session_state.get("results_table")
    frames = E.load_history(DATA / "history")
    if ctx and table is not None and len(table):
        frames = frames + [E.build_training_frame(ctx["feat"], table)]
    if not ctx or not frames:
        st.session_state["paste_msg"] = "Apply a full Fanteam results table first (or save earlier weeks in data/history)."
        return
    fit, _, after = E.fit_ownership(frames, ctx["rules"], E.OwnParams())
    before = E.evaluate_ownership(frames, ctx["params"], ctx["rules"])
    st.session_state["fit"] = {"params": asdict(fit), "before": before, "after": after, "weeks": len(frames)}


def cb_apply_fit() -> None:
    fit = st.session_state.get("fit")
    if fit:
        for k in E.FIT_FIELDS:
            st.session_state[f"op_{k}"] = float(fit["params"][k])


def cb_log() -> None:
    fit = st.session_state.get("fit")
    if not fit:
        return
    st.session_state["calib_log"].append({
        "slate": st.session_state.get("slate_name", ""), "logged": time.strftime("%Y-%m-%d %H:%M"),
        "players": fit["before"]["n"], "mae_before": round(fit["before"]["mae"], 3),
        "mae_after": round(fit["after"]["mae"], 3), "corr_before": round(fit["before"]["corr"], 3),
        "corr_after": round(fit["after"]["corr"], 3), **{f"fit_{k}": round(v, 3) for k, v in fit["params"].items()},
    })


# ----------------------------------------------------------------------------
# Sidebar: data, rules, models
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Slate")
    slate_name = st.text_input("Slate name", "NFL Week 6", key="slate_name")
    up = st.file_uploader("Projections CSV", type="csv",
                          help="Daily Fantasy Tips Fanteam export. Leave empty to use the file in /data.")
    gup = st.file_uploader("Games CSV (optional)", type="csv",
                           help="Columns: away, home, total, away_implied, home_implied")
    f1, f2 = st.columns(2)
    f1.button("Fetch latest", on_click=cb_fetch,
              help="Pulls this week's Fanteam projections and game totals from dailyfantasy.tips")
    f2.button("Clear fetch", on_click=cb_clear_fetch)
    if st.session_state.get("fetch_msg"):
        st.caption(st.session_state["fetch_msg"])
    with st.expander("Fanteam player list (optional)", expanded=False):
        st.caption("Paste the Fanteam player table. Players missing from the projections are added with a "
                   "price-and-form estimate, so cheap plays the source skipped are not lost.")
        st.text_area("Fanteam table", key="price_text", height=110, label_visibility="collapsed")

    with st.expander("Contest rules", expanded=False):
        st.caption("Check these against the Fanteam lobby each week.")
        budget = st.number_input("Budget (credits)", 50.0, 300.0, 110.0, 0.5)
        max_team = st.number_input("Max players per NFL team", 1, 9, 3, 1)
        dst_team = st.toggle("Defense counts toward team limit", True)
        c = st.columns(5)
        mins = {p: c[i].number_input(p, 0, 3, v, 1, key=f"min_{p}")
                for i, (p, v) in enumerate({"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}.items())}
        flex = st.toggle("FLEX slot (RB / WR / TE)", True)

    with st.expander("Availability", expanded=False):
        st.caption("Chance a player plays. Scales every projection in the optimizer and simulation.")
        pp_exp = st.slider("Expected", 0.80, 1.00, 0.98, 0.01)
        pp_pos = st.slider("Possible", 0.20, 1.00, 0.70, 0.05)
        pp_unx = st.slider("Unexpected", 0.00, 0.80, 0.25, 0.05)

    with st.expander("Ownership model", expanded=False):
        st.caption("Fitted on Week 6 Fanteam results. Refit after each lock on the Post-lock tab.")
        st.slider("Overall sharpness", 0.3, 2.5, step=0.05, key="op_sharp")
        st.slider("Projection", -0.5, 2.0, step=0.02, key="op_b_proj")
        st.slider("Points per credit", -0.5, 2.0, step=0.02, key="op_b_value")
        st.slider("Recent form", -0.5, 2.0, step=0.02, key="op_b_form")
        st.slider("Price (negative = cheap is popular)", -1.5, 0.5, step=0.02, key="op_b_price")
        st.slider("Team implied total", -0.5, 1.0, step=0.02, key="op_b_team")
        st.slider("Penalty: possible", 0.0, 3.0, step=0.05, key="op_pen_possible")
        st.slider("Penalty: unexpected", 0.0, 4.0, step=0.1, key="op_pen_unexpected")
        st.caption("Spread by position (higher = more concentrated)")
        for _pos in ("qb", "rb", "wr", "te", "dst"):
            st.slider(_pos.upper(), 0.3, 2.5, step=0.05, key=f"op_tau_{_pos}")
        st.button("Reset to fitted defaults", on_click=cb_reset_params)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
hero_slot = st.container()

def _week_no(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def _latest(pattern: str) -> Path | None:
    files = sorted(DATA.glob(pattern), key=_week_no)
    return files[-1] if files else None


fetched = st.session_state.get("fetched")
try:
    if up is not None:
        raw, data_label = read_csv_bytes(up.getvalue()), up.name
    elif fetched is not None:
        raw, data_label = fetched["players"], f"dailyfantasy.tips week {fetched['week']} ({fetched['updated']})"
    else:
        latest = _latest("nfl_fanteam*.csv")
        if latest is None:
            st.info("Upload a projections CSV or press Fetch latest in the sidebar to begin.")
            st.stop()
        raw, data_label = read_csv_bytes(latest.read_bytes()), latest.name
    players = E.load_players(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

added_players: list[str] = []
price_df = E.parse_fanteam_table(st.session_state.get("price_text", ""))
if len(price_df):
    players, added_players = E.add_unprojected(players, price_df)

if gup is not None:
    games0 = E.load_games(read_csv_bytes(gup.getvalue()))
elif fetched is not None and len(fetched["games"]) and fetched["games"]["away_implied"].notna().any():
    games0 = E.load_games(fetched["games"])
else:
    lg = _latest("*games*.csv")
    games0 = E.load_games(read_csv_bytes(lg.read_bytes())) if lg else E.games_from_players(players)
sig = f"{data_label}-{len(players)}-{len(games0)}-{int(games0['total'].fillna(0).sum() * 10)}"

tab_slate, tab_own, tab_build, tab_post = st.tabs(["Slate", "Ownership", "Build lineups", "Post-lock"])

# ----------------------------------------------------------------------------
# Slate tab (part 1: games)
# ----------------------------------------------------------------------------
with tab_slate:
    st.markdown("#### Games and implied totals")
    st.caption("Edit totals or implied points if lines move. Blank implied totals fall back to the slate average.")
    games = st.data_editor(
        games0, key=f"games_{sig}", hide_index=True, num_rows="fixed",
        column_config={
            "away": st.column_config.TextColumn("Away", disabled=True),
            "home": st.column_config.TextColumn("Home", disabled=True),
            "total": st.column_config.NumberColumn("Total", format="%.1f", step=0.5),
            "away_implied": st.column_config.NumberColumn("Away implied", format="%.1f", step=0.25),
            "home_implied": st.column_config.NumberColumn("Home implied", format="%.1f", step=0.25),
        },
    )

# ----------------------------------------------------------------------------
# Model: features + ownership
# ----------------------------------------------------------------------------
rules = E.Rules(budget=float(budget), slot_min={p: int(v) for p, v in mins.items()}, flex=bool(flex),
                max_per_team=int(max_team), dst_counts_for_team=bool(dst_team))
play_prob = {"expected": pp_exp, "possible": pp_pos, "unexpected": pp_unx}
params = E.OwnParams(**{k: float(st.session_state[f"op_{k}"]) for k in OWN_KEYS})
feat = E.add_features(players, E.implied_map(E.load_games(games)), play_prob)
feat["own_model"] = E.project_ownership(feat, params, rules)
st.session_state["_ctx"] = {"feat": feat, "rules": rules, "params": params}


def apply_layers(base: pd.Series, layers: list[dict]) -> pd.Series:
    """Later layers win: model < pasted post-lock < manual overrides."""
    out = base.copy()
    for layer in layers:
        if layer:
            mapped = feat["name"].map(layer)
            out = mapped.where(mapped.notna(), out)
    return out


with hero_slot:
    hero("Gridiron Lab", f"Fanteam NFL optimizer - {slate_name}",
         [(r.away, r.home, None if pd.isna(r.total) else float(r.total)) for r in games.itertuples()])

# ----------------------------------------------------------------------------
# Slate tab (part 2: players)
# ----------------------------------------------------------------------------
with tab_slate:
    st.markdown("#### Players")
    st.caption(f"Source: {data_label}")
    if added_players:
        st.info(f"Added {len(added_players)} players missing from the projections (estimated from price and form): "
                + ", ".join(added_players[:12]) + ("..." if len(added_players) > 12 else ""))
    f1, f2, f3, f4 = st.columns([1.2, 1.4, 1.2, 1.2])
    pos_sel = f1.multiselect("Position", E.POS_ORDER, default=E.POS_ORDER)
    team_sel = f2.multiselect("Team", sorted(feat.team.unique()))
    status_sel = f3.multiselect("Status", ["expected", "possible", "unexpected"],
                                default=["expected", "possible", "unexpected"])
    search = f4.text_input("Search player")
    view = feat[feat.pos.isin(pos_sel) & feat.status.isin(status_sel)]
    if team_sel:
        view = view[view.team.isin(team_sel)]
    if search:
        view = view[view.name.str.contains(search, case=False, regex=False)]
    show = view.sort_values("proj", ascending=False)[
        ["name", "pos", "team", "opp", "salary", "proj", "value", "floor", "ceil", "status", "implied", "form", "last"]]
    st.dataframe(
        show, hide_index=True, height=460,
        column_config={
            "name": "Player", "pos": "Pos", "team": "Team", "opp": "Opp",
            "salary": st.column_config.NumberColumn("Salary", format="%.1f"),
            "proj": st.column_config.NumberColumn("Proj", format="%.2f"),
            "value": st.column_config.NumberColumn("Pts/credit", format="%.2f"),
            "floor": st.column_config.NumberColumn("Floor", format="%.1f", help="20th percentile outcome"),
            "ceil": st.column_config.NumberColumn("Ceiling", format="%.1f", help="90th percentile outcome"),
            "status": "Status", "implied": st.column_config.NumberColumn("Team pts", format="%.1f"),
            "form": st.column_config.NumberColumn("Form", format="%.1f"),
            "last": st.column_config.NumberColumn("Last", format="%.1f"),
        },
    )
    st.caption("Salary against projection - players well above the cloud are the value plays.")
    st.scatter_chart(view, x="salary", y="proj", color="pos", height=320)

# ----------------------------------------------------------------------------
# Ownership tab
# ----------------------------------------------------------------------------
with tab_own:
    st.markdown("#### Projected ownership")
    st.caption("Model estimate, then your post-lock paste, then any manual edit. The most specific source wins. "
               "Edits here feed the optimizer, the contrarian limits and the field simulation.")
    own_df = (feat.assign(_p=feat.pos.map(POS_RANK)).sort_values(["_p", "proj"], ascending=[True, False])
              [["name", "pos", "team", "salary", "proj", "own_model"]].copy())
    own_df["override"] = np.nan
    edited = st.data_editor(
        own_df, key=f"own_ed_{st.session_state['own_ver']}", hide_index=True, height=420,
        disabled=["name", "pos", "team", "salary", "proj", "own_model"],
        column_config={
            "name": "Player", "pos": "Pos", "team": "Team",
            "salary": st.column_config.NumberColumn("Salary", format="%.1f"),
            "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
            "own_model": st.column_config.NumberColumn("Model own %", format="%.1f"),
            "override": st.column_config.NumberColumn("Override %", min_value=0.0, max_value=100.0, step=0.5,
                                                      format="%.1f", help="Type your own ownership estimate"),
        },
    )
    st.session_state["own_manual"] = {n: float(v) for n, v in zip(edited["name"], edited["override"]) if pd.notna(v)}
    feat["own"] = apply_layers(feat["own_model"], [st.session_state["own_paste"], st.session_state["own_manual"]])
    st.button("Clear pasted and manual ownership", on_click=cb_clear_overrides)

    top = feat.sort_values("own", ascending=False).head(18)
    st.markdown("##### Most owned")
    st.bar_chart(top.set_index("name")["own"], height=260)
    st.caption(f"Typical field lineup owns about {float((feat.own / 100 * feat.own).sum()):.0f}% in total across nine players.")

# ----------------------------------------------------------------------------
# Build tab
# ----------------------------------------------------------------------------
with tab_build:
    mode = st.radio("Lineup type", ["Cash", "GPP", "Contrarian", "Pure projection"], horizontal=True, key="mode")
    st.caption(MODE_HELP[mode])
    k = mode.lower().replace(" ", "_")
    cfg = E.default_cfg(mode)
    names_sorted = feat.sort_values("proj", ascending=False)["name"].tolist()
    qbs = feat[feat.pos == "QB"].sort_values("proj", ascending=False)["name"].tolist()

    a, b, c = st.columns(3)
    cfg["n"] = int(a.number_input("Lineups", 1, 150, cfg["n"], 1, key=f"{k}_n"))
    cfg["min_unique"] = int(b.number_input("Min different players between lineups", 1, 6, cfg["min_unique"], 1,
                                           key=f"{k}_uniq"))
    cfg["max_exposure"] = float(c.slider("Max exposure per player", 0.1, 1.0, float(cfg["max_exposure"]), 0.05,
                                         key=f"{k}_exp"))

    if mode == "Cash":
        cfg["floor_w"] = st.slider("Floor weight (0 = projection only, 1 = floor only)", 0.0, 1.0,
                                   cfg["floor_w"], 0.05, key=f"{k}_fw")
    if mode in ("GPP", "Contrarian"):
        a, b, c = st.columns(3)
        cfg["upside_w"] = a.slider("Upside weight (ceiling vs projection)", 0.0, 1.0, cfg["upside_w"], 0.05,
                                   key=f"{k}_uw")
        cfg["lev"] = b.slider("Ownership leverage", 0.0, 1.0, cfg["lev"], 0.05, key=f"{k}_lev",
                              help="Discounts popular players. 0.5 trims a 40%-owned player's value by 20%.")
        cfg["pct_opt"] = c.slider("Stay within % of best projection", 0.70, 1.0, cfg["pct_opt"], 0.01,
                                  key=f"{k}_pct")
        a, b, c = st.columns(3)
        cfg["stack_min"] = int(a.slider("QB stack: teammates (WR/TE)", 0, 3, cfg["stack_min"], 1, key=f"{k}_stk"))
        cfg["bring_back"] = int(b.slider("Bring-back from opponent", 0, 2, cfg["bring_back"], 1, key=f"{k}_bb"))
        cfg["rand"] = float(c.slider("Variety (random noise)", 0.0, 0.4, float(cfg["rand"]), 0.01, key=f"{k}_rand"))
    if mode == "Contrarian":
        with st.expander("Contrarian limits", expanded=True):
            a, b, c = st.columns(3)
            cfg["max_chalk"] = int(a.slider("Max chalk players", 0, 5, cfg["max_chalk"], 1, key=f"{k}_mc"))
            cfg["chalk_thr"] = float(b.slider("Chalk means owned above %", 10.0, 40.0, cfg["chalk_thr"], 1.0,
                                              key=f"{k}_ct"))
            cfg["own_cap_pct"] = float(c.slider("Total ownership vs typical field lineup (%)", 40.0, 120.0,
                                                cfg["own_cap_pct"], 5.0, key=f"{k}_cap"))
            a, b, c = st.columns(3)
            cfg["low_min"] = int(a.slider("Min low-owned players", 0, 6, cfg["low_min"], 1, key=f"{k}_lm"))
            cfg["low_thr"] = float(b.slider("Low-owned means below %", 2.0, 20.0, cfg["low_thr"], 1.0, key=f"{k}_lt"))
            cfg["uniq_w"] = float(c.slider("Reward unique lineups when ranking", 0.0, 1.5, cfg["uniq_w"], 0.1,
                                           key=f"{k}_uw2"))
    if mode in ("GPP", "Contrarian"):
        a, b = st.columns(2)
        cfg["use_sim"] = a.toggle("Rank by simulated top-1% odds", cfg["use_sim"], key=f"{k}_sim")
        cfg["pool_size"] = int(b.slider("Candidate pool size", 20, 300, cfg["pool_size"], 10, key=f"{k}_pool",
                                        disabled=not cfg["use_sim"]))

    with st.expander("Player pool and extra rules", expanded=False):
        a, b = st.columns(2)
        drop_unx = a.toggle("Exclude 'unexpected' players", True, key=f"{k}_dropunx")
        drop_pos = b.toggle("Exclude 'possible' players", False, key=f"{k}_droppos")
        a, b = st.columns(2)
        min_proj = a.slider("Ignore players projected below", 0.0, 10.0, 2.0, 0.5, key=f"{k}_minproj")
        cfg["dst_vs_offense"] = b.toggle("No defense facing my offensive players", True, key=f"{k}_dstvo")
        cfg["locks"] = st.multiselect("Lock players in", names_sorted, key=f"{k}_locks")
        banned = st.multiselect("Remove players (out or fade)", names_sorted, key=f"{k}_ban")
        cfg["qb_allowed"] = st.multiselect("Only these quarterbacks", qbs, key=f"{k}_qbs")
        excl_teams = st.multiselect("Exclude teams", sorted(feat.team.unique()), key=f"{k}_xt")
        a, b = st.columns(2)
        if a.toggle("Limit unspent credits", False, key=f"{k}_lo"):
            cfg["max_leftover"] = float(b.slider("Max credits left", 0.0, 10.0, 2.0, 0.5, key=f"{k}_lov"))
        a, b = st.columns(2)
        cfg["n_sims"] = int(a.select_slider("Simulations", [1000, 2000, 4000], 2000, key=f"{k}_nsim"))
        cfg["seed"] = int(b.number_input("Random seed", 0, 9999, 1, 1, key=f"{k}_seed"))

    if st.button(f"Build {mode} lineups", type="primary", key=f"{k}_run"):
        pool = feat.copy()
        if drop_unx:
            pool = pool[pool.status != "unexpected"]
        if drop_pos:
            pool = pool[pool.status != "possible"]
        pool = pool[(pool.proj >= min_proj) & ~pool.name.isin(banned) & ~pool.team.isin(excl_teams)]
        pool = pool.reset_index(drop=True)
        bar = st.progress(0.0, text="Optimizing...")
        try:
            res = E.build_lineups(pool, rules, cfg, progress=lambda f: bar.progress(min(f, 1.0), text=f"Optimizing {f:.0%}"))
            res["rules"], res["cfg"], res["built"] = rules, cfg, time.strftime("%H:%M:%S")
            st.session_state["results"][mode] = res
        except ValueError as exc:
            st.session_state["results"].pop(mode, None)
            st.error(str(exc))
        bar.empty()

    res = st.session_state["results"].get(mode)
    if res:
        pool, stats, rl = res["pool"], res["stats"], res["rules"]
        st.markdown(f"#### {mode} lineups")
        m = st.columns(5)
        m[0].metric("Lineups", len(stats))
        m[1].metric("Avg projection", f"{stats.proj.mean():.1f}")
        m[2].metric("Avg ownership sum", f"{stats.own_sum.mean():.0f}%")
        m[3].metric("Best top-1% odds", f"{stats.p_top1.max() * 100:.1f}%")
        m[4].metric("Avg cash odds", f"{stats.p_cash.mean() * 100:.0f}%")
        for note in res["notes"]:
            st.caption(note)
        ex = E.exposure_table(res)
        unsure = ex[ex.name.isin(pool[pool.status != "expected"].name)]
        if len(unsure):
            st.warning("Confirm before lock: " + ", ".join(
                f"{r.name} ({pool.loc[pool.name == r.name, 'status'].iloc[0]}, {r.exposure:.0f}% of lineups)"
                for r in unsure.itertuples()))

        v1, v2, v3 = st.tabs(["Lineups", "Table", "Exposure"])
        with v1:
            limit = len(stats) if st.toggle("Show all lineups", len(stats) <= 10, key=f"{k}_showall") else 10
            for i, L in enumerate(res["lineups"][:limit]):
                slots = [(s, pool.loc[j].to_dict()) for s, j in E.order_slots(pool, L, rl)]
                st.markdown(lineup_card(i + 1, slots, stats.iloc[i].to_dict()), unsafe_allow_html=True)
        with v2:
            tbl = stats.drop(columns=["idx"]).copy()
            for col in ("p_cash", "p_top10", "p_top1"):
                tbl[col] = tbl[col] * 100
            st.dataframe(tbl, hide_index=True, column_config={
                "lineup": "#", "salary": st.column_config.NumberColumn("Spend", format="%.1f"),
                "left": st.column_config.NumberColumn("Left", format="%.1f"),
                "proj": st.column_config.NumberColumn("Proj", format="%.1f"),
                "proj_adj": st.column_config.NumberColumn("Proj (availability)", format="%.1f"),
                "floor": st.column_config.NumberColumn("Floor", format="%.1f"),
                "ceil": st.column_config.NumberColumn("Ceiling", format="%.1f"),
                "own_sum": st.column_config.NumberColumn("Own sum %", format="%.0f"),
                "own_avg": st.column_config.NumberColumn("Own avg %", format="%.1f"),
                "p_cash": st.column_config.ProgressColumn("Cash odds %", min_value=0.0, max_value=100.0, format="%.0f"),
                "p_top10": st.column_config.ProgressColumn("Top 10% odds", min_value=0.0, max_value=100.0, format="%.1f"),
                "p_top1": st.column_config.ProgressColumn("Top 1% odds", min_value=0.0, max_value=30.0, format="%.1f"),
                "sim_mean": st.column_config.NumberColumn("Sim mean", format="%.1f"),
                "sim_p90": st.column_config.NumberColumn("Sim 90th", format="%.1f"),
            })
        with v3:
            st.dataframe(ex, hide_index=True, column_config={
                "exposure": st.column_config.ProgressColumn("Exposure %", min_value=0.0, max_value=100.0, format="%.0f"),
                "own": st.column_config.NumberColumn("Field own %", format="%.1f"),
                "leverage": st.column_config.NumberColumn("Leverage", format="%+.1f", help="Exposure minus field ownership"),
            })
        d1, d2 = st.columns(2)
        d1.download_button("Download lineups (CSV)", E.lineups_to_frame(res, rl).to_csv(index=False),
                           file_name=f"{slate_name.replace(' ', '_').lower()}_{k}_lineups.csv", mime="text/csv")
        d2.download_button("Download exposure (CSV)", ex.to_csv(index=False),
                           file_name=f"{slate_name.replace(' ', '_').lower()}_{k}_exposure.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Post-lock tab
# ----------------------------------------------------------------------------
with tab_post:
    st.markdown("#### Post-lock results and calibration")
    st.caption("After lock, copy the whole Fanteam table (Player, Pos, Team, Price, Own %, Form, Score) into the box. "
               "Anyone missing from a full table did not play and counts as 0%. The app swaps in the real ownership, "
               "checks both the ownership and the projections, and can refit the model on every week you have saved.")
    st.text_area("Fanteam results table", height=200, key="paste_text",
                 placeholder="Malik Nabers   WR   NYG   13.8M   30.23%   14.78   28.50\nKenneth Walker   RB   KC   19.1M   30.95%   27.52   27.90")
    c1, c2 = st.columns(2)
    c1.button("Apply results", type="primary", on_click=cb_apply_results)
    c2.button("Fit ownership model", on_click=cb_fit)
    if st.session_state.get("paste_msg"):
        st.caption(st.session_state["paste_msg"])
    if st.session_state.get("paste_unmatched"):
        st.warning("Not in the projections file (kept out of the comparison): "
                   + ", ".join(st.session_state["paste_unmatched"][:15]))

    paste, table = st.session_state["own_paste"], st.session_state.get("results_table")
    played = st.session_state.get("played_names") or set(paste)
    if paste:
        chk = feat[feat.name.isin(played)][["name", "pos", "team", "own_model"]].copy()
        chk["actual"] = chk.name.map(paste)
        chk["error"] = chk.own_model - chk.actual
        err = E.ownership_error(chk.own_model, chk.actual)
        st.markdown("##### Ownership: estimate vs real")
        m = st.columns(3)
        m[0].metric("Players compared", err["n"])
        m[1].metric("Average miss (pts of own%)", f"{err['mae']:.2f}")
        m[2].metric("Correlation", "n/a" if np.isnan(err["corr"]) else f"{err['corr']:.2f}")
        st.dataframe(chk.reindex(chk.error.abs().sort_values(ascending=False).index), hide_index=True, height=300,
                     column_config={"name": "Player", "pos": "Pos", "team": "Team",
                                    "own_model": st.column_config.NumberColumn("Model %", format="%.1f"),
                                    "actual": st.column_config.NumberColumn("Actual %", format="%.1f"),
                                    "error": st.column_config.NumberColumn("Model minus actual", format="%+.1f")})

    if table is not None and table["score"].notna().any():
        rows = []
        for r in table.itertuples(index=False):
            hit = E._match_row(feat, r.name, r.pos, r.team)
            if hit is not None and pd.notna(r.score) and feat.at[hit, "proj_src"] == "source":
                rows.append({"pos": r.pos, "proj": float(feat.at[hit, "proj"]), "score": float(r.score)})
        acc = pd.DataFrame(rows)
        if len(acc):
            st.markdown("##### Projections: expected vs real points")
            out = []
            for pos, g in acc.groupby("pos"):
                resid_cv = float((g.score - g.proj).std() / g.proj.mean())
                out.append({"pos": pos, "players": len(g), "avg_proj": g.proj.mean(), "avg_score": g.score.mean(),
                            "bias": g.score.mean() - g.proj.mean(),
                            "corr": float(g.proj.corr(g.score)) if len(g) > 2 else np.nan,
                            "implied_sigma": float(np.sqrt(np.log(1 + resid_cv ** 2))), "model_sigma": E.SIGMA[pos]})
            st.dataframe(pd.DataFrame(out), hide_index=True, column_config={
                "avg_proj": st.column_config.NumberColumn("Avg proj", format="%.1f"),
                "avg_score": st.column_config.NumberColumn("Avg score", format="%.1f"),
                "bias": st.column_config.NumberColumn("Bias", format="%+.1f"),
                "corr": st.column_config.NumberColumn("Corr", format="%.2f"),
                "implied_sigma": st.column_config.NumberColumn("Spread seen", format="%.2f",
                                                               help="Log-normal spread implied by this week's misses"),
                "model_sigma": st.column_config.NumberColumn("Spread in model", format="%.2f"),
            })
            st.caption("Spread seen well above the model's means the simulation is too tame for that position. "
                       "One week is noisy, so adjust SIGMA in engine.py only after several weeks agree.")
        training = E.build_training_frame(feat, table)
        st.download_button("Download training file for history (CSV)", training.to_csv(index=False),
                           file_name=f"{slate_name.replace(' ', '_').lower()}_training.csv", mime="text/csv",
                           help="Save into data/history/ and commit. Every saved week is used when you fit the model.")

    hist = E.load_history(DATA / "history")
    st.caption(f"{len(hist)} earlier week(s) saved in data/history are used when fitting.")
    fit = st.session_state.get("fit")
    if fit:
        st.markdown("##### Fitted settings")
        m = st.columns(4)
        m[0].metric("Weeks used", fit["weeks"])
        m[1].metric("Miss, current settings", f"{fit['before']['mae']:.2f}")
        m[2].metric("Miss, fitted", f"{fit['after']['mae']:.2f}")
        m[3].metric("Correlation, fitted", f"{fit['after']['corr']:.2f}")
        cmp = pd.DataFrame({"setting": E.FIT_FIELDS,
                            "current": [float(st.session_state[f"op_{k}"]) for k in E.FIT_FIELDS],
                            "fitted": [fit["params"][k] for k in E.FIT_FIELDS]})
        st.dataframe(cmp, hide_index=True, column_config={
            "current": st.column_config.NumberColumn(format="%.2f"), "fitted": st.column_config.NumberColumn(format="%.2f")})
        b1, b2 = st.columns(2)
        b1.button("Use fitted settings", on_click=cb_apply_fit)
        b2.button("Add to calibration log", on_click=cb_log)

    hist_file = DATA / "calibration_log.csv"
    log_df = pd.concat([pd.read_csv(hist_file) if hist_file.exists() else pd.DataFrame(),
                        pd.DataFrame(st.session_state["calib_log"])], ignore_index=True)
    if len(log_df):
        st.markdown("##### Calibration log")
        st.dataframe(log_df, hide_index=True)
        st.download_button("Download calibration log (CSV)", log_df.to_csv(index=False),
                           file_name="calibration_log.csv", mime="text/csv")
        st.caption("Save the file as data/calibration_log.csv and commit it so the history survives app restarts.")
