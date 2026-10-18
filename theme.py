"""theme.py - NFL look and feel for the Streamlit app (CSS + small HTML helpers)."""
from __future__ import annotations

import html

import streamlit as st

# Accent colours picked to stay legible on a dark navy background.
TEAM_COLORS = {
    "ARI": "#C8385A", "ATL": "#D3293F", "BAL": "#6F4FD6", "BUF": "#2E6BE6", "CAR": "#1796D6", "CHI": "#E8672B",
    "CIN": "#FB4F14", "CLE": "#FF6A2B", "DAL": "#4F7BD9", "DEN": "#FB6B24", "DET": "#2D9BE0", "GB": "#FFB612",
    "HOU": "#C8334B", "IND": "#4F8FE0", "JAX": "#1E9AA8", "KC": "#E8344E", "LAC": "#2AA3E8", "LAR": "#F2B21B",
    "LV": "#A7AEB2", "MIA": "#13B5BC", "MIN": "#8A5BE0", "NE": "#D9304A", "NO": "#D3BC8D", "NYG": "#4D78E0",
    "NYJ": "#2BAE72", "PHI": "#2BB59A", "PIT": "#FFB612", "SF": "#D32A2A", "SEA": "#6FC12C", "TB": "#E0342B",
    "TEN": "#59A5E8", "WAS": "#F2B21B",
}
SLOT_COLORS = {"QB": "#E4002B", "RB": "#1FA3FF", "WR": "#29C46B", "TE": "#F5A524", "FLEX": "#B26BFF", "DST": "#8D9BB5"}


def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#7C8AA5")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700;800&display=swap');

:root {
  --navy: #0A1226; --navy-2: #101B33; --navy-3: #16233F; --line: #25355A;
  --shield: #013369; --red: #E4002B; --chalk: #EAF0FA; --muted: #93A3C2; --turf: #1E8E5A;
}
html, body, [class*="css"], .stApp { font-family: 'Barlow', 'Segoe UI', sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 10% -10%, #14305F 0%, rgba(20,48,95,0) 60%), var(--navy); }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; letter-spacing: .2px; font-weight: 700 !important; }
section[data-testid="stSidebar"] { background: var(--navy-2); border-right: 1px solid var(--line); }
.block-container { padding-top: 1.4rem; max-width: 1280px; }

/* Hero: scoreboard strip + yard-line ticks */
.hero { position: relative; border: 1px solid var(--line); border-left: 6px solid var(--red); border-radius: 6px;
  padding: 18px 22px 30px; margin-bottom: 14px;
  background: linear-gradient(100deg, #0F2A57 0%, #0B1B3A 55%, #0A1226 100%); overflow: hidden; }
.hero h1 { font-size: 2.6rem; line-height: 1; margin: 0; color: #fff; font-weight: 800 !important; }
.hero p { margin: 6px 0 0; color: var(--muted); font-size: 1.02rem; }
.hero .ticks { position: absolute; left: 0; right: 0; bottom: 0; height: 18px;
  background: repeating-linear-gradient(90deg, rgba(234,240,250,.28) 0 2px, transparent 2px 38px);
  mask-image: linear-gradient(0deg, #000 40%, transparent); -webkit-mask-image: linear-gradient(0deg, #000 40%, transparent); }
.board { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 0; }
.game { background: rgba(255,255,255,.05); border: 1px solid var(--line); border-radius: 4px; padding: 5px 10px;
  font-family: 'Barlow Condensed', sans-serif; font-size: 1.05rem; color: var(--chalk); }
.game b { color: #fff; font-weight: 700; }
.game .tot { color: #FFD166; margin-left: 8px; font-weight: 600; }

/* Tabs */
button[data-baseweb="tab"] { font-family: 'Barlow Condensed', sans-serif; font-size: 1.15rem; font-weight: 600; }
div[data-baseweb="tab-highlight"] { background-color: var(--red) !important; }

/* Metrics */
[data-testid="stMetric"] { background: var(--navy-2); border: 1px solid var(--line); border-radius: 6px; padding: 10px 14px; }
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; }

/* Buttons */
.stButton > button[kind="primary"], .stDownloadButton > button { border-radius: 4px; font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem; font-weight: 700; letter-spacing: .3px; }

/* Lineup card */
.lc { background: var(--navy-2); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 16px; overflow: hidden; }
.lc-head { display: flex; flex-wrap: wrap; gap: 6px 22px; align-items: baseline; padding: 10px 14px;
  background: linear-gradient(90deg, #12305F, #101B33); border-bottom: 1px solid var(--line); }
.lc-title { font-family: 'Barlow Condensed', sans-serif; font-size: 1.35rem; font-weight: 800; color: #fff; }
.lc-stat { color: var(--muted); font-size: .92rem; } .lc-stat b { color: var(--chalk); font-weight: 600; }
.lc-stack { margin-left: auto; color: #FFD166; font-family: 'Barlow Condensed', sans-serif; font-size: 1.05rem; }
.row { display: grid; grid-template-columns: 52px 1fr 52px 62px 56px 56px; align-items: center; gap: 8px;
  padding: 6px 14px 6px 0; border-top: 1px solid rgba(37,53,90,.6); border-left: 5px solid var(--tc, #7C8AA5); }
.row.hd { color: var(--muted); font-size: .8rem; border-top: none; padding-top: 7px; padding-bottom: 3px; }
.slot { text-align: center; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: .95rem;
  color: #fff; border-radius: 3px; padding: 1px 0; margin-left: 8px; background: var(--sc, #556); }
.nm { color: #fff; font-weight: 600; } .nm small { color: var(--muted); font-weight: 400; margin-left: 6px; }
.nm .flag { color: #FFB020; font-weight: 700; margin-left: 6px; }
.num { text-align: right; font-variant-numeric: tabular-nums; color: var(--chalk); }
.tm { text-align: center; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; color: var(--tc); }
.hi { color: #FFD166; }
@media (max-width: 700px) {
  .hero h1 { font-size: 2rem; }
  .row { grid-template-columns: 44px 1fr 44px 52px 48px 48px; gap: 4px; font-size: .88rem; }
  .lc-stack { margin-left: 0; width: 100%; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, games: list[tuple[str, str, float | None]]) -> None:
    chips = []
    for away, home, total in games:
        tot = f'<span class="tot">{total:g}</span>' if total is not None else '<span class="tot" style="color:#93A3C2">TBD</span>'
        chips.append(f'<div class="game"><b>{html.escape(away)}</b> @ <b>{html.escape(home)}</b>{tot}</div>')
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
        f'<div class="board">{"".join(chips)}</div><div class="ticks"></div></div>',
        unsafe_allow_html=True,
    )


def lineup_card(n: int, slots: list[tuple[str, dict]], stats: dict) -> str:
    """slots = [(slot_name, player_row_dict)]; stats = lineup stats row (dict)."""
    rows = ['<div class="row hd"><div></div><div>Player</div><div style="text-align:center">Team</div>'
            '<div class="num">Salary</div><div class="num">Proj</div><div class="num">Own%</div></div>']
    for slot, p in slots:
        base = "".join(c for c in slot if c.isalpha())
        flag = '<span class="flag" title="Not a confirmed starter">?</span>' if p["status"] != "expected" else ""
        own_cls = "num hi" if p["own"] < 8 else "num"
        rows.append(
            f'<div class="row" style="--tc:{team_color(p["team"])}">'
            f'<div class="slot" style="--sc:{SLOT_COLORS.get(base, "#556")}">{html.escape(slot)}</div>'
            f'<div class="nm">{html.escape(p["name"])}{flag}<small>vs {html.escape(p["opp"])}</small></div>'
            f'<div class="tm">{html.escape(p["team"])}</div>'
            f'<div class="num">{p["salary"]:.1f}</div><div class="num">{p["proj"]:.1f}</div>'
            f'<div class="{own_cls}">{p["own"]:.0f}%</div></div>'
        )
    head = (
        f'<div class="lc-head"><span class="lc-title">Lineup {n}</span>'
        f'<span class="lc-stat">Proj <b>{stats["proj"]:.1f}</b></span>'
        f'<span class="lc-stat">Spend <b>{stats["salary"]:.1f}</b> (left {stats["left"]:.1f})</span>'
        f'<span class="lc-stat">Own sum <b>{stats["own_sum"]:.0f}%</b></span>'
        f'<span class="lc-stat">Top 1% <b>{stats["p_top1"] * 100:.1f}%</b></span>'
        f'<span class="lc-stat">Cash <b>{stats["p_cash"] * 100:.0f}%</b></span>'
        f'<span class="lc-stack">{html.escape(stats["stack"])}</span></div>'
    )
    return f'<div class="lc">{head}{"".join(rows)}</div>'
