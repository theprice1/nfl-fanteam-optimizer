"""
engine.py - Fanteam NFL optimizer core (no Streamlit imports, fully testable).

Sections
  1. Rules + data loading
  2. Player model (availability, floor / ceiling)
  3. Ownership projection (+ post-lock calibration)
  4. Lineup optimizer (PuLP, with a SciPy fallback)
  5. Correlated simulation + field model (ranks lineups for GPP)
  6. Lineup helpers (slot ordering, metrics, export)
"""
from __future__ import annotations

import difflib
import io
import itertools
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Rules + data loading
# --------------------------------------------------------------------------
POS_ORDER = ["QB", "RB", "WR", "TE", "DST"]
FLEX_POS = ("RB", "WR", "TE")
FLEX_SHARE = {"RB": 0.515, "WR": 0.313, "TE": 0.172}  # how the FLEX slot is shared (Week 6 results)

SIGMA = {"QB": 0.31, "RB": 0.55, "WR": 0.65, "TE": 0.75, "DST": 0.70}  # log-normal spread (Week 6 residuals)
DEFAULT_PLAY_PROB = {"expected": 0.98, "possible": 0.70, "unexpected": 0.25}
OWN_CAP = {"QB": 0.28, "RB": 0.40, "WR": 0.45, "TE": 0.45, "DST": 0.22}


@dataclass
class Rules:
    """Contest rules. Verify these against the Fanteam contest lobby each week."""

    budget: float = 110.0
    slot_min: dict = field(default_factory=lambda: {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1})
    flex: bool = True  # one extra RB / WR / TE
    max_per_team: int = 3
    dst_counts_for_team: bool = True

    @property
    def roster_size(self) -> int:
        return int(sum(self.slot_min.values()) + (1 if self.flex else 0))

    def slot_max(self, pos: str) -> int:
        return int(self.slot_min[pos] + (1 if (self.flex and pos in FLEX_POS) else 0))


COLMAP = {
    "player": "name", "name": "name", "pos": "pos", "position": "pos", "team": "team",
    "opp": "opp", "opponent": "opp", "salary": "salary", "price": "salary", "cost": "salary",
    "proj": "proj", "projection": "proj", "value": "value", "status": "status",
    "form": "form", "last": "last", "total": "total",
    "fanteam id": "fid", "fanteam_id": "fid", "id": "fid",
}


def load_players(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a projections CSV (Daily Fantasy Tips Fanteam export format)."""
    d = df.copy()
    d.columns = [COLMAP.get(str(c).strip().lower(), str(c).strip().lower()) for c in d.columns]
    missing = [c for c in ("name", "pos", "team", "salary", "proj") if c not in d.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    for col, default in (("opp", ""), ("status", "expected"), ("form", np.nan),
                         ("last", np.nan), ("total", np.nan), ("fid", "")):
        if col not in d.columns:
            d[col] = default
    d["name"] = d["name"].astype(str).str.strip()
    d["pos"] = (d["pos"].astype(str).str.upper().str.strip()
                .replace({"DEF": "DST", "D/ST": "DST", "DS": "DST", "D": "DST"}))
    d["team"] = d["team"].astype(str).str.upper().str.strip()
    d["opp"] = d["opp"].astype(str).str.upper().str.strip().replace({"NAN": ""})
    for col in ("salary", "proj", "form", "last", "total"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["status"] = (d["status"].astype(str).str.lower().str.strip()
                   .replace({"nan": "expected", "": "expected"}))
    d = d[d["pos"].isin(POS_ORDER)].dropna(subset=["salary", "proj"])
    d = d[(d["salary"] > 0) & (d["proj"] > 0)]
    d = d.drop_duplicates(["name", "pos", "team"]).reset_index(drop=True)
    d["value"] = d["proj"] / d["salary"]
    d["fid"] = d["fid"].astype(str).str.replace(r"\.0$", "", regex=True)
    return d


def load_games(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    for col in ("away", "home"):
        if col not in d.columns:
            raise ValueError("Games CSV needs columns: away, home, total, away_implied, home_implied")
        d[col] = d[col].astype(str).str.upper().str.strip()
    for col in ("total", "away_implied", "home_implied"):
        d[col] = pd.to_numeric(d.get(col, np.nan), errors="coerce")
    return d[["away", "home", "total", "away_implied", "home_implied"]].reset_index(drop=True)


def games_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no games file exists: one blank row per matchup found in the player file."""
    seen, rows = set(), []
    for t, o in players[["team", "opp"]].drop_duplicates().itertuples(index=False):
        if not o or (o, t) in seen:
            continue
        seen.add((t, o))
        rows.append({"away": t, "home": o, "total": np.nan, "away_implied": np.nan, "home_implied": np.nan})
    return pd.DataFrame(rows, columns=["away", "home", "total", "away_implied", "home_implied"])


def implied_map(games: pd.DataFrame) -> dict:
    out = {}
    for r in games.itertuples(index=False):
        if pd.notna(r.away_implied):
            out[r.away] = float(r.away_implied)
        if pd.notna(r.home_implied):
            out[r.home] = float(r.home_implied)
    return out


# --------------------------------------------------------------------------
# 2. Player model
# --------------------------------------------------------------------------
def add_features(players: pd.DataFrame, implied: dict, play_prob: dict | None = None) -> pd.DataFrame:
    """Adds availability, floor (20th pct), ceiling (90th pct) and team environment."""
    d = players.copy()
    pp = {**DEFAULT_PLAY_PROB, **(play_prob or {})}
    if "proj_src" not in d.columns:
        d["proj_src"] = "source"
    d["p_play"] = d["status"].map(pp).fillna(0.9)
    d["sigma"] = d["pos"].map(SIGMA)
    # Mean-preserving log-normal: E[x] = proj. Right-skewed, never negative.
    d["floor"] = d["proj"] * np.exp(-0.8416 * d["sigma"] - d["sigma"] ** 2 / 2)
    d["ceil"] = d["proj"] * np.exp(1.2816 * d["sigma"] - d["sigma"] ** 2 / 2)
    known = [v for v in implied.values() if not np.isnan(v)]
    mean_imp = float(np.mean(known)) if known else 22.0
    d["implied"] = d["team"].map(implied).fillna(mean_imp)
    per_team = d.drop_duplicates("team")["implied"]
    sd = float(per_team.std(ddof=0)) if len(per_team) > 1 else 0.0
    d["team_z"] = (d["implied"] - float(per_team.mean())) / (sd if sd > 1e-9 else 1.0)
    return d


# --------------------------------------------------------------------------
# 3. Ownership projection + calibration (fitted on real Fanteam results)
# --------------------------------------------------------------------------
@dataclass
class OwnParams:
    """Multinomial-logit ownership model, fitted on Week 6 Fanteam results (190 players).

    score = tau[pos] * sharp * (b_proj*z_proj + b_value*z_value + b_form*z_form
                                + b_price*z_price + b_team*team_z) - penalties
    Shares are a softmax of score within each position, scaled to that position's slot count.
    """

    sharp: float = 1.0        # global multiplier on every coefficient
    b_proj: float = 0.49      # projection, z-scored within position
    b_value: float = 0.48     # points per credit
    b_form: float = 0.30      # recent Fanteam form (public chases hot players)
    b_price: float = -0.24    # price (cheaper = more owned once value is controlled for)
    b_team: float = 0.08      # team implied total
    pen_possible: float = 0.07
    pen_unexpected: float = 2.0
    tau_qb: float = 0.85      # quarterbacks are spread out
    tau_rb: float = 1.25      # running backs are concentrated
    tau_wr: float = 1.00
    tau_te: float = 1.00
    tau_dst: float = 0.85


FIT_FIELDS = ["b_proj", "b_value", "b_form", "b_price", "b_team", "pen_possible",
              "tau_qb", "tau_rb", "tau_wr", "tau_te", "tau_dst"]
FEATURES = ["proj", "value", "form", "price", "team_z"]


def _allocate(w: np.ndarray, total: float, cap: float) -> np.ndarray:
    """Scale weights to sum to `total`, capping any player at `cap`, redistributing the excess."""
    w = np.asarray(w, float)
    own = np.zeros(len(w))
    free = np.ones(len(w), bool)
    remaining = total
    for _ in range(80):
        s = w[free].sum()
        if s <= 0 or remaining <= 1e-9:
            break
        idx = np.where(free)[0]
        trial = w[idx] / s * remaining
        over = trial > cap
        if not over.any():
            own[idx] = trial
            break
        own[idx[over]] = cap
        free[idx[over]] = False
        remaining -= cap * over.sum()
    return own


def design_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """z-scored features within each position (reference group = players with a real projection)."""
    X = pd.DataFrame(0.0, index=d.index, columns=FEATURES)
    form = d["form"] if "form" in d.columns else pd.Series(np.nan, index=d.index)
    for pos, g in d.groupby("pos"):
        cols = {
            "proj": g["proj"], "value": g["proj"] / g["salary"], "price": g["salary"],
            "form": form[g.index].fillna(form[g.index].median() if form[g.index].notna().any() else 0.0),
        }
        base_idx = g.index[g["proj"] >= 0.25 * g["proj"].max()]
        for name, s in cols.items():
            mu, sd = s[base_idx].mean(), s[base_idx].std(ddof=0)
            sd = 1.0 if (not np.isfinite(sd) or sd < 1e-9) else sd
            X.loc[g.index, name] = ((s - mu) / sd).clip(-2.5, 3.0)
        X.loc[g.index, "team_z"] = g["team_z"] if "team_z" in g.columns else 0.0
    return X


def _scores(d: pd.DataFrame, params: OwnParams) -> pd.Series:
    X = design_matrix(d)
    lin = (params.b_proj * X["proj"] + params.b_value * X["value"] + params.b_form * X["form"]
           + params.b_price * X["price"] + params.b_team * X["team_z"]) * params.sharp
    pen = d["status"].map({"possible": params.pen_possible, "unexpected": params.pen_unexpected}).fillna(0.0)
    tau = d["pos"].map({"QB": params.tau_qb, "RB": params.tau_rb, "WR": params.tau_wr,
                        "TE": params.tau_te, "DST": params.tau_dst})
    return lin * tau - pen


def project_ownership(d: pd.DataFrame, params: OwnParams, rules: Rules) -> pd.Series:
    """Projected ownership in percent, summing to (slots at each position) x 100."""
    score = _scores(d, params).clip(upper=8)
    own = pd.Series(0.0, index=d.index)
    for pos, g in d.groupby("pos"):
        total = rules.slot_min[pos] + (FLEX_SHARE[pos] if (rules.flex and pos in FLEX_SHARE) else 0.0)
        w = np.exp(score[g.index] - score[g.index].max())
        own[g.index] = _allocate(w.to_numpy(), total, OWN_CAP[pos]) * 100.0
    return own.round(2)


def _norm_name(s: str) -> str:
    s = s.lower().replace(".", "").replace("'", "").replace("’", "")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(re.sub(r"[^a-z ]", " ", s).split())


def parse_ownership_text(text: str) -> list[tuple[str, float]]:
    """Accepts lines like 'Josh Allen, 23.4%', 'Josh Allen<TAB>23.4', 'Josh Allen 23.4%'."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.*?)[,\t;|]+\s*([\d.]+)\s*%?\s*$", line) or re.match(r"^(.*?)\s+([\d.]+)\s*%?\s*$", line)
        if not m:
            continue
        try:
            rows.append((m.group(1).strip(), float(m.group(2))))
        except ValueError:
            continue
    return rows


PLAYER_LIST_COLS = ["name", "pos", "team", "price", "own_pct", "form", "score", "fid"]

_FT_ROW = re.compile(
    r"^\s*(?P<name>.+?)\s+(?P<pos>QB|RB|WR|TE|DST)\s+(?P<team>[A-Z]{2,3})\s+(?P<price>\d+(?:\.\d+)?)\s*M\s*"
    r"(?:(?P<own>\d+(?:\.\d+)?)\s*%|N/?A)?\s*(?:(?P<form>-?\d+(?:\.\d+)?)|N/?A)?\s*(?P<score>-?\d+(?:\.\d+)?)?\s*$",
    re.IGNORECASE)


def parse_fanteam_table(text: str) -> pd.DataFrame:
    """Parse text copied from the Fanteam player table (Player, Pos, Team, Price, Own %, Form, Score).

    Works before lock (no ownership or score yet) and after lock. Blank / N/A values become NaN.
    """
    rows = []
    for line in text.replace("\t", "  ").splitlines():
        m = _FT_ROW.match(line)
        if not m:
            continue
        g = m.groupdict()
        pos = g["pos"].upper()
        rows.append({
            "name": g["name"].strip(), "pos": pos, "team": g["team"].upper(), "price": float(g["price"]),
            "own_pct": float(g["own"]) if g["own"] else np.nan,
            "form": float(g["form"]) if g["form"] else np.nan,
            "score": float(g["score"]) if g["score"] else np.nan,
            "fid": "",
        })
    return pd.DataFrame(rows, columns=PLAYER_LIST_COLS)


_LIST_COLS = {
    "name": ("player", "name", "player name"), "pos": ("pos", "position"), "team": ("team", "club"),
    "price": ("price", "salary", "cost"), "own_pct": ("own %", "own%", "own", "ownership", "owned"),
    "form": ("form",), "score": ("score", "points", "pts"),
    "fid": ("fanteam id", "fanteam_id", "player id", "playerid", "id"),
}


def parse_player_list(data: bytes, filename: str = "") -> pd.DataFrame:
    """Read an uploaded Fanteam player list: a CSV with headers, or text copied from the Fanteam table."""
    text = data.decode("utf-8-sig", errors="replace")
    try:
        raw = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
        raw = pd.DataFrame()
    lower = {str(c).strip().lower(): c for c in raw.columns}
    pick = {k: next((lower[a] for a in alts if a in lower), None) for k, alts in _LIST_COLS.items()}
    if len(raw) and pick["name"] and pick["pos"] and pick["team"] and pick["price"]:
        def num(col):
            if col is None:
                return pd.Series(np.nan, index=raw.index)
            return pd.to_numeric(raw[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

        out = pd.DataFrame({
            "name": raw[pick["name"]].astype(str).str.strip(),
            "pos": (raw[pick["pos"]].astype(str).str.upper().str.strip()
                    .replace({"DEF": "DST", "D/ST": "DST", "DS": "DST", "D": "DST"})),
            "team": raw[pick["team"]].astype(str).str.upper().str.strip(),
            "price": num(pick["price"]), "own_pct": num(pick["own_pct"]),
            "form": num(pick["form"]), "score": num(pick["score"]),
            "fid": (raw[pick["fid"]].astype(str).str.replace(r"\.0$", "", regex=True) if pick["fid"]
                    else pd.Series("", index=raw.index)),
        })
        return out[out["pos"].isin(POS_ORDER) & out["price"].notna()].reset_index(drop=True)[PLAYER_LIST_COLS]
    return parse_fanteam_table(text)


def _match_row(d: pd.DataFrame, name: str, pos: str, team: str, cutoff: float = 0.7):
    cand = d[(d["pos"] == pos) & (d["team"] == team)]
    if cand.empty:
        return None
    if pos == "DST":
        return cand.index[0]
    keys = {_norm_name(n): i for i, n in zip(cand.index, cand["name"])}
    k = _norm_name(name)
    if k in keys:
        return keys[k]
    close = difflib.get_close_matches(k, list(keys), n=1, cutoff=cutoff)
    return keys[close[0]] if close else None


def match_ownership(rows: Sequence[tuple[str, float]], d: pd.DataFrame, cutoff: float = 0.84):
    """Fuzzy-match pasted (name, %) pairs to the slate. Returns (matched DataFrame, unmatched names)."""
    keys = {_norm_name(n): n for n in d["name"]}
    matched, unmatched = [], []
    for raw, pct in rows:
        k = _norm_name(raw)
        hit = keys.get(k)
        if hit is None:
            close = difflib.get_close_matches(k, list(keys), n=1, cutoff=cutoff)
            hit = keys[close[0]] if close else None
        if hit is None:
            unmatched.append(raw)
        else:
            matched.append({"name": hit, "actual": pct})
    out = pd.DataFrame(matched, columns=["name", "actual"]).drop_duplicates("name", keep="last")
    return out, unmatched


def fallback_projection_model(d: pd.DataFrame):
    """Least-squares proj ~ price + form for players the source has no projection for."""
    ok = d[d["form"].notna()] if "form" in d.columns else d
    med = float(ok["form"].median()) if len(ok) else 0.0
    X = np.c_[np.ones(len(ok)), ok["salary"], ok["form"].fillna(med)]
    beta = np.linalg.lstsq(X, ok["proj"], rcond=None)[0]

    def predict(price, form):
        f = pd.Series(form).fillna(med).to_numpy()
        return np.clip(beta[0] + beta[1] * np.asarray(price, float) + beta[2] * f, 1.0, None)

    return predict


def add_unprojected(d: pd.DataFrame, price_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add players that appear in a pasted Fanteam list but not in the projections file.

    Their projection is estimated from price and form, status 'expected', flagged proj_src='fallback'.
    """
    d = d.copy()
    if "proj_src" not in d.columns:
        d["proj_src"] = "source"
    predict = fallback_projection_model(d)
    opp_of = d.drop_duplicates("team").set_index("team")["opp"].to_dict()
    rows, added = [], []
    for r in price_df.itertuples(index=False):
        if _match_row(d, r.name, r.pos, r.team) is not None:
            continue
        proj = float(predict([r.price], [r.form])[0])
        if r.pos == "DST":
            continue
        rows.append({"name": r.name, "pos": r.pos, "team": r.team, "opp": opp_of.get(r.team, ""), "salary": r.price,
                     "proj": proj, "status": "expected", "form": r.form, "last": np.nan, "total": np.nan,
                     "fid": str(getattr(r, "fid", "") or ""), "value": proj / r.price, "proj_src": "fallback"})
        added.append(r.name)
    if rows:
        d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
    return d, added


def build_training_frame(d: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per player in a results table (Fanteam post-lock ownership + score), joined to projections.

    Players missing from the projections get a price/form-based projection. Players who did not
    play are not in the results table, so they are left out here rather than counted as 0%.
    """
    predict = fallback_projection_model(d)
    team_z = d.drop_duplicates("team").set_index("team")["team_z"].to_dict() if "team_z" in d.columns else {}
    rows = []
    for r in results.itertuples(index=False):
        hit = _match_row(d, r.name, r.pos, r.team)
        if hit is not None:
            proj, status, src = float(d.at[hit, "proj"]), d.at[hit, "status"], "source"
        else:
            proj, status, src = float(predict([r.price], [r.form])[0]), "expected", "fallback"
        rows.append({"name": r.name, "pos": r.pos, "team": r.team, "salary": r.price, "proj": proj,
                     "form": r.form, "status": status, "team_z": team_z.get(r.team, 0.0),
                     "actual": 0.0 if pd.isna(r.own_pct) else r.own_pct, "score": r.score, "proj_src": src})
    return pd.DataFrame(rows)


def load_history(folder) -> list[pd.DataFrame]:
    """Training files saved from earlier weeks (data/history/*_training.csv)."""
    from pathlib import Path

    frames = []
    for f in sorted(Path(folder).glob("*_training.csv")):
        t = pd.read_csv(f)
        if {"pos", "salary", "proj", "actual"} <= set(t.columns):
            t["week_file"] = f.stem
            frames.append(t)
    return frames


def ownership_error(pred: pd.Series, actual: pd.Series) -> dict:
    err = pred - actual
    corr = float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 2 and pred.std() > 0 and actual.std() > 0 else float("nan")
    return {"n": len(pred), "mae": float(err.abs().mean()), "bias": float(err.mean()), "corr": corr}


def evaluate_ownership(frames: Sequence[pd.DataFrame], params: OwnParams, rules: Rules) -> dict:
    pred = pd.concat([project_ownership(f, params, rules) for f in frames], ignore_index=True)
    act = pd.concat([f["actual"] for f in frames], ignore_index=True)
    return ownership_error(pred, act)


def fit_ownership(frames: Sequence[pd.DataFrame], rules: Rules, prior: OwnParams | None = None,
                  prior_weight: float = 0.02) -> tuple[OwnParams, dict, dict]:
    """Fit ownership coefficients on one or more weeks of real results.

    Maximises the multinomial-logit likelihood of each position's ownership shares, with a pull
    toward `prior` (prior_weight / number of weeks) so a single odd week cannot swing the model.
    Returns (fitted params, error before, error after).
    """
    from scipy.optimize import minimize

    prior = prior or OwnParams()
    theta0 = np.array([getattr(prior, k) for k in FIT_FIELDS], float)
    prepared = []
    for f in frames:
        f = f.reset_index(drop=True)
        X = design_matrix(f)
        prepared.append((f, X, (f["status"] == "possible").astype(float).to_numpy()))
    total_own = sum(float(f["actual"].sum()) for f in frames) or 1.0
    lam = prior_weight / max(len(frames), 1)

    def nll(theta: np.ndarray) -> float:
        b = dict(zip(FIT_FIELDS, theta))
        loss = 0.0
        for f, X, poss in prepared:
            lin = (b["b_proj"] * X["proj"] + b["b_value"] * X["value"] + b["b_form"] * X["form"]
                   + b["b_price"] * X["price"] + b["b_team"] * X["team_z"]).to_numpy()
            tau = f["pos"].map({"QB": b["tau_qb"], "RB": b["tau_rb"], "WR": b["tau_wr"],
                                "TE": b["tau_te"], "DST": b["tau_dst"]}).to_numpy()
            s = lin * tau - b["pen_possible"] * poss
            for pos in POS_ORDER:
                idx = np.where(f["pos"].to_numpy() == pos)[0]
                if len(idx) == 0:
                    continue
                sc = s[idx]
                lse = np.log(np.exp(sc - sc.max()).sum()) + sc.max()
                loss -= float(np.sum(f["actual"].to_numpy()[idx] * (sc - lse)))
        return loss / total_own + lam * float(np.sum((theta - theta0) ** 2))

    res = minimize(nll, theta0, method="L-BFGS-B",
                   bounds=[(-0.5, 2.0)] * 3 + [(-1.5, 0.5)] + [(-0.5, 1.0)] + [(-0.5, 1.5)] + [(0.3, 2.5)] * 5)
    fitted = replace(prior, **dict(zip(FIT_FIELDS, [float(x) for x in res.x])))
    return fitted, evaluate_ownership(frames, prior, rules), evaluate_ownership(frames, fitted, rules)


# --------------------------------------------------------------------------
# 4. Lineup optimizer
# --------------------------------------------------------------------------
def _solve_pulp(obj: np.ndarray, cons: list, n: int) -> list[int] | None:
    import pulp

    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)]
    prob += pulp.lpSum(float(obj[i]) * x[i] for i in range(n))
    for coefs, lo, hi in cons:
        expr = pulp.lpSum(float(c) * x[i] for i, c in coefs.items())
        if lo is not None:
            prob += expr >= lo
        if hi is not None:
            prob += expr <= hi
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return [i for i in range(n) if x[i].value() is not None and x[i].value() > 0.5]


def _solve_scipy(obj: np.ndarray, cons: list, n: int) -> list[int] | None:
    from scipy.optimize import Bounds, LinearConstraint, milp

    A = np.zeros((len(cons), n))
    lb = np.full(len(cons), -np.inf)
    ub = np.full(len(cons), np.inf)
    for r, (coefs, lo, hi) in enumerate(cons):
        for i, c in coefs.items():
            A[r, i] = c
        if lo is not None:
            lb[r] = lo
        if hi is not None:
            ub[r] = hi
    res = milp(c=-np.asarray(obj, float), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success or res.x is None:
        return None
    return [i for i in range(n) if res.x[i] > 0.5]


_PULP_BROKEN = False


def _solve_binary(obj: np.ndarray, cons: list, n: int) -> list[int] | None:
    """Maximise obj.x s.t. cons, x binary. cons = [(coef_dict, lo, hi)].

    Uses PuLP (CBC). If the installed PuLP version has an incompatible API or is missing,
    falls back to SciPy's MILP solver (HiGHS), which gives the same answers.
    """
    global _PULP_BROKEN
    if not _PULP_BROKEN:
        try:
            return _solve_pulp(obj, cons, n)
        except (ImportError, TypeError, AttributeError):
            _PULP_BROKEN = True
    return _solve_scipy(obj, cons, n)


class LineupOptimizer:
    """Holds the structural constraints for one build; re-solved with different objectives."""

    def __init__(self, pool: pd.DataFrame, rules: Rules, *, locks: Iterable[str] = (),
                 qb_allowed: Iterable[str] | None = None, stack_min: int = 0, stack_rb: bool = False,
                 bring_back: int = 0, bring_back_rb: bool = True, dst_vs_offense: bool = True,
                 max_leftover: float | None = None, own_sum_cap: float | None = None,
                 max_chalk: int | None = None, chalk_thr: float = 25.0,
                 low_min: int = 0, low_thr: float = 10.0):
        self.pool = pool.reset_index(drop=True)
        self.rules = rules
        self.n = len(self.pool)
        p = self.pool
        self.cons: list = []
        add = self.cons.append
        ids = {pos: [i for i in range(self.n) if p.pos[i] == pos] for pos in POS_ORDER}

        add(({i: 1 for i in range(self.n)}, rules.roster_size, rules.roster_size))
        add(({i: float(p.salary[i]) for i in range(self.n)}, None, rules.budget))
        if max_leftover is not None:
            add(({i: float(p.salary[i]) for i in range(self.n)}, rules.budget - max_leftover, None))
        for pos in POS_ORDER:
            add(({i: 1 for i in ids[pos]}, rules.slot_min[pos], rules.slot_max(pos)))
        for team, g in p.groupby("team"):
            members = [i for i in g.index if rules.dst_counts_for_team or p.pos[i] != "DST"]
            if len(members) > rules.max_per_team:
                add(({i: 1 for i in members}, None, rules.max_per_team))

        names = {nm: i for i, nm in enumerate(p.name)}
        for nm in locks:
            if nm in names:
                add(({names[nm]: 1}, 1, 1))
        if qb_allowed:
            allowed = [i for i in ids["QB"] if p.name[i] in set(qb_allowed)]
            add(({i: 1 for i in allowed}, 1, 1))

        stack_pos = ("WR", "TE") + (("RB",) if stack_rb else ())
        bb_pos = ("WR", "TE") + (("RB",) if bring_back_rb else ())
        for q in ids["QB"]:
            team, opp = p.team[q], p.opp[q]
            if stack_min > 0:
                mates = [i for i in range(self.n) if p.team[i] == team and p.pos[i] in stack_pos]
                coefs = {i: 1 for i in mates}
                coefs[q] = -stack_min
                add((coefs, 0, None))
            if bring_back > 0 and opp:
                foes = [i for i in range(self.n) if p.team[i] == opp and p.pos[i] in bb_pos]
                coefs = {i: 1 for i in foes}
                coefs[q] = -bring_back
                add((coefs, 0, None))
        if dst_vs_offense:
            for d_i in ids["DST"]:
                for i in range(self.n):
                    if p.pos[i] != "DST" and p.team[i] == p.opp[d_i] and p.opp[d_i]:
                        add(({d_i: 1, i: 1}, None, 1))

        own = p["own"].to_numpy() if "own" in p.columns else np.zeros(self.n)
        if own_sum_cap is not None:
            add(({i: float(own[i]) for i in range(self.n)}, None, own_sum_cap))
        if max_chalk is not None:
            add(({i: 1 for i in range(self.n) if own[i] > chalk_thr}, None, max_chalk))
        if low_min > 0:
            add(({i: 1 for i in range(self.n) if own[i] < low_thr}, low_min, None))

    def solve(self, obj: np.ndarray, extra: Sequence = ()) -> list[int] | None:
        return _solve_binary(np.asarray(obj, float), self.cons + list(extra), self.n)

    def generate(self, obj: np.ndarray, n_lineups: int, *, rand: float = 0.0, min_unique: int = 1,
                 max_exposure: float = 1.0, seed: int = 0, extra: Sequence = (), progress=None) -> list[list[int]]:
        rng = np.random.default_rng(seed)
        out: list[list[int]] = []
        counts = np.zeros(self.n)
        cap = max(1, math.ceil(max_exposure * n_lineups))
        for k in range(n_lineups):
            if progress:
                progress(k / n_lineups)
            o = obj * (1 + rng.uniform(-rand, rand, self.n)) if rand > 0 else obj
            ex = list(extra)
            ex += [({i: 1}, None, 0) for i in np.where(counts >= cap)[0]]
            ex += [({i: 1 for i in L}, None, self.rules.roster_size - min_unique) for L in out]
            sol = self.solve(o, ex)
            if sol is None:
                break
            out.append(sol)
            counts[sol] += 1
        return out


# --------------------------------------------------------------------------
# 5. Simulation + field model
# --------------------------------------------------------------------------
# loadings: (own pass env, own rush env, game env, opp pass env, opp rush env)
LOADINGS = {
    "QB": (0.55, 0.05, 0.25, 0.0, 0.0),
    "WR": (0.45, 0.00, 0.20, 0.0, 0.0),
    "TE": (0.40, 0.00, 0.20, 0.0, 0.0),
    "RB": (0.15, 0.35, 0.15, 0.0, 0.0),
    "DST": (0.0, 0.0, -0.15, -0.30, -0.15),
}


def simulate_scores(pool: pd.DataFrame, n_sims: int = 2000, seed: int = 7) -> np.ndarray:
    """(players x sims) correlated fantasy scores: shared team passing/rushing and game environments."""
    rng = np.random.default_rng(seed)
    teams = sorted(set(pool.team) | (set(pool.opp) - {""}))
    t_idx = {t: k for k, t in enumerate(teams)}
    games = {}
    for t, o in zip(pool.team, pool.opp):
        games.setdefault(frozenset((t, o)), len(games))
    pass_f = rng.standard_normal((len(teams), n_sims))
    rush_f = rng.standard_normal((len(teams), n_sims))
    game_f = rng.standard_normal((len(games), n_sims))
    n = len(pool)
    L = np.array([LOADINGS[p] for p in pool.pos])
    ti = np.array([t_idx[t] for t in pool.team])
    oi = np.array([t_idx.get(o, t_idx[t]) for t, o in zip(pool.team, pool.opp)])
    gi = np.array([games[frozenset((t, o))] for t, o in zip(pool.team, pool.opp)])
    shared = (L[:, 0:1] * pass_f[ti] + L[:, 1:2] * rush_f[ti] + L[:, 2:3] * game_f[gi]
              + L[:, 3:4] * pass_f[oi] + L[:, 4:5] * rush_f[oi])
    noise_sd = np.sqrt(np.clip(1 - (L ** 2).sum(1), 0.25, None))[:, None]
    z = (shared + noise_sd * rng.standard_normal((n, n_sims))) / np.sqrt((L ** 2).sum(1)[:, None] + noise_sd ** 2)
    sig = pool.sigma.to_numpy()[:, None]
    x = pool.proj.to_numpy()[:, None] * np.exp(sig * z - sig ** 2 / 2)
    plays = rng.random((n, n_sims)) < pool.p_play.to_numpy()[:, None]
    return x * plays


def sample_field(pool: pd.DataFrame, own_pct: np.ndarray, rules: Rules, n_field: int = 1500,
                 seed: int = 11, min_spend_frac: float = 0.92, keep: float = 1.0) -> list[list[int]]:
    """Ownership-weighted random lineups that respect budget and team limits (a stand-in for the field).

    Vectorised: weighted sampling without replacement via Gumbel top-k, in batches. `keep` < 1 keeps only
    that share of the sampled lineups, the ones with the highest total projection, because real entrants
    optimise rather than pick at random.
    """
    rng = np.random.default_rng(seed)
    frac = np.clip(np.asarray(own_pct, float), 0.01, None) / 100.0
    logw = np.log(frac)
    pos = pool.pos.to_numpy()
    sal = pool.salary.to_numpy()
    codes = pd.factorize(pool.team)[0]
    dst_mask = pos == "DST"
    groups = {p: np.where(pos == p)[0] for p in POS_ORDER}
    if any(len(groups[p]) < rules.slot_min[p] for p in POS_ORDER):
        return []
    flex_ids = np.concatenate([groups[p] for p in FLEX_POS])
    out: list[list[int]] = []
    for _ in range(60):
        B = max(4000, n_field * 3)
        cols = []
        for p in POS_ORDER:
            k = rules.slot_min[p]
            if k == 0:
                continue
            ids = groups[p]
            keys = logw[ids][None, :] + rng.gumbel(size=(B, len(ids)))
            top = np.argpartition(-keys, k - 1, axis=1)[:, :k]
            cols.append(ids[top])
        chosen = np.concatenate(cols, axis=1)
        if rules.flex:
            keys = logw[flex_ids][None, :] + rng.gumbel(size=(B, len(flex_ids)))
            taken = (flex_ids[None, :, None] == chosen[:, None, :]).any(2)
            keys[taken] = -np.inf
            chosen = np.concatenate([chosen, flex_ids[keys.argmax(1)][:, None]], axis=1)
        spend = sal[chosen].sum(1)
        ok = (spend <= rules.budget) & (spend >= rules.budget * min_spend_frac)
        tc = codes[chosen]
        counted = np.where(dst_mask[chosen] & (not rules.dst_counts_for_team), -1, tc)
        worst = np.zeros(B, int)
        for t in np.unique(tc):
            worst = np.maximum(worst, (counted == t).sum(1))
        ok &= worst <= rules.max_per_team
        out.extend(chosen[ok].tolist())
        if len(out) >= n_field / max(keep, 0.05):
            break
    if keep < 1.0 and out:
        proj = pool.proj.to_numpy()
        out.sort(key=lambda L: -proj[L].sum())
        return out[: max(int(len(out) * keep), 1)][:n_field]
    return out[:n_field]


def _indicator(lineups: Sequence[Sequence[int]], n: int) -> np.ndarray:
    A = np.zeros((len(lineups), n))
    for r, L in enumerate(lineups):
        A[r, list(L)] = 1.0
    return A


LADDERS = {
    # (top fraction of field, relative payout) - normalised so the field returns (1 - rake) of entry fees
    "Top-heavy (large field)": [(0.001, 150), (0.005, 50), (0.01, 25), (0.02, 12), (0.05, 5), (0.10, 2.5), (0.20, 1.5)],
    "Balanced (mid-size field)": [(0.01, 40), (0.03, 15), (0.07, 6), (0.15, 3), (0.25, 1.8)],
    "Flat (small field)": [(0.05, 8), (0.15, 3), (0.30, 1.8)],
}


def payout_tiers(kind: str, rake: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """(top fractions, payout as a multiple of the entry fee). A stand-in for the real prize structure."""
    ladder = LADDERS[kind]
    fr = np.array([f for f, _ in ladder], float)
    m = np.array([x for _, x in ladder], float)
    scale = (1 - rake) / float((np.diff(np.r_[0.0, fr]) * m).sum())
    return fr, m * scale


def evaluate_lineups(lineups: Sequence[Sequence[int]], X: np.ndarray, field: Sequence[Sequence[int]],
                     ladder: tuple[np.ndarray, np.ndarray] | None = None) -> pd.DataFrame:
    """Win-probability style metrics versus the simulated field, simulation by simulation.

    With a payout ladder it also returns `ev`: expected return as a multiple of the entry fee.
    """
    n = X.shape[0]
    S = _indicator(lineups, n) @ X
    if len(field) < 50:  # field sampler failed - fall back to the lineups themselves
        F = S
    else:
        F = _indicator(field, n) @ X
    t55, t90, t99 = (np.percentile(F, q, axis=0) for q in (55, 90, 99))
    out = pd.DataFrame({
        "p_cash": (S >= t55).mean(1),   # beats ~55% of field (double-up style)
        "p_top10": (S >= t90).mean(1),
        "p_top1": (S >= t99).mean(1),
        "sim_mean": S.mean(1),
        "sim_p90": np.percentile(S, 90, axis=1),
    })
    if ladder is not None:
        fr, mult = ladder
        pay = np.zeros_like(S)
        for j in reversed(range(len(fr))):  # loosest tier first so stricter tiers overwrite it
            pay = np.where(S >= np.percentile(F, 100 * (1 - fr[j]), axis=0)[None, :], mult[j], pay)
        out["ev"] = pay.mean(1)
    return out


def select_from_pool(pool_lineups: list[list[int]], metric: np.ndarray, n: int, min_unique: int,
                     max_exposure: float, roster_size: int, n_players: int) -> list[list[int]]:
    order = np.argsort(-metric)
    cap = max(1, math.ceil(max_exposure * n))
    counts = np.zeros(n_players)
    chosen: list[list[int]] = []
    for j in order:
        L = pool_lineups[j]
        if (counts[L] >= cap).any():
            continue
        if any(len(set(L) & set(C)) > roster_size - min_unique for C in chosen):
            continue
        chosen.append(L)
        counts[L] += 1
        if len(chosen) >= n:
            break
    return chosen


# --------------------------------------------------------------------------
# 6. Build orchestration + lineup helpers
# --------------------------------------------------------------------------
def default_cfg(mode: str) -> dict:
    base = {
        "mode": mode, "n": 5, "min_unique": 2, "max_exposure": 1.0, "rand": 0.0, "floor_w": 0.0, "upside_w": 0.0,
        "lev": 0.0, "stack_min": 0, "stack_rb": False, "bring_back": 0, "dst_vs_offense": True, "pct_opt": 0.0,
        "use_sim": False, "pool_size": 60, "uniq_w": 0.0, "max_chalk": None, "chalk_thr": 25.0, "low_min": 0,
        "low_thr": 10.0, "own_cap_pct": None, "max_leftover": None, "locks": [], "qb_allowed": [],
        "n_sims": 2000, "n_field": 1500, "seed": 1, "candidates": 120, "max_qbs": 8, "qb_max_exposure": 0.4,
        "ladder": "Top-heavy (large field)",
        "field_keep": 1.0, "risky_max_exposure": 0.35,
    }
    presets = {
        "Cash": {"n": 3, "rand": 0.03, "floor_w": 0.5, "min_unique": 2},
        "GPP": {"n": 20, "rand": 0.12, "upside_w": 0.7, "lev": 0.35, "stack_min": 1, "bring_back": 1,
                "pct_opt": 0.93, "use_sim": True, "max_exposure": 0.6, "min_unique": 2},
        "Contrarian": {"n": 20, "rand": 0.18, "upside_w": 0.8, "lev": 0.5, "stack_min": 1, "bring_back": 1,
                       "pct_opt": 0.95, "use_sim": True, "uniq_w": 0.5, "max_exposure": 0.4, "min_unique": 3,
                       "max_chalk": 2, "chalk_thr": 22.0, "low_min": 2, "low_thr": 8.0, "own_cap_pct": 85.0,
                       "pool_size": 60},
        "Pure projection": {"n": 3, "min_unique": 1},
        "Auto": {"n": 20, "rand": 0.10, "pct_opt": 0.90, "uniq_w": 0.35, "max_exposure": 0.5, "min_unique": 3,
                 "use_sim": True, "n_sims": 3000, "n_field": 2000, "candidates": 120, "max_qbs": 8,
                 "qb_max_exposure": 0.4, "ladder": "Top-heavy (large field)", "field_keep": 0.5,
                 "risky_max_exposure": 0.35},
    }
    base.update(presets[mode])
    return base


def objective_vector(pool: pd.DataFrame, cfg: dict) -> np.ndarray:
    p = pool.p_play.to_numpy()
    proj, floor, ceil = pool.proj.to_numpy(), pool.floor.to_numpy(), pool.ceil.to_numpy()
    own = pool.own.to_numpy() / 100.0
    mode = cfg["mode"]
    if mode == "Cash":
        return ((1 - cfg["floor_w"]) * proj + cfg["floor_w"] * floor) * p
    if mode == "Pure projection":
        return proj * p
    return ((1 - cfg["upside_w"]) * proj + cfg["upside_w"] * ceil) * p * (1 - cfg["lev"] * own)


def build_lineups(pool: pd.DataFrame, rules: Rules, cfg: dict, field_baseline: float | None = None,
                  progress=None) -> dict:
    """pool must contain: name,pos,team,opp,salary,proj,floor,ceil,p_play,sigma,own (percent)."""
    if cfg["mode"] == "Auto":
        return auto_portfolio(pool, rules, cfg, progress)
    pool = pool.reset_index(drop=True)
    notes: list[str] = []
    field_l = sample_field(pool, pool.own.to_numpy(), rules, cfg["n_field"], seed=cfg["seed"] + 200)
    if len(field_l) >= 50:  # typical own-sum of a field lineup, measured from the simulated field
        field_baseline = float(np.mean([pool.own.to_numpy()[L].sum() for L in field_l]))
    elif field_baseline is None:
        field_baseline = float((pool.own / 100 * pool.own).sum())
    own_cap = field_baseline * cfg["own_cap_pct"] / 100.0 if cfg.get("own_cap_pct") else None
    opt = LineupOptimizer(
        pool, rules, locks=cfg["locks"], qb_allowed=cfg["qb_allowed"], stack_min=cfg["stack_min"],
        stack_rb=cfg["stack_rb"], bring_back=cfg["bring_back"], dst_vs_offense=cfg["dst_vs_offense"],
        max_leftover=cfg["max_leftover"], own_sum_cap=own_cap, max_chalk=cfg["max_chalk"],
        chalk_thr=cfg["chalk_thr"], low_min=cfg["low_min"], low_thr=cfg["low_thr"])
    proj_adj = pool.proj.to_numpy() * pool.p_play.to_numpy()
    best = opt.solve(proj_adj)
    if best is None:
        raise ValueError("No valid lineup exists with these settings. Loosen locks, stacks, ownership limits or "
                         "the team cap.")
    extra = []
    if cfg["pct_opt"] > 0:
        floor_val = cfg["pct_opt"] * float(proj_adj[best].sum())
        extra.append(({i: float(proj_adj[i]) for i in range(len(pool))}, floor_val, None))

    obj = objective_vector(pool, cfg)
    n = int(cfg["n"])
    M = max(n, int(cfg["pool_size"])) if cfg["use_sim"] else n
    cand = opt.generate(obj, M, rand=cfg["rand"], min_unique=1 if cfg["use_sim"] else cfg["min_unique"],
                        max_exposure=min(1.0, cfg["max_exposure"] + 0.15) if cfg["use_sim"] else cfg["max_exposure"],
                        seed=cfg["seed"], extra=extra,
                        progress=progress)
    if not cand:
        raise ValueError("The optimizer found no lineups. Loosen the settings.")

    X = simulate_scores(pool, cfg["n_sims"], seed=cfg["seed"] + 100)
    if cfg["use_sim"]:
        ev = evaluate_lineups(cand, X, field_l)
        own_sum = np.array([pool.own.to_numpy()[L].sum() for L in cand])
        metric = ev["p_top1"].to_numpy() * (field_baseline / np.clip(own_sum, 1, None)) ** cfg["uniq_w"]
        metric = metric + 0.01 * ev["p_top10"].to_numpy()  # tie-break toward higher top-10% odds
        chosen = select_from_pool(cand, metric, n, cfg["min_unique"], cfg["max_exposure"], rules.roster_size, len(pool))
        notes.append(f"Ranked {len(cand)} candidate lineups by simulated top-1% odds against a "
                     f"{len(field_l)}-lineup ownership-weighted field.")
    else:
        chosen = cand
    if len(chosen) < n:
        notes.append(f"Only {len(chosen)} of {n} lineups met the uniqueness / exposure limits. Raise the candidate "
                     f"pool, variety or max exposure to get more.")
    ev = evaluate_lineups(chosen, X, field_l)
    stats = pd.DataFrame([lineup_metrics(pool, L, rules.budget) for L in chosen]).join(ev)
    stats.insert(0, "lineup", range(1, len(stats) + 1))
    if len(field_l) < 50:
        notes.append("Field simulation had too few valid lineups; win odds are approximate.")
    return {"pool": pool, "lineups": chosen, "stats": stats, "notes": notes, "field_baseline": field_baseline}


def order_slots(pool: pd.DataFrame, idxs: Sequence[int], rules: Rules) -> list[tuple[str, int]]:
    """Names each player's slot: QB, RB1.., WR1.., TE, FLEX, DST."""
    out: list[tuple[str, int]] = []
    extra: list[int] = []
    for pos in POS_ORDER:
        members = sorted([i for i in idxs if pool.pos[i] == pos], key=lambda i: -pool.proj[i])
        k = rules.slot_min[pos]
        for j, i in enumerate(members[:k]):
            out.append((pos if k == 1 else f"{pos}{j + 1}", i))
        extra += members[k:]
    if extra:
        out.insert(len(out) - 1, ("FLEX", extra[0]))  # keep DST last
    return out


def lineup_metrics(pool: pd.DataFrame, idxs: Sequence[int], budget: float) -> dict:
    sub = pool.loc[list(idxs)]
    qb = sub[sub.pos == "QB"]
    label = "no QB"
    if len(qb):
        q = qb.iloc[0]
        mates = sub[(sub.team == q.team) & (~sub.pos.isin(["QB", "DST"]))]
        foes = sub[(sub.opp == q.team) & (sub.pos != "DST")]
        label = f"{q.team} QB" + ("".join("+" + p for p in mates.pos) if len(mates) else " solo")
        if len(foes):
            label += f" | BB {q.opp} " + "+".join(foes.pos)
    return {
        "salary": float(sub.salary.sum()), "left": float(budget - sub.salary.sum()),
        "proj": float(sub.proj.sum()), "proj_adj": float((sub.proj * sub.p_play).sum()),
        "floor": float(sub.floor.sum()), "ceil": float(sub.ceil.sum()),
        "own_sum": float(sub.own.sum()) if "own" in sub else 0.0,
        "own_avg": float(sub.own.mean()) if "own" in sub else 0.0,
        "chalk": int((sub.own > 25).sum()) if "own" in sub else 0,
        "stack": label,
        "risky": int((sub.status != "expected").sum()),
        "idx": list(idxs),
    }


def lineups_to_frame(result: dict, rules: Rules) -> pd.DataFrame:
    """Wide export: one row per lineup with names + Fanteam IDs in slot order."""
    pool = result["pool"]
    rows = []
    for k, L in enumerate(result["lineups"], 1):
        row = {"lineup": k}
        for slot, i in order_slots(pool, L, rules):
            row[slot] = pool.name[i]
            row[f"{slot}_id"] = pool.fid[i]
        rows.append(row)
    return pd.DataFrame(rows)


def exposure_table(result: dict) -> pd.DataFrame:
    pool, lineups = result["pool"], result["lineups"]
    counts = pd.Series(0, index=pool.index, dtype=float)
    for L in lineups:
        counts[L] += 1
    t = pool[["name", "pos", "team", "salary", "proj", "own"]].copy()
    t["exposure"] = counts / max(len(lineups), 1) * 100
    t = t[t.exposure > 0].sort_values(["exposure", "proj"], ascending=False)
    t["leverage"] = t["exposure"] - t["own"]
    return t.reset_index(drop=True)


# --------------------------------------------------------------------------
# 7. Auto tournament portfolio
# --------------------------------------------------------------------------
def auto_candidates(pool: pd.DataFrame, rules: Rules, cfg: dict, progress=None) -> list[list[int]]:
    """Diversified candidate lineups: every viable QB x stack shape x leverage x upside setting, noised."""
    rng = np.random.default_rng(cfg["seed"])
    n = len(pool)
    p = pool.p_play.to_numpy()
    proj, ceil_ = pool.proj.to_numpy(), pool.ceil.to_numpy()
    own = pool.own.to_numpy() / 100.0
    proj_adj = proj * p
    qb_all = np.where(pool.pos.to_numpy() == "QB")[0]
    qb_ok = [i for i in qb_all if p[i] >= 0.5 and proj[i] >= 0.65 * proj[qb_all].max()]
    qb_ok = sorted(qb_ok, key=lambda i: -proj_adj[i])[: int(cfg["max_qbs"])]
    if cfg.get("qb_allowed"):
        keep = [i for i in qb_all if pool.name[i] in set(cfg["qb_allowed"])]
        qb_ok = keep or qb_ok
    variants = [(1, 0), (1, 1), (2, 1)]
    opts = {v: LineupOptimizer(pool, rules, locks=cfg["locks"], stack_min=v[0], bring_back=v[1],
                               dst_vs_offense=cfg["dst_vs_offense"], max_leftover=cfg["max_leftover"])
            for v in variants}
    best = opts[(1, 1)].solve(proj_adj) or opts[(1, 0)].solve(proj_adj)
    if best is None:
        raise ValueError("No valid lineup exists with these settings. Loosen locks or the team cap.")
    floor_val = cfg["pct_opt"] * float(proj_adj[best].sum())
    floor_con = ({i: float(proj_adj[i]) for i in range(n)}, floor_val, None)
    combos = [(v, lev, up) for v in variants for lev in (0.25, 0.55, 0.85) for up in (0.55, 0.8)]
    per_qb = []
    for q in qb_ok:
        order = list(rng.permutation(len(combos)))
        per_qb.append([(q, *combos[k]) for k in order])
    archetypes = [a for tier in zip(*per_qb) for a in tier] if per_qb else []
    seen: set = set()
    cands: list[list[int]] = []
    target = int(cfg["candidates"])
    per_arch = 2
    for step, (q, v, lev, up) in enumerate(archetypes):
        if len(cands) >= target:
            break
        if progress:
            progress(min(len(cands) / target, 1.0))
        obj = ((1 - up) * proj + up * ceil_) * p * (1 - lev * own) * (1 + rng.uniform(-cfg["rand"], cfg["rand"], n))
        found: list[list[int]] = []
        rotate: list[int] = []
        if len(cands) >= 15 and rng.random() < 0.6:  # rotate the core: ban a couple of the most-used players
            usage = np.bincount(np.concatenate(cands), minlength=n)
            top = [int(i) for i in np.argsort(-usage)[:6] if i != q]
            rotate = [int(i) for i in rng.choice(top, size=min(2, len(top)), replace=False)]
        for _ in range(per_arch):
            extra = ([({q: 1}, 1, 1), floor_con] + [({i: 1}, None, 0) for i in rotate]
                     + [({i: 1 for i in L}, None, rules.roster_size - 2) for L in found])
            sol = opts[v].solve(obj, extra)
            if sol is None:
                break
            found.append(sol)
            key = frozenset(sol)
            if key not in seen:
                seen.add(key)
                cands.append(sol)
    if not cands:
        raise ValueError("The optimizer found no lineups. Loosen the settings.")
    return cands


def select_portfolio(cands: list[list[int]], value: np.ndarray, qb_of: np.ndarray, n: int, min_unique: int,
                     max_exposure: float, qb_max_exposure: float, roster_size: int, n_players: int,
                     risky: np.ndarray | None = None, risky_max_exposure: float = 1.0):
    """Greedy: best value first, subject to player / QB / uncertain-player exposure caps and uniqueness."""
    order = np.argsort(-value)
    risky = np.zeros(n_players, bool) if risky is None else risky
    for relax in (1.0, 1.25, 1.5):
        cap = max(1, math.ceil(max_exposure * relax * n))
        rcap = max(1, math.ceil(min(risky_max_exposure, max_exposure) * relax * n))
        caps = np.where(risky, rcap, cap)
        qcap = max(1, math.ceil(qb_max_exposure * relax * n))
        counts, qcounts = np.zeros(n_players), {}
        chosen: list[int] = []
        for j in order:
            L = cands[j]
            if (counts[L] >= caps[L]).any() or qcounts.get(qb_of[j], 0) >= qcap:
                continue
            if any(len(set(L) & set(cands[c])) > roster_size - min_unique for c in chosen):
                continue
            chosen.append(int(j))
            counts[L] += 1
            qcounts[qb_of[j]] = qcounts.get(qb_of[j], 0) + 1
            if len(chosen) >= n:
                break
        if len(chosen) >= n or relax == 1.5:
            return chosen, relax
    return chosen, 1.5


def auto_portfolio(pool: pd.DataFrame, rules: Rules, cfg: dict, progress=None) -> dict:
    """One-click tournament build: diversified candidates -> simulated payout vs the field -> exposure-capped portfolio."""
    pool = pool.reset_index(drop=True)
    notes: list[str] = []
    own = pool.own.to_numpy()
    field_l = sample_field(pool, own, rules, cfg["n_field"], seed=cfg["seed"] + 200, keep=cfg["field_keep"])
    if len(field_l) >= 50:
        baseline = float(np.mean([own[L].sum() for L in field_l]))
    else:
        baseline = float((own / 100 * own).sum())
        notes.append("Field simulation had too few valid lineups; odds are approximate.")
    cands = auto_candidates(pool, rules, cfg, progress)
    X = simulate_scores(pool, cfg["n_sims"], seed=cfg["seed"] + 100)
    ladder = payout_tiers(cfg["ladder"])
    ev = evaluate_lineups(cands, X, field_l, ladder)
    own_sum = np.array([own[L].sum() for L in cands])
    value = ev["ev"].to_numpy() * (baseline / np.clip(own_sum, 1, None)) ** cfg["uniq_w"]
    qb_of = np.array([next(i for i in L if pool.pos[i] == "QB") for L in cands])
    n = int(cfg["n"])
    picked, relax = select_portfolio(cands, value, qb_of, n, cfg["min_unique"], cfg["max_exposure"],
                                     cfg["qb_max_exposure"], rules.roster_size, len(pool),
                                     risky=(pool.status != "expected").to_numpy(),
                                     risky_max_exposure=cfg["risky_max_exposure"])
    chosen = [cands[j] for j in picked]
    if relax > 1.0:
        notes.append(f"Exposure caps were relaxed by {int((relax - 1) * 100)}% to fill {len(chosen)} lineups.")
    if len(chosen) < n:
        notes.append(f"Only {len(chosen)} of {n} lineups met the uniqueness / exposure limits. Raise the candidate "
                     f"count or the exposure caps for more.")
    ev_c = evaluate_lineups(chosen, X, field_l, ladder)
    stats = pd.DataFrame([lineup_metrics(pool, L, rules.budget) for L in chosen]).join(ev_c)
    stats["adj_ev"] = value[picked]
    stats.insert(0, "lineup", range(1, len(stats) + 1))
    S = _indicator(chosen, len(pool)) @ X
    F = _indicator(field_l, len(pool)) @ X if len(field_l) >= 50 else S
    portfolio = {
        "expected_return": float(ev_c["ev"].mean()),
        "p_any_top1": float((S >= np.percentile(F, 99, axis=0)).any(0).mean()),
        "p_any_top10": float((S >= np.percentile(F, 90, axis=0)).any(0).mean()),
        "candidates": len(cands), "field_baseline": baseline, "ladder": cfg["ladder"],
    }
    notes.append(f"Scored {len(cands)} candidate lineups across {len(set(qb_of))} quarterbacks against a "
                 f"{len(field_l)}-lineup ownership-weighted field, using a '{cfg['ladder']}' payout curve.")
    return {"pool": pool, "lineups": chosen, "stats": stats, "notes": notes, "field_baseline": baseline,
            "portfolio": portfolio}


# --------------------------------------------------------------------------
# 8. Fanteam lineup upload file
# --------------------------------------------------------------------------
def upload_csv_text(rows: Sequence[Sequence]) -> str:
    """Fanteam upload layout: no header, one lineup per line, LF line endings, no trailing newline."""
    return "\n".join(",".join(str(v) for v in r) for r in rows)


def fanteam_upload(result: dict, rules: Rules, contest_id: str) -> tuple[str, list[str]]:
    """(csv text, names missing a Fanteam ID). Row = contest id, then player ids in slot order."""
    pool = result["pool"]
    rows, missing = [], []
    for L in result["lineups"]:
        row = [contest_id]
        for _, i in order_slots(pool, L, rules):
            fid = str(pool.at[i, "fid"]).strip()
            if not fid or fid.lower() == "nan":
                missing.append(str(pool.at[i, "name"]))
            row.append(fid)
        rows.append(row)
    return upload_csv_text(rows), sorted(set(missing))
