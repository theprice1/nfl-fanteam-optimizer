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
FLEX_SHARE = {
    "RB": 0.40,
    "WR": 0.45,
    "TE": 0.15,
}  # how the FLEX slot is shared in the field

SIGMA = {
    "QB": 0.30,
    "RB": 0.50,
    "WR": 0.60,
    "TE": 0.65,
    "DST": 0.55,
}  # log-normal spread
DEFAULT_PLAY_PROB = {"expected": 0.98, "possible": 0.70, "unexpected": 0.25}
OWN_CAP = {"QB": 0.34, "RB": 0.38, "WR": 0.36, "TE": 0.40, "DST": 0.26}


@dataclass
class Rules:
    """Contest rules. Verify these against the Fanteam contest lobby each week."""

    budget: float = 110.0
    slot_min: dict = field(
        default_factory=lambda: {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
    )
    flex: bool = True  # one extra RB / WR / TE
    max_per_team: int = 4
    dst_counts_for_team: bool = True

    @property
    def roster_size(self) -> int:
        return int(sum(self.slot_min.values()) + (1 if self.flex else 0))

    def slot_max(self, pos: str) -> int:
        return int(self.slot_min[pos] + (1 if (self.flex and pos in FLEX_POS) else 0))


COLMAP = {
    "player": "name",
    "name": "name",
    "pos": "pos",
    "position": "pos",
    "team": "team",
    "opp": "opp",
    "opponent": "opp",
    "salary": "salary",
    "price": "salary",
    "cost": "salary",
    "proj": "proj",
    "projection": "proj",
    "value": "value",
    "status": "status",
    "form": "form",
    "last": "last",
    "total": "total",
    "fanteam id": "fid",
    "fanteam_id": "fid",
    "id": "fid",
}


def load_players(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a projections CSV (Daily Fantasy Tips Fanteam export format)."""
    d = df.copy()
    d.columns = [
        COLMAP.get(str(c).strip().lower(), str(c).strip().lower()) for c in d.columns
    ]
    missing = [
        c for c in ("name", "pos", "team", "salary", "proj") if c not in d.columns
    ]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    for col, default in (
        ("opp", ""),
        ("status", "expected"),
        ("form", np.nan),
        ("last", np.nan),
        ("total", np.nan),
        ("fid", ""),
    ):
        if col not in d.columns:
            d[col] = default
    d["name"] = d["name"].astype(str).str.strip()
    d["pos"] = (
        d["pos"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"DEF": "DST", "D/ST": "DST", "DS": "DST", "D": "DST"})
    )
    d["team"] = d["team"].astype(str).str.upper().str.strip()
    d["opp"] = d["opp"].astype(str).str.upper().str.strip().replace({"NAN": ""})
    for col in ("salary", "proj", "form", "last", "total"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["status"] = (
        d["status"]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace({"nan": "expected", "": "expected"})
    )
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
            raise ValueError(
                "Games CSV needs columns: away, home, total, away_implied, home_implied"
            )
        d[col] = d[col].astype(str).str.upper().str.strip()
    for col in ("total", "away_implied", "home_implied"):
        d[col] = pd.to_numeric(d.get(col, np.nan), errors="coerce")
    return d[["away", "home", "total", "away_implied", "home_implied"]].reset_index(
        drop=True
    )


def games_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no games file exists: one blank row per matchup found in the player file."""
    seen, rows = set(), []
    for t, o in players[["team", "opp"]].drop_duplicates().itertuples(index=False):
        if not o or (o, t) in seen:
            continue
        seen.add((t, o))
        rows.append(
            {
                "away": t,
                "home": o,
                "total": np.nan,
                "away_implied": np.nan,
                "home_implied": np.nan,
            }
        )
    return pd.DataFrame(
        rows, columns=["away", "home", "total", "away_implied", "home_implied"]
    )


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
def add_features(
    players: pd.DataFrame, implied: dict, play_prob: dict | None = None
) -> pd.DataFrame:
    """Adds availability, floor (20th pct), ceiling (90th pct) and team environment."""
    d = players.copy()
    pp = {**DEFAULT_PLAY_PROB, **(play_prob or {})}
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
# 3. Ownership projection + calibration
# --------------------------------------------------------------------------
@dataclass
class OwnParams:
    sharp: float = 0.75  # how concentrated the field is on the top plays
    w_proj: float = 0.90  # weight on projection rank within position
    w_value: float = 0.70  # weight on points-per-credit within position
    w_team: float = 0.35  # weight on team implied total
    pen_possible: float = 0.70
    pen_unexpected: float = 2.00


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


def _zscore(g: pd.DataFrame, base: pd.DataFrame, col: str) -> pd.Series:
    """z-score of g[col] against the reference group `base`, clipped to [-2.5, 3]."""
    mu = base[col].mean()
    sd = base[col].std(ddof=0)
    sd = 1.0 if (not np.isfinite(sd) or sd < 1e-9) else sd
    return ((g[col] - mu) / sd).clip(-2.5, 3.0)


def project_ownership(d: pd.DataFrame, params: OwnParams, rules: Rules) -> pd.Series:
    """Projected ownership in percent, summing to (slots at each position) x 100."""
    score = pd.Series(0.0, index=d.index)
    for pos, g in d.groupby("pos"):
        base = g[g["proj"] >= 0.25 * g["proj"].max()]

        pen = (
            g["status"]
            .map({"possible": params.pen_possible, "unexpected": params.pen_unexpected})
            .fillna(0.0)
        )
        score[g.index] = (
            params.w_proj * _zscore(g, base, "proj")
            + params.w_value * _zscore(g, base, "value")
            + params.w_team * g["team_z"]
            - pen
        )
    w = np.exp(params.sharp * score.clip(upper=6))
    own = pd.Series(0.0, index=d.index)
    for pos, g in d.groupby("pos"):
        total = rules.slot_min[pos] + (
            FLEX_SHARE[pos] if (rules.flex and pos in FLEX_SHARE) else 0.0
        )
        own[g.index] = _allocate(w[g.index].to_numpy(), total, OWN_CAP[pos]) * 100.0
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
        m = re.match(r"^(.*?)[,\t;|]+\s*([\d.]+)\s*%?\s*$", line) or re.match(
            r"^(.*?)\s+([\d.]+)\s*%?\s*$", line
        )
        if not m:
            continue
        try:
            rows.append((m.group(1).strip(), float(m.group(2))))
        except ValueError:
            continue
    return rows


def match_ownership(
    rows: Sequence[tuple[str, float]], d: pd.DataFrame, cutoff: float = 0.84
):
    """Fuzzy-match pasted names to the slate. Returns (matched DataFrame, unmatched names)."""
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
    out = pd.DataFrame(matched, columns=["name", "actual"]).drop_duplicates(
        "name", keep="last"
    )
    return out, unmatched


def ownership_error(pred: pd.Series, actual: pd.Series) -> dict:
    err = pred - actual
    corr = (
        float(np.corrcoef(pred, actual)[0, 1])
        if len(pred) > 2 and pred.std() > 0 and actual.std() > 0
        else float("nan")
    )
    return {
        "n": len(pred),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "corr": corr,
    }


def fit_ownership_params(
    d: pd.DataFrame, rules: Rules, actual: dict, start: OwnParams
) -> tuple[OwnParams, dict, dict]:
    """Small grid search for the parameters that best reproduce pasted post-lock ownership."""
    names = d["name"]
    mask = names.isin(actual.keys())
    act = names[mask].map(actual)
    before = ownership_error(project_ownership(d, start, rules)[mask], act)
    best, best_err = start, before["mae"]
    for sharp, wv, wt, wp in itertools.product(
        [0.7, 0.9, 1.1, 1.3, 1.6, 2.0],
        [0.2, 0.5, 0.8, 1.1],
        [0.0, 0.3, 0.6],
        [0.5, 0.9, 1.3],
    ):
        p = replace(start, sharp=sharp, w_value=wv, w_team=wt, w_proj=wp)
        e = ownership_error(project_ownership(d, p, rules)[mask], act)["mae"]
        if e < best_err - 1e-9:
            best, best_err = p, e
    after = ownership_error(project_ownership(d, best, rules)[mask], act)
    return best, before, after


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
    res = milp(
        c=-np.asarray(obj, float),
        constraints=LinearConstraint(A, lb, ub),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
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

    def __init__(
        self,
        pool: pd.DataFrame,
        rules: Rules,
        *,
        locks: Iterable[str] = (),
        qb_allowed: Iterable[str] | None = None,
        stack_min: int = 0,
        stack_rb: bool = False,
        bring_back: int = 0,
        bring_back_rb: bool = True,
        dst_vs_offense: bool = True,
        max_leftover: float | None = None,
        own_sum_cap: float | None = None,
        max_chalk: int | None = None,
        chalk_thr: float = 25.0,
        low_min: int = 0,
        low_thr: float = 10.0,
    ):
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
            add(
                (
                    {i: float(p.salary[i]) for i in range(self.n)},
                    rules.budget - max_leftover,
                    None,
                )
            )
        for pos in POS_ORDER:
            add(({i: 1 for i in ids[pos]}, rules.slot_min[pos], rules.slot_max(pos)))
        for team, g in p.groupby("team"):
            members = [
                i for i in g.index if rules.dst_counts_for_team or p.pos[i] != "DST"
            ]
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
                mates = [
                    i
                    for i in range(self.n)
                    if p.team[i] == team and p.pos[i] in stack_pos
                ]
                coefs = {i: 1 for i in mates}
                coefs[q] = -stack_min
                add((coefs, 0, None))
            if bring_back > 0 and opp:
                foes = [
                    i for i in range(self.n) if p.team[i] == opp and p.pos[i] in bb_pos
                ]
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

    def generate(
        self,
        obj: np.ndarray,
        n_lineups: int,
        *,
        rand: float = 0.0,
        min_unique: int = 1,
        max_exposure: float = 1.0,
        seed: int = 0,
        extra: Sequence = (),
        progress=None,
    ) -> list[list[int]]:
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
            ex += [
                ({i: 1 for i in L}, None, self.rules.roster_size - min_unique)
                for L in out
            ]
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


def simulate_scores(
    pool: pd.DataFrame, n_sims: int = 2000, seed: int = 7
) -> np.ndarray:
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
    shared = (
        L[:, 0:1] * pass_f[ti]
        + L[:, 1:2] * rush_f[ti]
        + L[:, 2:3] * game_f[gi]
        + L[:, 3:4] * pass_f[oi]
        + L[:, 4:5] * rush_f[oi]
    )
    noise_sd = np.sqrt(np.clip(1 - (L**2).sum(1), 0.25, None))[:, None]
    z = (shared + noise_sd * rng.standard_normal((n, n_sims))) / np.sqrt(
        (L**2).sum(1)[:, None] + noise_sd**2
    )
    sig = pool.sigma.to_numpy()[:, None]
    x = pool.proj.to_numpy()[:, None] * np.exp(sig * z - sig**2 / 2)
    plays = rng.random((n, n_sims)) < pool.p_play.to_numpy()[:, None]
    return x * plays


def sample_field(
    pool: pd.DataFrame,
    own_pct: np.ndarray,
    rules: Rules,
    n_field: int = 1500,
    seed: int = 11,
    min_spend_frac: float = 0.92,
) -> list[list[int]]:
    """Ownership-weighted random lineups that respect budget and team limits (a stand-in for the field).

    Vectorised: weighted sampling without replacement via Gumbel top-k, in batches.
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
        if len(out) >= n_field:
            break
    return out[:n_field]


def _indicator(lineups: Sequence[Sequence[int]], n: int) -> np.ndarray:
    A = np.zeros((len(lineups), n))
    for r, L in enumerate(lineups):
        A[r, list(L)] = 1.0
    return A


def evaluate_lineups(
    lineups: Sequence[Sequence[int]], X: np.ndarray, field: Sequence[Sequence[int]]
) -> pd.DataFrame:
    """Win-probability style metrics versus the simulated field, simulation by simulation."""
    n = X.shape[0]
    S = _indicator(lineups, n) @ X
    if len(field) < 50:  # field sampler failed - fall back to the lineups themselves
        F = S
    else:
        F = _indicator(field, n) @ X
    t55, t90, t99 = (np.percentile(F, q, axis=0) for q in (55, 90, 99))
    return pd.DataFrame(
        {
            "p_cash": (S >= t55).mean(1),  # beats ~55% of field (double-up style)
            "p_top10": (S >= t90).mean(1),
            "p_top1": (S >= t99).mean(1),
            "sim_mean": S.mean(1),
            "sim_p90": np.percentile(S, 90, axis=1),
        }
    )


def select_from_pool(
    pool_lineups: list[list[int]],
    metric: np.ndarray,
    n: int,
    min_unique: int,
    max_exposure: float,
    roster_size: int,
    n_players: int,
) -> list[list[int]]:
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
        "mode": mode,
        "n": 5,
        "min_unique": 2,
        "max_exposure": 1.0,
        "rand": 0.0,
        "floor_w": 0.0,
        "upside_w": 0.0,
        "lev": 0.0,
        "stack_min": 0,
        "stack_rb": False,
        "bring_back": 0,
        "dst_vs_offense": True,
        "pct_opt": 0.0,
        "use_sim": False,
        "pool_size": 60,
        "uniq_w": 0.0,
        "max_chalk": None,
        "chalk_thr": 25.0,
        "low_min": 0,
        "low_thr": 10.0,
        "own_cap_pct": None,
        "max_leftover": None,
        "locks": [],
        "qb_allowed": [],
        "n_sims": 2000,
        "n_field": 1500,
        "seed": 1,
    }
    presets = {
        "Cash": {"n": 3, "rand": 0.03, "floor_w": 0.5, "min_unique": 2},
        "GPP": {
            "n": 20,
            "rand": 0.12,
            "upside_w": 0.7,
            "lev": 0.35,
            "stack_min": 1,
            "bring_back": 1,
            "pct_opt": 0.93,
            "use_sim": True,
            "max_exposure": 0.6,
            "min_unique": 2,
        },
        "Contrarian": {
            "n": 20,
            "rand": 0.18,
            "upside_w": 0.8,
            "lev": 0.5,
            "stack_min": 1,
            "bring_back": 1,
            "pct_opt": 0.95,
            "use_sim": True,
            "uniq_w": 0.5,
            "max_exposure": 0.4,
            "min_unique": 3,
            "max_chalk": 2,
            "chalk_thr": 22.0,
            "low_min": 2,
            "low_thr": 8.0,
            "own_cap_pct": 85.0,
            "pool_size": 60,
        },
        "Pure projection": {"n": 3, "min_unique": 1},
    }
    base.update(presets[mode])
    return base


def objective_vector(pool: pd.DataFrame, cfg: dict) -> np.ndarray:
    p = pool.p_play.to_numpy()
    proj, floor, ceil = (
        pool.proj.to_numpy(),
        pool.floor.to_numpy(),
        pool.ceil.to_numpy(),
    )
    own = pool.own.to_numpy() / 100.0
    mode = cfg["mode"]
    if mode == "Cash":
        return ((1 - cfg["floor_w"]) * proj + cfg["floor_w"] * floor) * p
    if mode == "Pure projection":
        return proj * p
    return (
        ((1 - cfg["upside_w"]) * proj + cfg["upside_w"] * ceil)
        * p
        * (1 - cfg["lev"] * own)
    )


def build_lineups(
    pool: pd.DataFrame,
    rules: Rules,
    cfg: dict,
    field_baseline: float | None = None,
    progress=None,
) -> dict:
    """pool must contain: name,pos,team,opp,salary,proj,floor,ceil,p_play,sigma,own (percent)."""
    pool = pool.reset_index(drop=True)
    notes: list[str] = []
    field_l = sample_field(
        pool, pool.own.to_numpy(), rules, cfg["n_field"], seed=cfg["seed"] + 200
    )
    if (
        len(field_l) >= 50
    ):  # typical own-sum of a field lineup, measured from the simulated field
        field_baseline = float(np.mean([pool.own.to_numpy()[L].sum() for L in field_l]))
    elif field_baseline is None:
        field_baseline = float((pool.own / 100 * pool.own).sum())
    own_cap = (
        field_baseline * cfg["own_cap_pct"] / 100.0 if cfg.get("own_cap_pct") else None
    )
    opt = LineupOptimizer(
        pool,
        rules,
        locks=cfg["locks"],
        qb_allowed=cfg["qb_allowed"],
        stack_min=cfg["stack_min"],
        stack_rb=cfg["stack_rb"],
        bring_back=cfg["bring_back"],
        dst_vs_offense=cfg["dst_vs_offense"],
        max_leftover=cfg["max_leftover"],
        own_sum_cap=own_cap,
        max_chalk=cfg["max_chalk"],
        chalk_thr=cfg["chalk_thr"],
        low_min=cfg["low_min"],
        low_thr=cfg["low_thr"],
    )
    proj_adj = pool.proj.to_numpy() * pool.p_play.to_numpy()
    best = opt.solve(proj_adj)
    if best is None:
        raise ValueError(
            "No valid lineup exists with these settings. Loosen locks, stacks, ownership limits or "
            "the team cap."
        )
    extra = []
    if cfg["pct_opt"] > 0:
        floor_val = cfg["pct_opt"] * float(proj_adj[best].sum())
        extra.append(
            ({i: float(proj_adj[i]) for i in range(len(pool))}, floor_val, None)
        )

    obj = objective_vector(pool, cfg)
    n = int(cfg["n"])
    M = max(n, int(cfg["pool_size"])) if cfg["use_sim"] else n
    cand = opt.generate(
        obj,
        M,
        rand=cfg["rand"],
        min_unique=1 if cfg["use_sim"] else cfg["min_unique"],
        max_exposure=min(1.0, cfg["max_exposure"] + 0.15)
        if cfg["use_sim"]
        else cfg["max_exposure"],
        seed=cfg["seed"],
        extra=extra,
        progress=progress,
    )
    if not cand:
        raise ValueError("The optimizer found no lineups. Loosen the settings.")

    X = simulate_scores(pool, cfg["n_sims"], seed=cfg["seed"] + 100)
    if cfg["use_sim"]:
        ev = evaluate_lineups(cand, X, field_l)
        own_sum = np.array([pool.own.to_numpy()[L].sum() for L in cand])
        metric = (
            ev["p_top1"].to_numpy()
            * (field_baseline / np.clip(own_sum, 1, None)) ** cfg["uniq_w"]
        )
        metric = (
            metric + 0.01 * ev["p_top10"].to_numpy()
        )  # tie-break toward higher top-10% odds
        chosen = select_from_pool(
            cand,
            metric,
            n,
            cfg["min_unique"],
            cfg["max_exposure"],
            rules.roster_size,
            len(pool),
        )
        notes.append(
            f"Ranked {len(cand)} candidate lineups by simulated top-1% odds against a "
            f"{len(field_l)}-lineup ownership-weighted field."
        )
    else:
        chosen = cand
    if len(chosen) < n:
        notes.append(
            f"Only {len(chosen)} of {n} lineups met the uniqueness / exposure limits. Raise the candidate "
            f"pool, variety or max exposure to get more."
        )
    ev = evaluate_lineups(chosen, X, field_l)
    stats = pd.DataFrame([lineup_metrics(pool, L, rules.budget) for L in chosen]).join(
        ev
    )
    stats.insert(0, "lineup", range(1, len(stats) + 1))
    if len(field_l) < 50:
        notes.append(
            "Field simulation had too few valid lineups; win odds are approximate."
        )
    return {
        "pool": pool,
        "lineups": chosen,
        "stats": stats,
        "notes": notes,
        "field_baseline": field_baseline,
    }


def order_slots(
    pool: pd.DataFrame, idxs: Sequence[int], rules: Rules
) -> list[tuple[str, int]]:
    """Names each player's slot: QB, RB1.., WR1.., TE, FLEX, DST."""
    out: list[tuple[str, int]] = []
    extra: list[int] = []
    for pos in POS_ORDER:
        members = sorted(
            [i for i in idxs if pool.pos[i] == pos], key=lambda i: -pool.proj[i]
        )
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
        label = f"{q.team} QB" + (
            "".join("+" + p for p in mates.pos) if len(mates) else " solo"
        )
        if len(foes):
            label += f" | BB {q.opp} " + "+".join(foes.pos)
    return {
        "salary": float(sub.salary.sum()),
        "left": float(budget - sub.salary.sum()),
        "proj": float(sub.proj.sum()),
        "proj_adj": float((sub.proj * sub.p_play).sum()),
        "floor": float(sub.floor.sum()),
        "ceil": float(sub.ceil.sum()),
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
