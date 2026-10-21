"""
scraper.py - pull the weekly NFL projections from Daily Fantasy Tips (https://dailyfantasy.tips/nfl/).

The page is server-rendered: every site's projections are plain HTML tables, so no browser or
JavaScript is needed. This module uses only the standard library, finds tables by their column
names (not their position), and returns the same data the page's own "Download CSV" button gives.

Command line (run from the project folder):
    python scraper.py                      fetch the live page, write data/nfl_fanteam_weekNN.csv + weekNN_games.csv
    python scraper.py --debug              list every table found (use this if the page layout changes)
    python scraper.py --html saved.html    parse a page you saved from your browser instead of fetching

Be a polite visitor: fetch once or twice a week, not in a loop. Check the site's terms and
robots.txt (https://dailyfantasy.tips/robots.txt) before automating, and keep this for personal use.
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://dailyfantasy.tips/nfl/"
USER_AGENT = "Mozilla/5.0 (compatible; GridironLab personal optimizer; weekly fetch)"
SITES = ("draftkings", "fanduel", "yahoo", "fanteam")
REQUIRED = {"player", "pos", "team", "salary", "proj"}


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    headings: list[str] = field(default_factory=list)  # nearest preceding h1-h4 texts, newest last


class _TableParser(HTMLParser):
    """Collects every <table> with its header cells, body rows and the headings above it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[Table] = []
        self.headings: list[str] = []
        self.page_text: list[str] = []
        self._tag_stack: list[str] = []
        self._heading_buf: list[str] | None = None
        self._table: Table | None = None
        self._row: list[str] | None = None
        self._row_has_th = False
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        self._tag_stack.append(tag)
        if tag in ("h1", "h2", "h3", "h4"):
            self._heading_buf = []
        elif tag == "table":
            self._table = Table(headers=[], rows=[], headings=self.headings[-4:])
        elif tag == "tr" and self._table is not None:
            self._row, self._row_has_th = [], False
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._row_has_th = self._row_has_th or tag == "th"

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4") and self._heading_buf is not None:
            text = " ".join("".join(self._heading_buf).split())
            if text:
                self.headings.append(text)
            self._heading_buf = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row_has_th and not self._table.headers:
                self._table.headers = self._row
            elif self._row:
                self._table.rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data):
        if self._heading_buf is not None:
            self._heading_buf.append(data)
        if self._cell is not None:
            self._cell.append(data)
        self.page_text.append(data)


def fetch_html(url: str = URL, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return resp.read().decode("utf-8", errors="replace")


def extract_tables(html: str) -> tuple[list[Table], str]:
    p = _TableParser()
    p.feed(html)
    return p.tables, " ".join(" ".join(p.page_text).split())


def _num(x) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else np.nan


def _site_of(t: Table) -> str | None:
    heads = " ".join(h.lower() for h in t.headings)
    for s in SITES:
        if s in heads:
            return s
    cols = " ".join(t.headers).lower()
    for s, key in (("draftkings", "dk id"), ("fanduel", "fd id"), ("yahoo", "yahoo id"), ("fanteam", "fanteam id")):
        if key in cols:
            return s
    return None


def find_projection_table(tables: list[Table], site: str = "fanteam") -> Table:
    """The 'All projections' table for a site: has Player/Pos/Team/Salary/Proj columns."""
    best, best_n = None, -1
    for t in tables:
        cols = {h.strip().lower() for h in t.headers}
        if REQUIRED <= cols and _site_of(t) == site and len(t.rows) > best_n:
            best, best_n = t, len(t.rows)
    if best is None:
        raise ValueError(
            f"Could not find the {site} projections table. Run `python scraper.py --debug` and send me the "
            "list of tables, or save the page from your browser and use --html.")
    return best


def table_to_frame(t: Table) -> pd.DataFrame:
    width = len(t.headers)
    rows = [r[:width] + [""] * (width - len(r)) for r in t.rows]
    df = pd.DataFrame(rows, columns=[h.strip() for h in t.headers])
    for col in df.columns:
        low = col.lower()
        if low in ("salary", "proj", "value", "form", "last", "total"):
            df[col] = df[col].map(_num)
    return df


def find_games_table(tables: list[Table]) -> Table | None:
    for t in tables:
        cols = [h.lower() for h in t.headers]
        if "game" in cols and any("implied" in c for c in cols):
            return t
    return None


def games_to_frame(t: Table | None) -> pd.DataFrame:
    """away, home, total, away_implied, home_implied (implied is blank until the line is posted)."""
    cols = ["away", "home", "total", "away_implied", "home_implied"]
    if t is None:
        return pd.DataFrame(columns=cols)
    idx = {h.strip().lower(): i for i, h in enumerate(t.headers)}
    out = []
    for r in t.rows:
        m = re.match(r"^\s*([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\s*$", r[idx["game"]]) if "game" in idx else None
        if not m:
            continue
        get = lambda key: _num(r[idx[key]]) if key in idx and idx[key] < len(r) else np.nan  # noqa: E731
        out.append({"away": m.group(1), "home": m.group(2), "total": get("total"),
                    "away_implied": get("away implied"), "home_implied": get("home implied")})
    return pd.DataFrame(out, columns=cols)


def find_week(page_text: str) -> tuple[int | None, str]:
    wk = re.search(r"WEEK\s+(\d{1,2})", page_text, flags=re.IGNORECASE)
    upd = re.search(r"Last updated:\s*([\d/]+,\s*[\d:]+)", page_text)
    return (int(wk.group(1)) if wk else None, upd.group(1) if upd else "")


def scrape(html: str | None = None, site: str = "fanteam") -> dict:
    """Return {'week', 'updated', 'players' (raw CSV columns), 'games', 'tables'}."""
    html = html if html is not None else fetch_html()
    tables, text = extract_tables(html)
    week, updated = find_week(text)
    return {
        "week": week, "updated": updated, "tables": tables,
        "players": table_to_frame(find_projection_table(tables, site)),
        "games": games_to_frame(find_games_table(tables)),
    }


def save(result: dict, folder: str | Path = "data") -> tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    wk = f"{result['week']:02d}" if result["week"] else "xx"
    p_path, g_path = folder / f"nfl_fanteam_week{wk}.csv", folder / f"week{wk}_games.csv"
    result["players"].to_csv(p_path, index=False)
    result["games"].to_csv(g_path, index=False)
    return p_path, g_path


def describe_tables(tables: list[Table]) -> str:
    lines = []
    for i, t in enumerate(tables):
        lines.append(f"[{i}] rows={len(t.rows):4d} site={_site_of(t)} headings={t.headings[-2:]} columns={t.headers}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch NFL projections from Daily Fantasy Tips")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--html", help="parse a saved HTML file instead of fetching")
    ap.add_argument("--out", default="data", help="folder for the CSV files")
    ap.add_argument("--debug", action="store_true", help="print every table found and exit")
    args = ap.parse_args()
    html = Path(args.html).read_text(encoding="utf-8", errors="replace") if args.html else fetch_html(args.url)
    if args.debug:
        tables, _ = extract_tables(html)
        print(describe_tables(tables))
        return
    result = scrape(html)
    p_path, g_path = save(result, args.out)
    print(f"Week {result['week']} (page updated {result['updated']})")
    print(f"  {len(result['players'])} Fanteam rows -> {p_path}")
    print(f"  {len(result['games'])} games        -> {g_path}")


if __name__ == "__main__":
    main()
