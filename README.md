# Gridiron Lab: Fanteam NFL optimizer

A Streamlit app that turns the Daily Fantasy Tips Fanteam projections CSV into cash, GPP and contrarian lineups. It projects ownership (Fanteam shows none before lock), lets you paste the real numbers after lock, and ranks GPP lineups by simulated top-1% odds against a field built from that ownership.

It follows the same shape as your EuroLeague optimizer (PuLP model, presets, ownership calibration), rebuilt for NFL roster rules and correlation.

## Files

```
nfl-fanteam-optimizer/
├── app.py                     Streamlit interface (tabs: Slate, Ownership, Build lineups, Post-lock)
├── engine.py                  Model, ownership, optimizer, simulation, calibration (no Streamlit code)
├── theme.py                   NFL styling: CSS, team colours, lineup cards
├── requirements.txt           streamlit, pandas, numpy, pulp, scipy
├── .streamlit/config.toml     Dark navy and NFL red theme
├── data/
│   ├── nfl_fanteam_week6.csv  Your Week 6 projections (204 players)
│   └── week6_games.csv        Game totals and implied points from the article
├── tests/test_engine.py       Smoke test: builds all four lineup types and checks every rule
├── .vscode/                   Interpreter setting and two run/debug configs
└── .gitignore
```

## 1. Check the contest rules first

I could not confirm Fanteam's weekly NFL roster format from public pages, so the defaults are assumptions. Open the contest lobby and compare:

| Setting | Default in the app | Where it came from |
|---|---|---|
| Budget | 110 credits | The Daily Fantasy Tips Fanteam section |
| Roster | QB, 2 RB, 3 WR, TE, FLEX (RB/WR/TE), DST = 9 players | Standard NFL DFS layout. **Verify.** |
| Max per NFL team | 4 | Assumption. Fanteam's season-long NFL game uses 3. **Verify.** |
| Defense counts toward team limit | Yes | Assumption |

Everything above is editable in the sidebar under **Contest rules**, no code changes needed. If the roster differs, change the position counts and the FLEX switch there.

## 2. Set up in VS Code

Prerequisites: Python 3.11 or 3.12, Git, VS Code with the Python extension.

1. Unzip the project and open the folder in VS Code (File > Open Folder).
2. Open a terminal (Terminal > New Terminal) and create a virtual environment.

   Windows (PowerShell):
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   macOS or Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
3. Install the packages: `pip install -r requirements.txt`
4. Select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > the `.venv` one.
5. Run the engine test: `python tests/test_engine.py`. You should see four lineup tables and `OK` at the end (about 40 seconds).
6. Start the app: `streamlit run app.py`. It opens at http://localhost:8501 with the Week 6 data already loaded.

If step 5 or 6 throws an error, copy the full message back to me and I will fix it. I could only test the engine in my environment (using SciPy's solver), not the Streamlit screens or the PuLP solver, so the first local run is the real check.

## 3. Push to GitHub

Your GitHub username is `theprice1`.

1. Create an empty repository on github.com named `nfl-fanteam-optimizer`. Do not tick "add a README" because the project already has one. Public is simplest for Streamlit Cloud's free tier, and private works if you grant Streamlit access.
2. In the VS Code terminal:
   ```
   git init
   git add .
   git commit -m "Initial NFL Fanteam optimizer"
   git branch -M main
   git remote add origin https://github.com/theprice1/nfl-fanteam-optimizer.git
   git push -u origin main
   ```
3. VS Code may open a browser window to sign in to GitHub the first time. Approve it and rerun `git push` if needed.

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app > "Deploy a public app from GitHub".
3. Repository `theprice1/nfl-fanteam-optimizer`, branch `main`, main file path `app.py`.
4. Advanced settings: choose Python 3.12. No secrets are needed.
5. Deploy. The first build takes a few minutes while it installs PuLP. After that, every `git push` redeploys automatically.

Streamlit Cloud forgets files written while the app runs. Anything you want to keep (calibration log, new projections) must be committed to the repo.

## 5. Weekly workflow

**Before lock**
1. Download the new projections CSV from Daily Fantasy Tips (same columns as this week's) and save it as `data/nfl_fanteam_week7.csv`. The app loads the newest `nfl_fanteam*.csv` in `data/`, or you can upload one in the sidebar without touching the repo.
2. Update the games file (`data/week7_games.csv`: away, home, total, away_implied, home_implied) or edit the totals on the Slate tab. Blank implied totals fall back to the slate average.
3. Check the **Status** column. Anything marked `possible` is scaled by the availability slider, and the build screen warns you about any such player used in your lineups.
4. Build **Cash** (a few lineups), **GPP** (about 20) and **Contrarian** (about 20) on the Build tab. Download the CSVs. They include slot order and Fanteam IDs.

**After lock**
5. Copy the real ownership from the Fanteam lobby into the **Post-lock** tab, one `Player, 23.4%` per line, then click **Apply pasted ownership**. Those numbers replace the estimate everywhere.
6. Click **Fit model to pasted ownership** to see how far off the estimate was and which settings would have matched. **Use fitted settings** applies them for next time, and **Add to calibration log** records the result.
7. Download the calibration log, save it as `data/calibration_log.csv`, commit and push. After a few weeks the fitted settings should settle, and you can set them as the new defaults in `engine.py` (`OwnParams`).

## 6. How the three lineup types differ

| | Cash | GPP | Contrarian |
|---|---|---|---|
| Objective | Mix of projection and 20th-percentile floor | Mix of projection and 90th-percentile ceiling | Same as GPP |
| Ownership | Ignored | Mild penalty (leverage 0.35) | Heavier penalty (0.5) plus limits |
| Stacks | None | QB plus 1 WR/TE, 1 bring-back | Same |
| Selection | Best by objective | 60 candidates ranked by simulated top-1% odds | Same, rewarding low-ownership lineups |
| Limits | None | Stay within 93% of best projection | Within 95%, max 2 chalk players, total ownership 85% of a typical field lineup, at least 2 low-owned players |

Common to all: defense never faces your own offensive players (switchable), unspent credits are shown, and you can lock players, remove players, restrict quarterbacks, exclude teams and cap exposure.

## 7. How the model works

- **Player outcomes.** Each projection is the mean of a log-normal distribution, with wider spread for TE, WR and DST than for QB. That gives right-skewed outcomes, and the floor (20th percentile) and ceiling (90th percentile) come from the same curve.
- **Availability.** Expected 98%, possible 70%, unexpected 25% by default. Projections are multiplied by these when optimizing, and simulated players miss games at those rates.
- **Ownership estimate.** Within each position, a score from projection rank, points per credit and team implied total (minus penalties for uncertain status) is turned into shares that sum to the number of slots at that position, with caps so no single player can exceed a realistic level. It is a heuristic. Treat it as a starting point and overwrite with your own read or the post-lock paste.
- **Simulation.** Players share team passing, team rushing and game-environment factors, so a QB and his receivers rise together, and a defense does badly when the opposing offense does well. Lineups are scored against 1,500 ownership-weighted random lineups, simulation by simulation.

## 8. Limitations to keep in mind

- The simulation measures finishing position against a field, not prize sharing, so contrarian lineups look worse on top-1% odds than on projection alone. Their benefit (fewer duplicated lineups when a low-owned build hits) is what the ownership penalty and uniqueness reward aim at, but the app does not simulate payouts.
- Projections come from the source CSV unchanged. The app does not know about injuries, weather or line moves unless you edit status, totals or ownership yourself.
- A full GPP or contrarian build solves 60 or more optimization problems and takes roughly 20 to 60 seconds, with a progress bar. Shrink the candidate pool in the Build tab if you want it faster.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| "No valid lineup exists" | Loosen locks, QB stack, bring-back, contrarian limits, or raise the team cap |
| Fewer lineups than requested | Raise candidate pool size, variety or max exposure, or lower min different players |
| `ModuleNotFoundError: pulp` on Streamlit Cloud | Confirm `requirements.txt` is in the repo root and reboot the app |
| Fonts look default | The theme loads Google Fonts. Offline, it falls back to system fonts and everything still works |
| Name not matched in ownership paste | Paste names as Fanteam shows them. The app tolerates punctuation and suffixes such as Jr. and III |
