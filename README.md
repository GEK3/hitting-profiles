# Hitting Profiles

Live leaderboard for SEAGER-based plate discipline and damage metrics.

## Metrics

| Metric | Definition |
|---|---|
| SEAGER | SElective AGgression Engagement Rate. Selection Tendency − Hittable Pitches Taken |
| Damage/BBE | % of air balls clearing 80th-pctile EV threshold at that spray/launch angle |
| Selectivity | Good Takes / Good Decisions |
| Hittable Pitch Take | Bad Takes / Total Takes (lower = better) |
| Chase | O-Swing% (lower = better) |
| Z-Contact | Contact rate in zone |
| Whiff vs. Secondaries | Whiff/swing vs breaking + offspeed (lower = better) |

## Setup

### 1. Run pipeline locally (first time)

```bash
pip install -r requirements.txt
python pipeline.py --start 2026-03-27 --end 2026-04-22
```

This writes `data/leaderboard.csv`. Commit it to the repo.

### 2. Run Streamlit locally

```bash
streamlit run streamlit_app.py
```

### 3. Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → connect your repo → main file: `streamlit_app.py`
4. Deploy

### 4. Enable automatic daily refresh

The GitHub Action at `.github/workflows/update_data.yml` runs at 8 AM ET daily.

For the action to push back to the repo, go to:
**Settings → Actions → General → Workflow permissions** → set to "Read and write permissions"

Then the leaderboard auto-updates every day without any manual work.

## Architecture

```
pipeline.py          pulls Statcast via pybaseball, computes all metrics
data/leaderboard.csv updated daily by GitHub Actions, read by the app
streamlit_app.py     reads the CSV, renders heatmap table with controls
```
