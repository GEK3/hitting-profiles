"""
pipeline.py — Hitting Profiles data pipeline
Run locally or via GitHub Actions to refresh data/leaderboard.csv

Usage:
    python pipeline.py --start 2026-03-27 --end 2026-04-22
    python pipeline.py  # defaults to full current season
"""

import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
pybaseball.cache.enable()

OUTPUT_PATH = Path(__file__).parent / "data" / "leaderboard.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def season_dates():
    year = datetime.date.today().year
    return f"{year}-03-20", datetime.date.today().strftime("%Y-%m-%d")


def compute_spray_angle(df):
    hc_x = df["hc_x"].fillna(125.42) - 125.42
    hc_y = 198.27 - df["hc_y"].fillna(198.27)
    angle = np.degrees(np.arctan2(hc_x, hc_y))
    lhh = df["stand"] == "L"
    angle[lhh] = -angle[lhh]
    return angle


# ── Metric computations ───────────────────────────────────────────────────────

def compute_damage_and_pulled_fb(batted):
    bb = batted.dropna(subset=["launch_speed", "launch_angle"]).copy()
    bb["spray_angle"] = compute_spray_angle(bb)

    # Build 80th-pctile EV thresholds per spray-angle degree and launch-angle degree
    bb["spray_bin"] = bb["spray_angle"].round(0).clip(-45, 45).astype(int)
    bb["la_bin"] = bb["launch_angle"].round(0).clip(-90, 90).astype(int)

    spray_curve = (bb.groupby("spray_bin")["launch_speed"]
                   .quantile(0.80).reset_index()
                   .rename(columns={"launch_speed": "ev_thr_spray"}))
    la_curve = (bb.groupby("la_bin")["launch_speed"]
                .quantile(0.80).reset_index()
                .rename(columns={"launch_speed": "ev_thr_la"}))

    air = bb[bb["launch_angle"] > 0].copy()
    air = air.merge(spray_curve, on="spray_bin", how="left")
    air = air.merge(la_curve, on="la_bin", how="left")
    air["is_damage"] = (
        (air["launch_speed"] >= air["ev_thr_spray"]) |
        (air["launch_speed"] >= air["ev_thr_la"])
    )

    damage = (air.groupby("batter")
              .agg(damage_count=("is_damage", "sum"), bbe=("is_damage", "count"))
              .reset_index())
    damage["Damage/BBE"] = (damage["damage_count"] / damage["bbe"] * 100).round(1)

    # Pulled FB: LA > 20, spray_angle >= 15 (pull side)
    pulled = (bb.groupby("batter")
              .apply(lambda g: pd.Series({
                  "Pulled FB (%)": round(
                      ((g["launch_angle"] > 20) & (g["spray_angle"] >= 15)).sum() / len(g) * 100, 1
                  ) if len(g) > 0 else np.nan
              }))
              .reset_index())

    return damage[["batter", "Damage/BBE"]], pulled


def compute_seager(pitches):
    p = pitches.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "delta_run_exp"]).copy()
    p["px_bin"] = pd.cut(p["plate_x"], bins=np.linspace(-2, 2, 13), labels=False)
    p["pz_bin"] = pd.cut(p["plate_z"], bins=np.linspace(0, 5, 13), labels=False)

    swing_desc = {"hit_into_play", "swinging_strike", "swinging_strike_blocked",
                  "foul", "foul_tip", "foul_bunt", "missed_bunt"}
    take_desc  = {"called_strike", "ball", "blocked_ball", "pitchout", "hit_by_pitch"}

    p["is_swing"] = p["description"].isin(swing_desc)
    p["is_take"]  = p["description"].isin(take_desc)
    p["in_zone"]  = (p["plate_x"].abs() < 0.83) & p["plate_z"].between(1.5, 3.5)

    key = ["balls", "strikes", "px_bin", "pz_bin"]
    swing_rv = (p[p["is_swing"]].groupby(key)["delta_run_exp"]
                .mean().reset_index().rename(columns={"delta_run_exp": "swing_rv"}))
    take_rv  = (p[p["is_take"]].groupby(key)["delta_run_exp"]
                .mean().reset_index().rename(columns={"delta_run_exp": "take_rv"}))

    p = p.merge(swing_rv, on=key, how="left").merge(take_rv, on=key, how="left")
    p["swing_rv"] = p["swing_rv"].fillna(0)
    p["take_rv"]  = p["take_rv"].fillna(0)
    p["ev_delta"] = p["swing_rv"] - p["take_rv"]
    p["should_swing"] = p["ev_delta"] > 0

    def calc(g):
        sw = g[g["is_swing"]]
        tk = g[g["is_take"]]
        good_sw = (sw["should_swing"]).sum()
        good_tk = (~tk["should_swing"]).sum()
        gd = good_sw + good_tk
        hit_taken = (tk["should_swing"]).sum()
        tot_tk = len(tk)
        sel = good_tk / gd if gd > 0 else np.nan
        hpt = hit_taken / tot_tk if tot_tk > 0 else np.nan
        seager = sel - hpt if pd.notna(sel) and pd.notna(hpt) else np.nan
        o_sw = g[g["is_swing"] & ~g["in_zone"]]
        o_all = g[~g["in_zone"]]
        chase = len(o_sw) / len(o_all) if len(o_all) > 0 else np.nan
        z_sw = g[g["is_swing"] & g["in_zone"]]
        z_contact_events = {"hit_into_play", "foul", "foul_tip"}
        z_contact = (z_sw["description"].isin(z_contact_events).sum() / len(z_sw)
                     if len(z_sw) > 0 else np.nan)
        return pd.Series({
            "SEAGER":                  round(seager * 100, 1) if pd.notna(seager) else np.nan,
            "Selectivity (%)":         round(sel * 100, 1)    if pd.notna(sel)    else np.nan,
            "Hittable Pitch Take (%)": round(hpt * 100, 1)    if pd.notna(hpt)    else np.nan,
            "Chase (%)":               round(chase * 100, 1)  if pd.notna(chase)  else np.nan,
            "Z-Contact (%)":           round(z_contact * 100, 1) if pd.notna(z_contact) else np.nan,
            "Zone (%)":                round(g["in_zone"].mean() * 100, 1),
            "Z-Swing (%)":             round(len(z_sw) / g["in_zone"].sum() * 100, 1)
                                       if g["in_zone"].sum() > 0 else np.nan,
        })

    return p.groupby("batter").apply(calc).reset_index()


def compute_whiff_sec(pitches):
    secondary = {"SL","CU","KC","SV","ST","CUO","SLO","CH","FS","FO","SC","CS"}
    p = pitches[pitches["pitch_type"].isin(secondary)].copy()
    swing_desc = {"hit_into_play","swinging_strike","swinging_strike_blocked",
                  "foul","foul_tip","foul_bunt","missed_bunt"}
    whiff_desc = {"swinging_strike","swinging_strike_blocked","missed_bunt"}
    p["is_swing"] = p["description"].isin(swing_desc)
    p["is_whiff"] = p["description"].isin(whiff_desc)
    result = (p.groupby("batter")
              .apply(lambda g: pd.Series({
                  "Whiff vs. Secondaries (%)": round(
                      g["is_whiff"].sum() / g["is_swing"].sum() * 100, 1
                  ) if g["is_swing"].sum() > 0 else np.nan
              }))
              .reset_index())
    return result


def compute_ev_metrics(batted):
    bb = batted.dropna(subset=["launch_speed"])
    result = (bb.groupby("batter")["launch_speed"]
              .agg(ev_90=lambda x: round(np.percentile(x, 90), 1),
                   max_ev=lambda x: round(x.max(), 1))
              .reset_index()
              .rename(columns={"ev_90": "90th Pctile EV", "max_ev": "Max EV"}))
    return result


def compute_pa(pitches):
    pa_events = {
        "strikeout","single","double","triple","home_run","walk","hit_by_pitch",
        "field_out","force_out","grounded_into_double_play","sac_fly","sac_bunt",
        "fielders_choice","fielders_choice_out","double_play","strikeout_double_play",
        "intent_walk","field_error","other_out","catcher_interf",
    }
    return (pitches[pitches["events"].isin(pa_events)]
            .groupby("batter").size().reset_index(name="PA"))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(start: str, end: str) -> pd.DataFrame:
    print(f"Pulling Statcast data {start} → {end}…")
    raw = statcast(start, end)
    print(f"  {len(raw):,} pitches")

    names  = raw[["batter", "player_name"]].drop_duplicates("batter")
    stands = raw[["batter", "stand"]].drop_duplicates("batter")
    teams  = (raw.sort_values("game_date")
              .groupby("batter")["away_team"].last()
              .reset_index().rename(columns={"away_team": "Team"}))

    batted = raw[raw["type"] == "X"].copy()
    if "stand" not in batted.columns:
        batted = batted.merge(stands, on="batter", how="left")

    print("Computing Damage/BBE and Pulled FB%…")
    damage, pulled_fb = compute_damage_and_pulled_fb(batted)

    print("Computing EV metrics…")
    ev = compute_ev_metrics(batted)

    print("Computing SEAGER / plate discipline…")
    seager_df = compute_seager(raw)

    print("Computing Whiff vs. Secondaries…")
    whiff_sec = compute_whiff_sec(raw)

    print("Computing PA…")
    pa = compute_pa(raw)

    df = names.copy()
    for d in [teams, pa, damage, pulled_fb, ev, seager_df, whiff_sec]:
        df = df.merge(d, on="batter", how="left")

    df = df.rename(columns={"player_name": "Name"})
    df = df.drop(columns=["batter"], errors="ignore")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()

    start, end = args.start, args.end
    if not start or not end:
        start, end = season_dates()

    run(start, end)
