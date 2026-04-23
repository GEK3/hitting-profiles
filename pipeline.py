"""pipeline.py - Hitting Profiles data pipeline"""
import argparse, datetime, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pybaseball
from pybaseball import statcast
warnings.filterwarnings("ignore")
pybaseball.cache.enable()
OUTPUT_PATH = Path(__file__).parent / "data" / "leaderboard.csv"

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

def get_current_teams(raw):
    r = raw.copy()
    r["batter_team"] = np.where(r["inning_topbot"] == "Top", r["away_team"], r["home_team"])
    return (r.sort_values("game_date").groupby("batter")["batter_team"].last()
            .reset_index().rename(columns={"batter_team": "Team"}))

def compute_pa(pitches):
    pa_events = {"strikeout","single","double","triple","home_run","walk","hit_by_pitch",
        "field_out","force_out","grounded_into_double_play","sac_fly","sac_bunt",
        "fielders_choice","fielders_choice_out","double_play","strikeout_double_play",
        "intent_walk","field_error","other_out","catcher_interf"}
    return pitches[pitches["events"].isin(pa_events)].groupby("batter").size().reset_index(name="PA")

def get_player_info(mlbam_ids):
    import urllib.request, json
    results = {}
    batch_size = 100
    for i in range(0, len(mlbam_ids), batch_size):
        batch = mlbam_ids[i:i+batch_size]
        ids_str = ",".join(str(x) for x in batch)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}&fields=people,id,fullName,primaryPosition,abbreviation"
        try:
            with urllib.request.urlopen(url) as r:
                data = json.loads(r.read())
            for p in data["people"]:
                results[p["id"]] = {
                    "name": p["fullName"],
                    "pos": p.get("primaryPosition", {}).get("abbreviation", "?")
                }
        except Exception as e:
            print(f"  API error: {e}")
    return results

def get_hitter_ids(raw, start):
    batter_ids = raw["batter"].dropna().astype(int).unique().tolist()
    print(f"  Looking up {len(batter_ids)} batter IDs via MLB API...")
    info = get_player_info(batter_ids)
    hitters = set(k for k,v in info.items() if v["pos"] != "P")
    pitchers = set(k for k,v in info.items() if v["pos"] == "P")
    print(f"  Excluded {len(pitchers)} pitchers")
    print(f"  {len(hitters)} hitters remaining")
    return hitters, info

def compute_damage_and_pulled_fb(batted):
    bb = batted.dropna(subset=["launch_speed","launch_angle"]).copy()
    bb["spray_angle"] = compute_spray_angle(bb)
    bb["spray_bin"] = bb["spray_angle"].round(0).clip(-45,45).astype(int)
    bb["la_bin"] = bb["launch_angle"].round(0).clip(-90,90).astype(int)
    spray_curve = (bb.groupby("spray_bin")["launch_speed"].quantile(0.80)
                   .reset_index().rename(columns={"launch_speed":"ev_thr_spray"}))
    la_curve = (bb.groupby("la_bin")["launch_speed"].quantile(0.80)
                .reset_index().rename(columns={"launch_speed":"ev_thr_la"}))
    air = bb[bb["launch_angle"] > 0].copy()
    air = air.merge(spray_curve, on="spray_bin", how="left")
    air = air.merge(la_curve, on="la_bin", how="left")
    air["is_damage"] = ((air["launch_speed"] >= air["ev_thr_spray"]) | (air["launch_speed"] >= air["ev_thr_la"]))
    damage = air.groupby("batter").agg(damage_count=("is_damage","sum"),bbe=("is_damage","count")).reset_index()
    damage["Damage/BBE"] = (damage["damage_count"] / damage["bbe"] * 100).round(1)
    pulled = (bb.groupby("batter").apply(lambda g: pd.Series({
        "Pulled FB (%)": round(((g["launch_angle"]>20)&(g["spray_angle"]>=15)).sum()/len(g)*100,1)
        if len(g)>0 else np.nan})).reset_index())
    return damage[["batter","Damage/BBE"]], pulled

def compute_seager(pitches):
    p = pitches.dropna(subset=["plate_x","plate_z","balls","strikes","delta_run_exp"]).copy()
    p["px_bin"] = pd.cut(p["plate_x"], bins=np.linspace(-2,2,13), labels=False)
    p["pz_bin"] = pd.cut(p["plate_z"], bins=np.linspace(0,5,13), labels=False)
    swing_desc = {"hit_into_play","swinging_strike","swinging_strike_blocked","foul","foul_tip","foul_bunt","missed_bunt"}
    take_desc = {"called_strike","ball","blocked_ball","pitchout","hit_by_pitch"}
    p["is_swing"] = p["description"].isin(swing_desc)
    p["is_take"] = p["description"].isin(take_desc)
    p["in_zone"] = (p["plate_x"].abs() < 0.83) & p["plate_z"].between(1.5, 3.5)
    key = ["balls","strikes","px_bin","pz_bin"]
    swing_rv = p[p["is_swing"]].groupby(key)["delta_run_exp"].mean().reset_index().rename(columns={"delta_run_exp":"swing_rv"})
    take_rv = p[p["is_take"]].groupby(key)["delta_run_exp"].mean().reset_index().rename(columns={"delta_run_exp":"take_rv"})
    p = p.merge(swing_rv, on=key, how="left").merge(take_rv, on=key, how="left")
    p["swing_rv"] = p["swing_rv"].fillna(0)
    p["take_rv"] = p["take_rv"].fillna(0)
    p["ev_delta"] = p["swing_rv"] - p["take_rv"]
    p["should_swing"] = p["ev_delta"] > 0
    def calc(g):
        sw = g[g["is_swing"]]; tk = g[g["is_take"]]
        good_sw = sw["should_swing"].sum()
        good_tk = (~tk["should_swing"]).sum()
        gd = good_sw + good_tk
        hit_taken = tk["should_swing"].sum()
        tot_tk = len(tk)
        sel = good_tk/gd if gd>0 else np.nan
        hpt = hit_taken/tot_tk if tot_tk>0 else np.nan
        seager = sel-hpt if pd.notna(sel) and pd.notna(hpt) else np.nan
        o_sw = g[g["is_swing"] & ~g["in_zone"]]
        o_all = g[~g["in_zone"]]
        chase = len(o_sw)/len(o_all) if len(o_all)>0 else np.nan
        z_sw = g[g["is_swing"] & g["in_zone"]]
        z_contact = (z_sw["description"].isin({"hit_into_play","foul","foul_tip"}).sum()/len(z_sw) if len(z_sw)>0 else np.nan)
        return pd.Series({
            "SEAGER": round(seager*100,1) if pd.notna(seager) else np.nan,
            "Selectivity (%)": round(sel*100,1) if pd.notna(sel) else np.nan,
            "Hittable Pitch Take (%)": round(hpt*100,1) if pd.notna(hpt) else np.nan,
            "Chase (%)": round(chase*100,1) if pd.notna(chase) else np.nan,
            "Z-Contact (%)": round(z_contact*100,1) if pd.notna(z_contact) else np.nan,
            "Zone (%)": round(g["in_zone"].mean()*100,1),
            "Z-Swing (%)": round(len(z_sw)/g["in_zone"].sum()*100,1) if g["in_zone"].sum()>0 else np.nan,
        })
    return p.groupby("batter").apply(calc).reset_index()

def compute_whiff_sec(pitches):
    secondary = {"SL","CU","KC","SV","ST","CUO","SLO","CH","FS","FO","SC","CS"}
    p = pitches[pitches["pitch_type"].isin(secondary)].copy()
    swing_desc = {"hit_into_play","swinging_strike","swinging_strike_blocked","foul","foul_tip","foul_bunt","missed_bunt"}
    whiff_desc = {"swinging_strike","swinging_strike_blocked","missed_bunt"}
    p["is_swing"] = p["description"].isin(swing_desc)
    p["is_whiff"] = p["description"].isin(whiff_desc)
    return (p.groupby("batter").apply(lambda g: pd.Series({
        "Whiff vs. Secondaries (%)": round(g["is_whiff"].sum()/g["is_swing"].sum()*100,1)
        if g["is_swing"].sum()>0 else np.nan})).reset_index())

def compute_ev_metrics(batted):
    bb = batted.dropna(subset=["launch_speed"])
    return (bb.groupby("batter")["launch_speed"]
            .agg(ev_90=lambda x: round(np.percentile(x,90),1), max_ev=lambda x: round(x.max(),1))
            .reset_index().rename(columns={"ev_90":"90th Pctile EV","max_ev":"Max EV"}))

def run(start, end):
    print(f"Pulling Statcast data {start} to {end}...")
    raw = statcast(start, end)
    print(f"  {len(raw):,} pitches")
    teams = get_current_teams(raw)
    stands = raw[["batter","stand"]].dropna().drop_duplicates("batter")
    print("Filtering to hitters...")
    hitter_ids, player_info = get_hitter_ids(raw, start)
    raw = raw[raw["batter"].isin(hitter_ids)].copy()
    names = pd.DataFrame([{"batter": k, "player_name": v["name"]} for k,v in player_info.items() if v["pos"] != "P"])
    print(f"  {raw[chr(98)+chr(97)+chr(116)+chr(116)+chr(101)+chr(114)].nunique()} unique hitters")
    batted = raw[raw["type"] == "X"].copy()
    if "stand" not in batted.columns:
        batted = batted.merge(stands, on="batter", how="left")
    print("Computing Damage/BBE and Pulled FB%...")
    damage, pulled_fb = compute_damage_and_pulled_fb(batted)
    print("Computing EV metrics...")
    ev = compute_ev_metrics(batted)
    print("Computing SEAGER / plate discipline...")
    seager_df = compute_seager(raw)
    print("Computing Whiff vs. Secondaries...")
    whiff_sec = compute_whiff_sec(raw)
    print("Computing PA...")
    pa = compute_pa(raw)
    df = names[names["batter"].isin(hitter_ids)].copy()
    for d in [teams, pa, damage, pulled_fb, ev, seager_df, whiff_sec]:
        df = df.merge(d, on="batter", how="left")
    df = df.rename(columns={"player_name":"Name"})
    df = df.drop(columns=["batter"], errors="ignore")
    df = df.sort_values("SEAGER", ascending=False).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {str(OUTPUT_PATH)}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    start, end = args.start, args.end
    if not start or not end:
        start, end = season_dates()
    run(start, end)