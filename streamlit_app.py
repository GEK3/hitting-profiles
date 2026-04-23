import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime

st.set_page_config(page_title="Hitting Profiles", page_icon="⚾", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem}.subtitle{font-size:0.8rem;color:#888;margin-top:-8px;margin-bottom:12px}</style>""", unsafe_allow_html=True)

METRIC_COLS=["SEAGER","Damage/BBE","Selectivity (%)","Hittable Pitch Take (%)","Chase (%)","Z-Contact (%)","Whiff vs. Secondaries (%)","Z-Swing (%)","Zone (%)"]
LOWER_BETTER={"Hittable Pitch Take (%)","Chase (%)","Whiff vs. Secondaries (%)"}
COL_LABELS={"SEAGER":"SEAGER","Damage/BBE":"Damage/BBE","Selectivity (%)":"Selectivity","Hittable Pitch Take (%)":"Hittable Take","Chase (%)":"Chase","Z-Contact (%)":"Z-Contact","Whiff vs. Secondaries (%)":"Whiff vs Sec","Z-Swing (%)":"Z-Swing","Zone (%)":"Zone"}
COL_HELP={"SEAGER":"SElective AGgression Engagement Rate. Selection Tendency minus Hittable Pitches Taken. Higher = better.","Damage/BBE":"% of air balls clearing the 80th-pctile EV threshold at that spray/launch angle.","Selectivity (%)":"Good Takes / Good Decisions. Higher = more selective.","Hittable Pitch Take (%)":"Bad Takes / Total Takes. Lower = more opportunistic.","Chase (%)":"O-Swing%. Lower = better zone discipline.","Z-Contact (%)":"Contact rate on pitches in the zone. Higher = better.","Whiff vs. Secondaries (%)":"Whiff/swing vs breaking + offspeed. Lower = harder to put away.","Z-Swing (%)":"Swing rate on pitches in the zone.","Zone (%)":"% of pitches thrown in the strike zone."}

@st.cache_data(ttl=3600)
def load_data():
    p = Path(__file__).parent / "data" / "leaderboard.csv"
    if not p.exists():
        st.error("No data file found."); st.stop()
    df = pd.read_csv(p)
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def pct_rank(df_full, df_sub):
    out = df_sub[["Name","Team"]].copy()
    if "PA" in df_sub.columns: out["PA"] = df_sub["PA"]
    for col in METRIC_COLS:
        if col not in df_full.columns: continue
        series = df_full[col].dropna()
        lower = col in LOWER_BETTER
        out[col] = df_sub[col].apply(lambda v: np.nan if pd.isna(v) else float(np.mean(series>v)*100) if lower else float(np.mean(series<v)*100))
    return out

def style_table(display_df, pct_df):
    fmt = display_df.copy()
    for col in METRIC_COLS:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    fmt = fmt.rename(columns=COL_LABELS)
    s = fmt.style
    for col in METRIC_COLS:
        dc = COL_LABELS.get(col,col)
        if col not in pct_df.columns or dc not in fmt.columns: continue
        gmap = pct_df[col].values
        if len(gmap)==len(fmt):
            s = s.background_gradient(cmap="RdYlGn",subset=[dc],vmin=0,vmax=100,gmap=gmap)
    s = s.set_properties(**{"text-align":"right","font-size":"12px"})
    s = s.set_properties(subset=["Name"],**{"text-align":"left","font-weight":"600"})
    if "Team" in fmt.columns: s = s.set_properties(subset=["Team"],**{"text-align":"left","color":"#888"})
    return s

def main():
    st.title("⚾ Hitting Profiles")
    st.markdown('<p class="subtitle">Plate discipline & damage metrics — SEAGER framework</p>', unsafe_allow_html=True)
    df_raw = load_data()
    p = Path(__file__).parent / "data" / "leaderboard.csv"
    if p.exists():
        updated = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d, %Y")
        st.caption(f"Data updated: {updated} · {len(df_raw)} players")
    st.divider()

    c1,c2,c3,c4 = st.columns([2,1,1,1])
    with c1: search = st.text_input("Search player or team", placeholder="e.g. Ohtani, NYY")
    with c2: mode = st.radio("View",["Raw values","Percentiles"],horizontal=True)
    with c3:
        available = [c for c in METRIC_COLS if c in df_raw.columns]
        sort_col = st.selectbox("Sort by", available, index=0)
    with c4: sort_dir = st.radio("Order",["Descending","Ascending"],horizontal=True)

    st.markdown("")
    d1,d2,d3,d4,d5 = st.columns(5)
    with d1:
        pa_min = st.number_input("Min PA",min_value=0,max_value=700,value=0,step=10)
        pa_max = st.number_input("Max PA",min_value=0,max_value=700,value=700,step=10)
    with d2:
        teams = sorted(df_raw["Team"].dropna().unique().tolist()) if "Team" in df_raw.columns else []
        sel_teams = st.multiselect("Team", teams)
    with d3:
        positions = sorted(df_raw["Position"].dropna().unique().tolist()) if "Position" in df_raw.columns else []
        sel_pos = st.multiselect("Position", positions)
    with d4:
        seager_min = st.number_input("Min SEAGER",value=float(df_raw["SEAGER"].min()) if "SEAGER" in df_raw.columns else 0.0,step=0.5,format="%.1f")
    with d5:
        dmg_min = st.number_input("Min Damage/BBE",min_value=0.0,max_value=100.0,value=0.0,step=1.0,format="%.1f")

    with st.expander("Show/hide columns"):
        tcols = st.columns(len(METRIC_COLS))
        toggles = {}
        for i,col in enumerate(METRIC_COLS):
            with tcols[i]: toggles[col] = st.checkbox(COL_LABELS.get(col,col),value=True)

    df = df_raw.copy()
    if search:
        mask = df["Name"].str.lower().str.contains(search.lower(),na=False)|df["Team"].str.lower().str.contains(search.lower(),na=False)
        df = df[mask]
    if "PA" in df.columns: df = df[(df["PA"]>=pa_min)&(df["PA"]<=pa_max)]
    if "SEAGER" in df.columns: df = df[df["SEAGER"]>=seager_min]
    if "Damage/BBE" in df.columns: df = df[df["Damage/BBE"]>=dmg_min]
    if sel_teams: df = df[df["Team"].isin(sel_teams)]
    if sel_pos and "Position" in df.columns: df = df[df["Position"].isin(sel_pos)]
    if sort_col in df.columns: df = df.sort_values(sort_col,ascending=(sort_dir=="Ascending")).reset_index(drop=True)

    st.caption(f"Showing {len(df)} players")

    dcols = ["Name","Team"]
    if "Position" in df.columns: dcols.append("Position")
    if "PA" in df.columns: dcols.append("PA")
    vis = [c for c in METRIC_COLS if c in df.columns and toggles.get(c,True)]
    dcols += vis
    df_disp = df[[c for c in dcols if c in df.columns]].copy()

    if mode=="Percentiles":
        pct_df = pct_rank(df_raw, df_disp)
        for col in vis:
            if col in pct_df.columns: df_disp[col] = pct_df[col]
        pct_style = pct_df
    else:
        pct_style = pct_rank(df_raw, df_disp)

    st.dataframe(style_table(df_disp, pct_style), use_container_width=True, height=650, hide_index=True, key=f"{sort_col}_{sort_dir}_{len(df_disp)}")

    with st.expander("Metric definitions"):
        for col,desc in COL_HELP.items():
            st.markdown(f"**{col}** — {desc}")

    st.divider()
    with st.expander("Data controls"):
        st.markdown("The leaderboard updates automatically each day via GitHub Actions.")
        if st.button("Clear cached data"):
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
