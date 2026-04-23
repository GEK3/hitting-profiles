import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os

st.set_page_config(
    page_title="Hitting Profiles",
    page_icon="⚾",
    layout="wide",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-size: 1.4rem !important; font-weight: 600 !important; }
    .stDataFrame { font-size: 12px; }
    div[data-testid="metric-container"] { background: #f8f8f8; border-radius: 8px; padding: 8px 12px; }
    .subtitle { font-size: 0.8rem; color: #888; margin-top: -8px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Column config ─────────────────────────────────────────────────────────────
METRIC_COLS = [
    "SEAGER",
    "Damage/BBE",
    "Selectivity (%)",
    "Hittable Pitch Take (%)",
    "Chase (%)",
    "Z-Contact (%)",
    "Whiff vs. Secondaries (%)",
    "Z-Swing (%)",
    "Zone (%)",
]

# Columns where LOWER values = better (will be inverted for heat map)
LOWER_BETTER = {"Hittable Pitch Take (%)", "Chase (%)", "Whiff vs. Secondaries (%)"}

COL_LABELS = {
    "SEAGER":                    "SEAGER",
    "Damage/BBE":                "Damage/BBE",
    "Selectivity (%)":           "Selectivity",
    "Hittable Pitch Take (%)":   "Hittable Take",
    "Chase (%)":                 "Chase",
    "Z-Contact (%)":             "Z-Contact",
    "Whiff vs. Secondaries (%)": "Whiff vs Sec",
    "Z-Swing (%)":               "Z-Swing",
    "Zone (%)":                  "Zone",
}

COL_HELP = {
    "SEAGER":                    "SElective AGgression Engagement Rate — Selection Tendency minus Hittable Pitches Taken. Higher = better balance of attacking good pitches while laying off bad ones.",
    "Damage/BBE":                "% of air balls clearing the 80th-pctile EV threshold at that spray angle or launch angle. Measures raw power application.",
    "Selectivity (%)":           "Good Takes / Good Decisions. How often good decisions result from taking pitches. Higher = more selective.",
    "Hittable Pitch Take (%)":   "Bad Takes / Total Takes. % of takes that were hittable pitches. Lower = more opportunistic (fewer free swings left on table).",
    "Chase (%)":                 "O-Swing%. Lower = better zone discipline.",
    "Z-Contact (%)":             "Contact rate on pitches in the zone. Higher = better bat-to-ball in zone.",
    "Whiff vs. Secondaries (%)": "Whiff/swing rate against breaking balls and offspeed. Lower = harder to put away.",
    "Z-Swing (%)":               "Swing rate on pitches in the zone. Context-dependent — not directly good or bad.",
    "Zone (%)":                  "% of pitches thrown in the strike zone to this hitter.",
}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    data_path = Path(__file__).parent / "data" / "leaderboard.csv"
    if not data_path.exists():
        st.error("No data file found. Run pipeline.py first or wait for the GitHub Action to run.")
        st.stop()
    df = pd.read_csv(data_path)
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    pct = df[["Name", "Team"]].copy()
    if "PA" in df.columns:
        pct["PA"] = df["PA"]
    for col in METRIC_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        lower = col in LOWER_BETTER
        pct[col] = df[col].apply(
            lambda v: np.nan if pd.isna(v) else
            float(np.mean(series < v) * 100) if not lower else
            float(np.mean(series > v) * 100)
        )
    return pct


def style_heatmap(df: pd.DataFrame, mode: str):
    display = df.copy()

    # Format display values
    if mode == "raw":
        for col in METRIC_COLS:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                )
    else:
        for col in METRIC_COLS:
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda v: f"{round(v)}" if pd.notna(v) else "—"
                )

    # Rename columns for display
    display = display.rename(columns=COL_LABELS)

    styler = display.style

    # Apply per-column background gradient using the underlying numeric pct values
    pct_df = compute_percentiles(load_data()) if mode == "raw" else df
    for col in METRIC_COLS:
        if col not in pct_df.columns:
            continue
        display_col = COL_LABELS.get(col, col)
        if display_col not in display.columns:
            continue
        vals = pct_df[col].copy()
        # Map 0-100 to color: low=red, mid=yellow, high=green
        def bg(val_str, col=col, vals=vals, mode=mode, display=display):
            if val_str == "—":
                return ""
            try:
                idx = display.columns.get_loc(COL_LABELS.get(col, col))
            except Exception:
                return ""
            return ""

        # Use background_gradient on a numeric proxy column
        styler = styler.background_gradient(
            cmap="RdYlGn",
            subset=[display_col],
            vmin=0,
            vmax=100,
            gmap=pct_df[col].values,
        )

    styler = styler.set_properties(**{
        "text-align": "right",
        "font-size": "12px",
        "padding": "4px 10px",
    })
    styler = styler.set_properties(
        subset=["Name"],
        **{"text-align": "left", "font-weight": "500", "min-width": "140px"}
    )
    styler = styler.set_properties(
        subset=["Team"],
        **{"text-align": "left", "color": "#888", "font-size": "11px"}
    )
    styler = styler.set_table_styles([
        {"selector": "thead th", "props": [
            ("background-color", "#f0f0f0"),
            ("font-size", "11px"),
            ("font-weight", "600"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.04em"),
            ("padding", "6px 10px"),
            ("border-bottom", "1px solid #ddd"),
        ]},
        {"selector": "tbody tr:hover td", "props": [("background-color", "#f5f5f5 !important")]},
    ])
    return styler


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    st.title("⚾ Hitting Profiles")
    st.markdown('<p class="subtitle">Plate discipline & damage metrics — SEAGER framework</p>',
                unsafe_allow_html=True)

    df_raw = load_data()

    # Last updated
    data_path = Path(__file__).parent / "data" / "leaderboard.csv"
    if data_path.exists():
        mtime = data_path.stat().st_mtime
        import datetime
        updated = datetime.datetime.fromtimestamp(mtime).strftime("%b %d, %Y")
        st.caption(f"Data updated: {updated} · {len(df_raw)} players")

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])

    with col1:
        search = st.text_input("Search player or team", placeholder="e.g. Judge, NYY", label_visibility="collapsed")

    with col2:
        mode = st.radio("View", ["Raw values", "Percentiles"], horizontal=True, label_visibility="collapsed")
        mode_key = "raw" if mode == "Raw values" else "pct"

    with col3:
        seager_min = st.slider(
            "Min SEAGER",
            min_value=float(df_raw["SEAGER"].min()) if "SEAGER" in df_raw else 0.0,
            max_value=float(df_raw["SEAGER"].max()) if "SEAGER" in df_raw else 30.0,
            value=float(df_raw["SEAGER"].min()) if "SEAGER" in df_raw else 0.0,
            step=0.5,
            format="%.1f",
        )

    with col4:
        if "PA" in df_raw.columns:
            pa_min = st.number_input("Min PA", min_value=0, max_value=700, value=0, step=50)
        else:
            pa_min = 0

    # ── Filter ────────────────────────────────────────────────────────────────
    df = df_raw.copy()

    if search:
        mask = (
            df["Name"].str.lower().str.contains(search.lower(), na=False) |
            df["Team"].str.lower().str.contains(search.lower(), na=False)
        )
        df = df[mask]

    if "SEAGER" in df.columns:
        df = df[df["SEAGER"] >= seager_min]

    if "PA" in df.columns and pa_min > 0:
        df = df[df["PA"] >= pa_min]

    df = df.sort_values("SEAGER", ascending=False).reset_index(drop=True)

    st.caption(f"Showing {len(df)} players")

    # ── Build display dataframe ───────────────────────────────────────────────
    display_cols = ["Name", "Team"]
    if "PA" in df.columns:
        display_cols.append("PA")
    display_cols += [c for c in METRIC_COLS if c in df.columns]

    if mode_key == "pct":
        pct_df = compute_percentiles(df_raw)
        # Filter pct_df to same rows as df
        pct_df = pct_df[pct_df["Name"].isin(df["Name"])].copy()
        pct_df = pct_df.sort_values("SEAGER", ascending=False).reset_index(drop=True)
        display_df = pct_df[[c for c in display_cols if c in pct_df.columns]]
    else:
        display_df = df[[c for c in display_cols if c in df.columns]]

    # ── Render ────────────────────────────────────────────────────────────────
    pct_for_style = compute_percentiles(df_raw) if mode_key == "raw" else display_df.copy()
    # Align pct_for_style to display_df rows
    pct_for_style = pct_for_style[pct_for_style["Name"].isin(display_df["Name"])].copy()
    pct_for_style = pct_for_style.sort_values("SEAGER", ascending=False).reset_index(drop=True)

    # Format numeric display
    fmt_df = display_df.copy()
    if mode_key == "raw":
        for col in METRIC_COLS:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    else:
        for col in METRIC_COLS:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].apply(lambda v: f"{round(v)}" if pd.notna(v) else "—")

    fmt_df = fmt_df.rename(columns=COL_LABELS)

    # Build styler with gradient mapped to percentile values
    styler = fmt_df.style
    for col in METRIC_COLS:
        if col not in pct_for_style.columns:
            continue
        display_col = COL_LABELS.get(col, col)
        if display_col not in fmt_df.columns:
            continue
        gmap = pct_for_style[col].values
        if len(gmap) == len(fmt_df):
            styler = styler.background_gradient(
                cmap="RdYlGn",
                subset=[display_col],
                vmin=0,
                vmax=100,
                gmap=gmap,
            )

    styler = styler.set_properties(**{"text-align": "right", "font-size": "12px"})
    styler = styler.set_properties(
        subset=["Name"], **{"text-align": "left", "font-weight": "600"}
    )
    if "Team" in fmt_df.columns:
        styler = styler.set_properties(
            subset=["Team"], **{"text-align": "left", "color": "#888"}
        )

    st.dataframe(styler, use_container_width=True, height=600)

    # ── Metric glossary ───────────────────────────────────────────────────────
    with st.expander("Metric definitions"):
        for col, desc in COL_HELP.items():
            st.markdown(f"**{col}** — {desc}")

    # ── Manual refresh ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Data controls"):
        st.markdown("The leaderboard updates automatically each day via GitHub Actions. "
                    "To trigger a manual refresh, push to your repo or re-run the workflow from GitHub Actions.")
        if st.button("Clear cached data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
