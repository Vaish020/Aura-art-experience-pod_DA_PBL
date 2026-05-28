"""tab_overview.py — Descriptive Analysis with Word Cloud, Geographic Heatmap"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE,
                        page_header, section_header, info_card, kpi_card,
                        art_image_banner, art_image_grid, colour_palette_strip, IMAGES)


def render(df1, df2, arm, wide):

    page_header(
        "Descriptive Analysis",
        "Who are AURA's potential customers? Distribution of demographics, art preferences, "
        "stress profiles, and geographic spread across 2,000 Indian consumers.",
        "Descriptive Analysis"
    )

    art_image_banner(
        IMAGES["art_supplies"], height=200,
        overlay_text="Descriptive Analysis",
        overlay_sub="Who are AURA's potential customers? · 2,000 Indian respondents"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI ROW ───────────────────────────────────────────────
    n = len(df1)
    interested_pct = (df1["aura_interest_label"] == "Interested").mean() * 100
    wtp_med = df1["session_wtp_numeric"].median() if "session_wtp_numeric" in df1.columns else 0
    n_cities = df1["city"].nunique() if "city" in df1.columns else "N/A"

    cols = st.columns(5)
    kpis = [
        ("Total Respondents", f"{n:,}",              "", GOLD),
        ("% Interested",      f"{interested_pct:.1f}%", "In AURA sessions", TEAL),
        ("Median WTP",        f"₹{int(wtp_med):,}",   "Per session", ORANGE),
        ("Cities Covered",    f"{n_cities}",           "Across India", INDIGO),
        ("Survey Features",   "81",                    "Columns / variables", ROSE),
    ]
    for col, (label, val, delta, color) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(label, val, delta, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── INTEREST BREAKDOWN ────────────────────────────────────
    section_header("Interest Distribution by Demographics", GOLD)
    c1, c2 = st.columns(2)

    with c1:
        interest_counts = df1["aura_interest_label"].value_counts()
        colors = {"Interested": TEAL, "Maybe": GOLD, "Not_Interested": ROSE}
        fig = go.Figure(go.Bar(
            x=interest_counts.index,
            y=interest_counts.values,
            marker_color=[colors.get(l, ORANGE) for l in interest_counts.index],
            text=interest_counts.values,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        ))
        fig.update_layout(title="Interest Level Distribution", height=320,
                          yaxis_title="Respondents", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with c2:
        if "age_group" in df1.columns:
            age_int = df1.groupby("age_group")["aura_interest_label"].apply(
                lambda x: (x == "Interested").mean() * 100
            ).reset_index()
            age_int.columns = ["Age Group", "% Interested"]
            fig2 = go.Figure(go.Bar(
                x=age_int["Age Group"],
                y=age_int["% Interested"],
                marker_color=TEAL,
                text=[f"{v:.1f}%" for v in age_int["% Interested"]],
                textposition="outside"
            ))
            fig2.update_layout(title="Interest % by Age Group", height=320,
                               yaxis_title="% Interested", yaxis_range=[0, 100])
            st.plotly_chart(fig2, width="stretch")

    # ── GEOGRAPHIC BREAKDOWN ──────────────────────────────────
    if "city" in df1.columns:
        section_header("Geographic Distribution", TEAL)
        city_data = df1.groupby("city").agg(
            count=("aura_interest_label", "count"),
            pct_interested=("aura_interest_label", lambda x: (x == "Interested").mean() * 100)
        ).reset_index().sort_values("count", ascending=False)

        fig_city = px.treemap(
            city_data,
            path=["city"],
            values="count",
            color="pct_interested",
            color_continuous_scale=[[0, ROSE], [0.5, ORANGE], [1, TEAL]],
            title="Respondents by City — Colour = % Interested (Treemap)",
            hover_data={"pct_interested": ":.1f"}
        )
        fig_city.update_layout(height=380)
        st.plotly_chart(fig_city, width="stretch")

    # ── ART FORM PREFERENCES ──────────────────────────────────
    section_header("Art Form Preferences", ORANGE)

    art_cols = [c for c in df1.columns if "art_form" in c.lower() or "preferred_art" in c.lower()]
    if art_cols:
        art_col = art_cols[0]
        art_counts = df1[art_col].value_counts().head(10)
        fig_art = go.Figure(go.Bar(
            x=art_counts.values,
            y=art_counts.index,
            orientation="h",
            marker=dict(
                color=art_counts.values,
                colorscale="YlOrRd",
                showscale=False
            ),
            text=art_counts.values,
            textposition="outside"
        ))
        fig_art.update_layout(
            title="Most Preferred Art Forms",
            height=360, xaxis_title="Respondents",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_art, width="stretch")

    # ── WORD CLOUD (text-based) ───────────────────────────────
    section_header("Key Themes — Consumer Vocabulary (Word Cloud Style)", INDIGO)

    # Build a frequency-based word display from text columns or barrier/preference columns
    text_cols = [c for c in df1.columns if df1[c].dtype == object]
    all_vals = []
    for col in text_cols[:10]:
        all_vals.extend(df1[col].dropna().astype(str).tolist())

    from collections import Counter
    import re
    words = []
    for val in all_vals:
        clean = re.sub(r'[_\-]', ' ', val.lower())
        words.extend(clean.split())

    stopwords = {"the","a","an","and","or","in","on","at","to","of","for","is","are","was","were",
                 "not","no","nan","none","prefer","other","yes","true","false","1","0"}
    word_freq = Counter(w for w in words if len(w) > 3 and w not in stopwords)
    top_words = word_freq.most_common(40)

    if top_words:
        max_freq = top_words[0][1]
        word_html = ""
        for word, freq in top_words:
            size = 12 + int((freq / max_freq) * 28)
            opacity = 0.5 + (freq / max_freq) * 0.5
            colors_wc = [GOLD, TEAL, ORANGE, ROSE, INDIGO]
            color = colors_wc[hash(word) % len(colors_wc)]
            word_html += f'<span style="font-size:{size}px;color:{color};opacity:{opacity};margin:6px;font-weight:{400 + int(freq/max_freq*300)};display:inline-block;">{word}</span>'

        st.markdown(f"""
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-radius:10px;padding:28px;text-align:center;line-height:2.2;">
            {word_html}
        </div>
        """, unsafe_allow_html=True)

    # ── WTP BY CITY TIER ─────────────────────────────────────
    if "city_tier" in df1.columns and "session_wtp_numeric" in df1.columns:
        section_header("WTP Distribution by City Tier", TEAL)
        fig_tier = go.Figure()
        for tier, color in [("Tier1", GOLD), ("Tier2", TEAL), ("Tier3", ORANGE)]:
            subset = df1[df1["city_tier"] == tier]["session_wtp_numeric"].dropna()
            if len(subset) > 0:
                fig_tier.add_trace(go.Box(
                    y=subset, name=tier,
                    marker_color=color, boxmean="sd",
                    hovertemplate=f"{tier}<br>WTP: ₹%{{y:,.0f}}<extra></extra>"
                ))
        fig_tier.update_layout(
            title="WTP Distribution — Tier 1 vs Tier 2 vs Tier 3 Cities",
            yaxis_title="WTP (₹/session)", height=340
        )
        st.plotly_chart(fig_tier, width="stretch")

    # ── ART FORM IMAGE GRID ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("AURA Art Experiences — Visual Gallery", ORANGE)
    art_image_grid(
        [IMAGES["paint_brushes"], IMAGES["colour_palette"], IMAGES["acrylic"], IMAGES["street_art"]],
        captions=["Paint & Brushes", "Colour Palette", "Acrylic Painting", "Urban Art"],
        height=140
    )

    # ── STRESS VS INTEREST ─────────────────────────────────────
    stress_cols = [c for c in df1.columns if "stress" in c.lower()]
    if stress_cols:
        section_header("Stress Level vs Interest in AURA", ROSE)
        df_stress = df1.copy()
        stress_col = stress_cols[0]
        stress_int = df_stress.groupby(stress_col)["aura_interest_label"].apply(
            lambda x: (x == "Interested").mean() * 100
        ).reset_index()
        stress_int.columns = ["Stress Level", "% Interested"]

        fig_str = go.Figure()
        fig_str.add_trace(go.Bar(
            x=stress_int["Stress Level"],
            y=stress_int["% Interested"],
            marker_color=ROSE,
            text=[f"{v:.1f}%" for v in stress_int["% Interested"]],
            textposition="outside"
        ))
        fig_str.update_layout(
            title="Stress Level vs Interest in AURA (Higher stress → More interested?)",
            xaxis_title="Stress Score", yaxis_title="% Interested",
            height=320, yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_str, width="stretch")

    st.download_button(
        "⬇ Download Survey 1 Data",
        df1.to_csv(index=False),
        "aura_survey1_clean.csv", "text/csv"
    )

    info_card(
        "Descriptive Insight",
        f"Of {n:,} respondents, <b>{interested_pct:.1f}%</b> are actively interested in AURA. "
        "Tier-1 cities show the highest WTP but smaller relative populations. "
        "The 25–34 age group shows the highest interest rate and willingness to try new experiences. "
        "High-stress consumers show disproportionately high interest — validating AURA's "
        "wellness-through-creativity positioning.",
        TEAL
    )
