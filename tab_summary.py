"""tab_summary.py — Executive Summary with art hero banner and image grid"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, SURFACE3, INK, PALETTE,
                        kpi_card, section_header, info_card,
                        art_image_banner, art_image_grid, colour_palette_strip,
                        IMAGES, BRUSH_STROKE_SVG)


def render(df1, df2, arm, wide, clf_results, reg_results, best_k, silhouettes):

    # ── HERO BANNER ──────────────────────────────────────────
    art_image_banner(IMAGES["hero_paint"], height=200,
        overlay_text="AURA Art Experience Pod",
        overlay_sub="India's first subscription-based art experience pod network · Data Analytics Dashboard"
    )

    # ── KPI ROW ───────────────────────────────────────────────
    interested_pct = (df1["aura_interest_label"] == "Interested").mean() * 100
    maybe_pct      = (df1["aura_interest_label"] == "Maybe").mean() * 100
    total_n        = len(df1)
    wtp_col        = "session_wtp_numeric"
    avg_wtp        = df1[wtp_col].median() if wtp_col in df1.columns else 550
    best_clf       = max(clf_results, key=lambda k: clf_results[k]["accuracy"])
    best_cv        = clf_results[best_clf].get("cv_mean", clf_results[best_clf]["accuracy"])
    best_reg       = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_r2        = reg_results[best_reg]["r2"]
    sil_score      = max(silhouettes)

    cols = st.columns(6)
    kpis = [
        ("Respondents",    f"{total_n:,}",            "Survey 1 sample",          GOLD),
        ("Interested",     f"{interested_pct:.1f}%",  f"+ {maybe_pct:.1f}% Maybe",TEAL),
        ("Median WTP",     f"₹{int(avg_wtp):,}",      "Per session",              ORANGE),
        ("Best CV Score",  f"{best_cv*100:.1f}%",     f"{best_clf}",              INDIGO),
        ("Reg R²",         f"{best_r2:.3f}",           f"{best_reg}",              ROSE),
        ("Clusters (K)",   f"{best_k}",                f"Silhouette {sil_score:.3f}", TEAL),
    ]
    for col, (label, value, delta, color) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ART FORM IMAGES ───────────────────────────────────────
    section_header("The AURA Experience — Art Forms in the Pod", GOLD)
    art_image_grid(
        [IMAGES["mandala"], IMAGES["watercolour"], IMAGES["pottery"],
         IMAGES["sketching"], IMAGES["calligraphy"]],
        captions=["Mandala", "Watercolour", "Clay Pottery", "Sketching", "Calligraphy"],
        height=130
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── INTEREST + MODEL COMPARISON ───────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        section_header("Interest Distribution", GOLD)
        interest_counts = df1["aura_interest_label"].value_counts()
        colors_map = {"Interested": TEAL, "Maybe": GOLD, "Not_Interested": ROSE}
        fig_pie = go.Figure(go.Pie(
            labels=interest_counts.index, values=interest_counts.values,
            hole=0.6,
            marker_colors=[colors_map.get(l, ORANGE) for l in interest_counts.index],
            textinfo="percent+label", textfont=dict(size=10, color=INK),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig_pie.update_layout(height=300, showlegend=False,
            annotations=[dict(text=f"<b>{total_n:,}</b><br>Total",
                              x=0.5, y=0.5, font=dict(size=13, color=INK), showarrow=False)])
        st.plotly_chart(fig_pie, width="stretch")

    with c2:
        section_header("WTP Distribution", TEAL)
        if wtp_col in df1.columns:
            fig_hist = go.Figure(go.Histogram(
                x=df1[wtp_col].dropna(), nbinsx=20,
                marker=dict(color=GOLD, opacity=0.85,
                            line=dict(color=ORANGE, width=0.5)),
                hovertemplate="₹%{x}<br>Count: %{y}<extra></extra>"
            ))
            fig_hist.add_vline(x=avg_wtp, line_dash="dash", line_color=TEAL,
                               annotation_text=f"Median ₹{int(avg_wtp):,}",
                               annotation_font_color=TEAL)
            fig_hist.update_layout(height=300, xaxis_title="WTP (₹/session)",
                                   yaxis_title="Respondents", showlegend=False)
            st.plotly_chart(fig_hist, width="stretch")

    with c3:
        section_header("Model Accuracy", INDIGO)
        model_names = list(clf_results.keys())
        accs = [clf_results[m]["accuracy"] * 100 for m in model_names]
        cvs  = [(clf_results[m].get("cv_mean", clf_results[m]["accuracy"])) * 100 for m in model_names]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Test Accuracy", x=model_names, y=accs,
                                  marker_color=GOLD, text=[f"{a:.1f}%" for a in accs],
                                  textposition="outside"))
        fig_bar.add_trace(go.Bar(name="CV Mean", x=model_names, y=cvs,
                                  marker_color=TEAL, text=[f"{c:.1f}%" for c in cvs],
                                  textposition="outside"))
        fig_bar.update_layout(barmode="group", height=300, yaxis_title="Accuracy (%)",
                               yaxis_range=[0, 115], legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_bar, width="stretch")

    # ── BUSINESS CONTEXT IMAGES ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Business Context — India Urban Market", ORANGE)

    c_img1, c_img2 = st.columns(2)
    with c_img1:
        art_image_banner(IMAGES["urban_art"], height=120,
                          overlay_text="India's Wellness-Creativity Gap",
                          overlay_sub="Rising stress · Scarce accessible art spaces")
    with c_img2:
        art_image_banner(IMAGES["studio"], height=120,
                          overlay_text="The AURA Pod Concept",
                          overlay_sub="Mall · Co-working · Transit hubs · ₹200–₹1,200/session")

    # ── PIPELINE ──────────────────────────────────────────────
    section_header("4-Layer Analytics Pipeline", TEAL)
    layers = [
        (GOLD,   "01 Descriptive",  "WHO are they?",
         "Demographics · City breakdown · Art preferences · Stress profiles"),
        (TEAL,   "02 Diagnostic",   "WHY interested?",
         "Funnel · Cross-tabs · Correlation heatmap · Barrier analysis"),
        (ORANGE, "03 Predictive",   "WHAT will they do?",
         "Decision Tree · K-Means · RFM · Hierarchical · RF · XGB · LR · Apriori · ARIMA"),
        (ROSE,   "04 Prescriptive", "WHAT should AURA do?",
         "A/B pricing simulator · Launch sequence · City priority · Channel mix"),
    ]
    pcols = st.columns(4)
    for col, (color, title, q, desc) in zip(pcols, layers):
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{SURFACE2},{SURFACE3});
            border:1px solid rgba(255,255,255,0.06);border-top:3px solid {color};
            border-radius:10px;padding:18px;min-height:180px;">
                <div style="color:{color};font-size:10px;letter-spacing:0.15em;
                text-transform:uppercase;font-weight:700;margin-bottom:8px;">{title}</div>
                <div style="color:{INK};font-size:13px;font-weight:600;margin-bottom:10px;">{q}</div>
                <div style="color:{MUTED};font-size:11px;line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    info_card(
        "Key Finding",
        f"<b>{interested_pct:.1f}%</b> of 2,000 Indian consumers express interest in AURA, "
        f"with a median WTP of <b>₹{int(avg_wtp):,}/session</b>. Best classification model achieves "
        f"<b>{best_cv*100:.1f}% cross-validated accuracy</b>. K-Means identifies "
        f"<b>{best_k} distinct personas</b> (Silhouette = {sil_score:.3f}). "
        "AURA has a clear, data-backed path to launch.",
        GOLD
    )
