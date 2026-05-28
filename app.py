"""
app.py — AURA India Group Analytics Dashboard (v2 — Full Course Coverage)
==========================================================================
Full ML analytics pipeline — descriptive to prescriptive.
Run: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="AURA India — Group PBL Dashboard",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AURA India — Art Experience Pod Analytics Dashboard"}
)

import pandas as pd
import numpy as np
from aura_theme import (GLOBAL_CSS, GOLD, TEAL, ORANGE, ROSE, INDIGO,
                        MUTED, SURFACE, SURFACE2, INK, BG, kpi_card)
from aura_data import (load_data, train_classification_models,
                       train_regression_models, train_clustering)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:24px 0 16px;border-bottom:1px solid rgba(240,192,64,0.15);margin-bottom:18px;">
        <div style="font-size:36px;font-weight:900;
        background:linear-gradient(135deg,{GOLD},{ORANGE},{ROSE});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        letter-spacing:0.08em;line-height:1;">AURA</div>
        <div style="font-size:9px;letter-spacing:0.22em;text-transform:uppercase;
        color:{TEAL};margin-top:6px;font-weight:600;">🎨 Art Experience Pod · India</div>
        <div style="font-size:9px;letter-spacing:0.15em;text-transform:uppercase;
        color:{MUTED};margin-top:3px;">Art Experience Pod · India · 2026</div>
        <div style="margin-top:12px;display:flex;gap:4px;">
            <div style="flex:1;height:4px;background:{GOLD};border-radius:2px;"></div>
            <div style="flex:1;height:4px;background:{TEAL};border-radius:2px;"></div>
            <div style="flex:1;height:4px;background:{ORANGE};border-radius:2px;"></div>
            <div style="flex:1;height:4px;background:{ROSE};border-radius:2px;"></div>
            <div style="flex:1;height:4px;background:{INDIGO};border-radius:2px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='color:{GOLD};font-size:10px;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:10px;font-weight:700;'>Dashboard</div>", unsafe_allow_html=True)

    tabs_config = [
        ("🏠", "Executive Summary",      "KPIs · Business Overview"),
        ("📊", "Overview",               "Descriptive Analysis"),
        ("🔍", "Diagnostic",             "Why Customers Convert"),
        ("🌳", "Decision Tree",          "Classification Rules"),
        ("🧩", "Clustering",             "K-Means Segmentation"),
        ("🌿", "RFM + Hierarchical",     "RFM + Dendrogram"),
        ("🔗", "Association Rules",      "Market Basket Analysis"),
        ("📈", "Regression",             "WTP Prediction"),
        ("📉", "Time Series + ARIMA",    "Forecasting"),
        ("💬", "Text Mining + Ethics",   "NLP + AI Ethics"),
        ("🤖", "Classification Models",  "RF + XGBoost + LR"),
        ("🚀", "Predict New",            "Score New Customers"),
        ("💡", "Prescriptive Strategy",  "Launch Decisions"),
    ]

    selected_tab = st.radio(
        "Select analysis:",
        [f"{icon} {name}" for icon, name, _ in tabs_config],
        label_visibility="collapsed"
    )

    for icon, name, sub in tabs_config:
        if selected_tab == f"{icon} {name}":
            st.markdown(f"<div style='color:{MUTED};font-size:10px;margin-top:-8px;margin-bottom:16px;padding-left:2px;'>{sub}</div>",
                        unsafe_allow_html=True)
            break

    st.markdown(f"""
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);">
        <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">Analysis Modules</div>
        <div style="font-size:10px;color:{MUTED};line-height:2.0;">
            🎨 Overview · Diagnostic<br>
            🌳 Decision Tree · Classification<br>
            🔗 Association Rules (Apriori)<br>
            🧩 K-Means · RFM · Hierarchical<br>
            📈 Regression · ARIMA · Forecast<br>
            💬 Text Mining · Sentiment<br>
            ⚖️ AI Ethics · ESG
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);">
        <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;">Dataset</div>
        <div style="font-size:10px;color:{TEAL};line-height:2.0;">
            ● Survey 1: 2,000 respondents<br>
            ● Survey 2: 1,314 deep profiles<br>
            ● ARM Transactions: 2,000 rows<br>
            ● Features: 81 columns
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:24px;color:{MUTED};font-size:9px;'>AURA India · Data Analytics · 2026</div>",
                unsafe_allow_html=True)

# ── LOAD DATA ──────────────────────────────────────────────────
with st.spinner("Loading AURA dataset..."):
    df1, df2, arm, wide = load_data()

# ── TRAIN MODELS ───────────────────────────────────────────────
with st.spinner("Training models..."):
    clf_models, clf_results, clf_feat_imp, X_test_clf, y_test_clf, X_train_clf, y_train_clf = \
        train_classification_models(df1)
    reg_models, reg_results, reg_feat_imp, X_test_reg, y_test_reg, reg_scaler = \
        train_regression_models(df1)
    km_model, df_clustered, km_scaler, best_k, k_range, inertias, silhouettes, pca = \
        train_clustering(df1)

# ── RENDER ─────────────────────────────────────────────────────
tab_name = selected_tab.split(" ", 1)[1]

if tab_name == "Executive Summary":
    import tab_summary
    tab_summary.render(df1, df2, arm, wide, clf_results, reg_results, best_k, silhouettes)

elif tab_name == "Overview":
    import tab_overview
    tab_overview.render(df1, df2, arm, wide)

elif tab_name == "Diagnostic":
    import tab_diagnostic
    tab_diagnostic.render(df1, df2, arm, wide)

elif tab_name == "Decision Tree":
    import tab_decision_tree
    tab_decision_tree.render(df1, df2, arm, wide)

elif tab_name == "Clustering":
    import tab_clustering
    tab_clustering.render(df1, df2, arm, wide,
                          km_model, df_clustered, km_scaler,
                          best_k, k_range, inertias, silhouettes, pca)

elif tab_name == "RFM + Hierarchical":
    import tab_hierarchical
    tab_hierarchical.render(df1, df2, arm, wide)

elif tab_name == "Association Rules":
    import tab_arm
    tab_arm.render(df1, df2, arm, wide)

elif tab_name == "Regression":
    import tab_regression
    tab_regression.render(df1, df2, arm, wide,
                          reg_models, reg_results, reg_feat_imp,
                          X_test_reg, y_test_reg, reg_scaler)

elif tab_name == "Time Series + ARIMA":
    import tab_timeseries
    tab_timeseries.render(df1, df2, arm, wide)

elif tab_name == "Text Mining + Ethics":
    import tab_text_ethics
    tab_text_ethics.render(df1, df2, arm, wide)

elif tab_name == "Classification Models":
    import tab_classification
    tab_classification.render(df1, df2, arm, wide,
                              clf_models, clf_results, clf_feat_imp,
                              X_test_clf, y_test_clf, X_train_clf, y_train_clf)

elif tab_name == "Predict New":
    import tab_predict
    tab_predict.render(df1, df2, arm, wide,
                       clf_models, reg_models, km_model, km_scaler)

elif tab_name == "Prescriptive Strategy":
    import tab_prescriptive
    tab_prescriptive.render(df1, df2, arm, wide,
                            clf_models, reg_models, km_model, km_scaler,
                            df_clustered, best_k)
