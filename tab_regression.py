"""tab_regression.py — WTP Regression + Live Predictor Widget"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, SURFACE3, INK, PALETTE,
                        page_header, section_header, info_card,
                        art_image_banner, colour_palette_strip, IMAGES)
from aura_data import encode_features, REG_FEATURES, ORDINAL_MAPS


def render(df1, df2, arm, wide, reg_models, reg_results, reg_feat_imp,
           X_test_reg, y_test_reg, reg_scaler):

    page_header(
        "WTP Regression — Session Pricing Prediction",
        "Three regression models predict a customer's Willingness to Pay (₹/session). "
        "Includes 5-fold cross-validation, residual analysis, and a live WTP predictor.",
        "Regression"
    )

    art_image_banner(
        IMAGES["acrylic"], height=170,
        overlay_text="WTP Regression — Session Pricing Prediction",
        overlay_sub="What will each customer pay? · RF · GBM · Linear Regression"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── MODEL COMPARISON TABLE ────────────────────────────────
    section_header("Regression Model Performance", GOLD)

    rows = []
    for name, res in reg_results.items():
        cv_str = f"{res['cv_r2_mean']:.3f} ± {res['cv_r2_std']:.3f}" if res.get("cv_r2_mean") else "N/A"
        rows.append({
            "Algorithm":    name,
            "RMSE (₹)":     f"₹{res['rmse']:,.0f}",
            "MAE (₹)":      f"₹{res['mae']:,.0f}",
            "R² Score":     f"{res['r2']:.4f}",
            "5-Fold CV R²": cv_str,
        })
    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare.set_index("Algorithm"), width="stretch")

    # R² comparison bar
    fig_r2 = go.Figure()
    colors = [GOLD, TEAL, ORANGE]
    model_names = list(reg_results.keys())
    r2_vals = [reg_results[m]["r2"] * 100 for m in model_names]
    cv_vals  = [(reg_results[m].get("cv_r2_mean") or reg_results[m]["r2"]) * 100 for m in model_names]

    fig_r2.add_trace(go.Bar(name="Test R²", x=model_names, y=r2_vals,
                             marker_color=colors[:len(model_names)],
                             text=[f"{v:.1f}%" for v in r2_vals], textposition="outside"))
    fig_r2.add_trace(go.Bar(name="CV R² (mean)", x=model_names, y=cv_vals,
                             marker_color=[c for c in colors[:len(model_names)]],
                             opacity=0.5,
                             text=[f"{v:.1f}%" for v in cv_vals], textposition="outside"))
    fig_r2.update_layout(
        barmode="group", title="R² Comparison: Test vs Cross-Validated",
        yaxis_title="R² (%)", height=300, legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_r2, width="stretch")

    # ── ACTUAL vs PREDICTED ───────────────────────────────────
    section_header("Actual vs Predicted WTP", TEAL)

    best_model_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
    res_best = reg_results[best_model_name]

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=res_best["y_test"], y=res_best["y_pred"],
        mode="markers",
        marker=dict(color=GOLD, size=5, opacity=0.6,
                    line=dict(color=SURFACE2, width=0.5)),
        name="Predictions",
        hovertemplate="Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>"
    ))
    min_val = min(res_best["y_test"].min(), res_best["y_pred"].min())
    max_val = max(res_best["y_test"].max(), res_best["y_pred"].max())
    fig_avp.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", line=dict(color=TEAL, dash="dash", width=2),
        name="Perfect Prediction"
    ))
    fig_avp.update_layout(
        title=f"Actual vs Predicted WTP — {best_model_name} (R²={res_best['r2']:.3f})",
        xaxis_title="Actual WTP (₹)", yaxis_title="Predicted WTP (₹)",
        height=380
    )
    st.plotly_chart(fig_avp, width="stretch")

    # ── RESIDUALS ─────────────────────────────────────────────
    section_header("Residual Analysis", ORANGE)

    residuals = res_best["y_pred"] - res_best["y_test"]
    c1, c2 = st.columns(2)

    with c1:
        fig_res = go.Figure(go.Scatter(
            x=res_best["y_pred"], y=residuals,
            mode="markers",
            marker=dict(color=ORANGE, size=4, opacity=0.6),
            hovertemplate="Predicted: ₹%{x:,.0f}<br>Residual: ₹%{y:,.0f}<extra></extra>"
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color=TEAL)
        fig_res.update_layout(
            title="Residuals vs Predicted (Random scatter = good)",
            xaxis_title="Predicted (₹)", yaxis_title="Residual (₹)", height=300
        )
        st.plotly_chart(fig_res, width="stretch")

    with c2:
        fig_hist = go.Figure(go.Histogram(
            x=residuals, nbinsx=30,
            marker_color=INDIGO, opacity=0.8,
            hovertemplate="Residual: %{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        fig_hist.update_layout(
            title="Residual Distribution (should be ~Normal, centred at 0)",
            xaxis_title="Residual (₹)", yaxis_title="Count", height=300
        )
        st.plotly_chart(fig_hist, width="stretch")

    # ── FEATURE IMPORTANCE ─────────────────────────────────────
    section_header("What Drives WTP? — Feature Importance", GOLD)

    fig_imp = go.Figure(go.Bar(
        x=reg_feat_imp["importance"].head(10),
        y=reg_feat_imp["feature"].head(10),
        orientation="h",
        marker=dict(
            color=reg_feat_imp["importance"].head(10),
            colorscale="YlOrRd",
            showscale=False
        ),
        text=[f"{v:.3f}" for v in reg_feat_imp["importance"].head(10)],
        textposition="outside"
    ))
    fig_imp.update_layout(
        title="Top 10 WTP Predictors — Random Forest Feature Importance",
        height=380, xaxis_title="Importance",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_imp, width="stretch")

    # ── LIVE WTP PREDICTOR ────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{SURFACE2},{SURFACE3});
    border:1px solid {GOLD}44;border-radius:12px;padding:28px;margin:24px 0 8px;">
        <div style="color:{GOLD};font-size:14px;font-weight:800;letter-spacing:0.06em;
        text-transform:uppercase;margin-bottom:6px;">🎯 Live WTP Predictor</div>
        <div style="color:{MUTED};font-size:13px;margin-bottom:20px;">
            Adjust the sliders to profile a customer and get an instant WTP prediction.
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.selectbox("Income Bracket", ["Below_25k", "25k_50k", "50k_1L", "1L_2L", "Above_2L"],
                               index=2, key="pred_income")
        age = st.selectbox("Age Group", ["18_24", "25_34", "35_44", "45_54", "55_plus"],
                           index=1, key="pred_age")
        art_exp = st.selectbox("Art Experience", ["Complete_Beginner", "Curious_Beginner",
                                                    "Casual_Hobbyist", "Regular_Hobbyist", "Advanced"],
                               index=2, key="pred_art")

    with col2:
        tech = st.slider("Tech Comfort (1–5)", 1, 5, 3, key="pred_tech")
        insta = st.slider("Instagram Influence (1–5)", 1, 5, 3, key="pred_insta")
        price_sens = st.slider("Price Sensitivity (1=low, 5=high)", 1, 5, 3, key="pred_ps")

    with col3:
        social = st.selectbox("Social Sharing", ["Keep_Private", "Close_Circle",
                                                  "Probably_Post", "Definitely_Post"],
                              index=2, key="pred_social")
        city_tier = st.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"], index=0, key="pred_city")
        leisure_spend = st.slider("Monthly Leisure Spend (₹)", 500, 10000, 3000, step=500, key="pred_leisure")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔮 Predict WTP Now", key="predict_wtp_btn"):
        input_dict = {
            "income_bracket":            income,
            "art_experience_level":      art_exp,
            "subscription_count":        "1_2",
            "tech_comfort_score":        tech,
            "city_tier":                 city_tier,
            "social_sharing_propensity": social,
            "instagram_influence_score": insta,
            "monthly_leisure_spend":     leisure_spend,
            "price_sensitivity_score":   price_sens,
            "age_group":                 age,
            "online_exp_purchase_freq":  "Occasionally",
            "creative_self_identity":    "Somewhat_Creative",
            "recommend_likelihood":      7,
            "visit_frequency_intent":    "Occasionally",
        }

        df_input = pd.DataFrame([input_dict])
        X_input = encode_features(df_input, REG_FEATURES).fillna(0)

        predictions = {}
        for mname, model in reg_models.items():
            if isinstance(model, tuple):
                m, sc = model
                pred = m.predict(sc.transform(X_input))[0]
            else:
                pred = model.predict(X_input)[0]
            predictions[mname] = max(100, int(round(pred, -1)))

        # Display results
        pred_cols = st.columns(len(predictions) + 1)
        for col, (mname, pred) in zip(pred_cols, predictions.items()):
            with col:
                st.markdown(f"""
                <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
                border-top:2px solid {GOLD};border-radius:8px;padding:20px;text-align:center;">
                    <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;
                    text-transform:uppercase;margin-bottom:8px;">{mname}</div>
                    <div style="color:{GOLD};font-size:32px;font-weight:800;">₹{pred:,}</div>
                    <div style="color:{MUTED};font-size:11px;margin-top:4px;">per session</div>
                </div>
                """, unsafe_allow_html=True)

        # Ensemble average
        avg_pred = int(round(np.mean(list(predictions.values())), -1))
        with pred_cols[-1]:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{GOLD}22,{TEAL}22);
            border:1px solid {GOLD}66;border-top:2px solid {TEAL};
            border-radius:8px;padding:20px;text-align:center;">
                <div style="color:{TEAL};font-size:10px;letter-spacing:0.12em;
                text-transform:uppercase;margin-bottom:8px;">Ensemble Avg</div>
                <div style="color:{TEAL};font-size:32px;font-weight:800;">₹{avg_pred:,}</div>
                <div style="color:{MUTED};font-size:11px;margin-top:4px;">recommended price</div>
            </div>
            """, unsafe_allow_html=True)

        # Pricing recommendation
        if avg_pred < 300:
            tier = "Below_200 / 200_400 bracket — offer intro sessions at ₹199"
            strategy = "🟡 Price-sensitive segment. Lead with a free trial or heavy first-session discount."
        elif avg_pred < 600:
            tier = "200_400 / 400_700 bracket — standard tier pricing"
            strategy = "🟢 Core AURA segment. Standard pricing with Buy-3-Get-1 bundle."
        elif avg_pred < 1000:
            tier = "700_1200 bracket — premium segment"
            strategy = "🟢 Premium customer. Offer monthly passes and exclusive time slots."
        else:
            tier = "Above_1200 bracket — ultra-premium"
            strategy = "🟢 Corporate/Serious Hobbyist. Push corporate packages and unlimited plans."

        info_card(
            "Pricing Recommendation",
            f"WTP bracket: <b>{tier}</b><br>{strategy}",
            TEAL
        )

    info_card(
        "Regression Insight",
        f"<b>{best_model_name}</b> achieves R²={res_best['r2']:.3f} — explaining "
        f"{res_best['r2']*100:.1f}% of WTP variance. Income bracket and art experience level are the "
        "strongest predictors, confirming that AURA should target Tier-1 city professionals with "
        "existing creative hobbies for its ₹700–₹1,200 premium sessions.",
        GOLD
    )

    st.download_button(
        "⬇ Download Predictions CSV",
        pd.DataFrame({"actual": res_best["y_test"], "predicted": res_best["y_pred"]}).to_csv(index=False),
        "aura_wtp_predictions.csv", "text/csv"
    )
