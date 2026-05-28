"""tab_predict.py — New Customer Scoring: Upload CSV → Instant Predictions"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE,
                        page_header, section_header, info_card)
from aura_data import predict_new_customers, CLF_FEATURES, REG_FEATURES, CLU_FEATURES


def render(df1, df2, arm, wide, clf_models, reg_models, km_model, km_scaler):

    page_header(
        "Predict New Customers",
        "Upload a CSV of new survey respondents and get instant predictions: "
        "interest label, confidence score, predicted WTP, assigned persona, and recommended action.",
        "Score New Customers"
    )

    # ── TEMPLATE DOWNLOAD ─────────────────────────────────────
    section_header("Step 1 — Download Template CSV", GOLD)

    all_feat = list(set(CLF_FEATURES + REG_FEATURES + CLU_FEATURES))
    template_df = pd.DataFrame(columns=all_feat)

    # Add one example row
    example = {col: "" for col in all_feat}
    example.update({
        "income_bracket": "50k_1L",
        "age_group": "25_34",
        "city_tier": "Tier1",
        "art_experience_level": "Casual_Hobbyist",
        "tech_comfort_score": 3,
        "instagram_influence_score": 4,
        "monthly_leisure_spend": 3000,
        "price_sensitivity_score": 3,
        "session_wtp": "400_700",
        "social_sharing_propensity": "Probably_Post",
        "visit_frequency_intent": "Occasionally",
        "creative_self_identity": "Somewhat_Creative",
        "creativity_mindset": "Growth",
        "social_orientation": "Ambivert",
        "reward_preference": "Both",
        "decision_autonomy": "Fully_Independent",
        "participation_barrier": "No_Barrier",
        "recommend_likelihood": 7,
        "subscription_count": "1_2",
        "online_exp_purchase_freq": "Occasionally",
    })
    template_df = pd.concat([template_df, pd.DataFrame([example])], ignore_index=True)

    st.download_button(
        "⬇ Download Prediction Template CSV",
        template_df.to_csv(index=False),
        "aura_prediction_template.csv",
        "text/csv",
        help="Fill in this template with new respondent data, then upload below"
    )

    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
    border-left:3px solid {GOLD};border-radius:8px;padding:16px 20px;margin:12px 0 24px;">
        <div style="color:{GOLD};font-size:11px;font-weight:700;letter-spacing:0.1em;
        text-transform:uppercase;margin-bottom:8px;">Column Requirements</div>
        <div style="color:{MUTED};font-size:12px;line-height:1.8;">
            The template has all required columns pre-filled with one example row.
            Fill in your respondents using the same categorical values (e.g. income_bracket = "50k_1L",
            city_tier = "Tier1", etc.). Numeric fields (tech_comfort_score 1–5) should be integers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── UPLOAD ────────────────────────────────────────────────
    section_header("Step 2 — Upload New Respondent Data", TEAL)

    uploaded = st.file_uploader(
        "Upload CSV with new customers",
        type=["csv"],
        help="Must have the same columns as the template"
    )

    if uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            st.success(f"✓ Loaded {len(df_new):,} new respondents · {len(df_new.columns)} columns")
            st.dataframe(df_new.head(5), width="stretch")

            with st.spinner("Running prediction pipeline..."):
                best_clf = max(clf_models, key=lambda k: True)  # use first (RF)
                clf_model = clf_models.get("Random Forest", list(clf_models.values())[0])
                reg_model = reg_models.get("Random Forest", list(reg_models.values())[0])

                results = predict_new_customers(df_new, clf_model, reg_model, km_model, km_scaler)

            section_header("Prediction Results", GOLD)

            # Summary KPIs
            pred_counts = results["predicted_interest"].value_counts()
            avg_wtp_pred = results["predicted_wtp_inr"].mean()
            avg_conf = results["confidence_score"].mean()

            kpi_cols = st.columns(4)
            kpi_data = [
                ("Interested",    f"{pred_counts.get('Interested', 0)}",    f"{pred_counts.get('Interested', 0)/len(results)*100:.1f}%", TEAL),
                ("Maybe",         f"{pred_counts.get('Maybe', 0)}",         f"{pred_counts.get('Maybe', 0)/len(results)*100:.1f}%", GOLD),
                ("Not Interested",f"{pred_counts.get('Not Interested', 0)}",f"{pred_counts.get('Not Interested', 0)/len(results)*100:.1f}%", ROSE),
                ("Avg Pred WTP",  f"₹{int(avg_wtp_pred):,}",               f"Confidence {avg_conf:.0%}", ORANGE),
            ]
            for col, (label, val, delta, color) in zip(kpi_cols, kpi_data):
                from aura_theme import kpi_card
                with col:
                    st.markdown(kpi_card(label, val, delta, color), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Interest pie
            c1, c2 = st.columns([1, 2])
            with c1:
                fig_pie = go.Figure(go.Pie(
                    labels=pred_counts.index,
                    values=pred_counts.values,
                    hole=0.55,
                    marker_colors=[{"Interested": TEAL, "Maybe": GOLD, "Not Interested": ROSE}.get(l, ORANGE)
                                   for l in pred_counts.index],
                    textinfo="percent+label",
                    textfont=dict(size=10)
                ))
                fig_pie.update_layout(height=260, showlegend=False,
                                      title="Predicted Interest Split")
                st.plotly_chart(fig_pie, width="stretch")

            with c2:
                # WTP distribution
                fig_wtp = go.Figure(go.Histogram(
                    x=results["predicted_wtp_inr"],
                    nbinsx=15,
                    marker_color=GOLD,
                    opacity=0.85
                ))
                fig_wtp.add_vline(x=avg_wtp_pred, line_dash="dash", line_color=TEAL,
                                  annotation_text=f"Avg ₹{int(avg_wtp_pred):,}",
                                  annotation_font_color=TEAL)
                fig_wtp.update_layout(
                    title="Predicted WTP Distribution",
                    xaxis_title="Predicted WTP (₹)",
                    yaxis_title="Count", height=260
                )
                st.plotly_chart(fig_wtp, width="stretch")

            # Full results table
            section_header("Full Predictions Table", TEAL)
            display_cols = ["predicted_interest", "confidence_score", "predicted_wtp_inr",
                            "assigned_cluster", "recommended_action"]
            display_cols = [c for c in display_cols if c in results.columns]
            st.dataframe(results[display_cols], width="stretch")

            # Priority actions
            section_header("Priority Action List", ORANGE)
            high_prio = results[results["predicted_interest"] == "Interested"].sort_values(
                "predicted_wtp_inr", ascending=False)
            if len(high_prio) > 0:
                st.markdown(f"<div style='color:{TEAL};font-size:13px;margin-bottom:12px;'>"
                            f"<b>{len(high_prio)}</b> high-priority customers identified.</div>",
                            unsafe_allow_html=True)
                st.dataframe(high_prio[display_cols].head(20), width="stretch")

            # Download
            st.download_button(
                "⬇ Download Full Predictions CSV",
                results.to_csv(index=False),
                "aura_predictions_output.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure the CSV has the same column format as the template.")
    else:
        st.markdown(f"""
        <div style="background:{SURFACE2};border:2px dashed rgba(255,255,255,0.1);
        border-radius:10px;padding:48px;text-align:center;">
            <div style="font-size:32px;margin-bottom:12px;">📂</div>
            <div style="color:{MUTED};font-size:14px;">
                Upload a CSV to get instant customer predictions
            </div>
            <div style="color:{MUTED};font-size:12px;margin-top:8px;">
                Download the template above first if you don't have the right format
            </div>
        </div>
        """, unsafe_allow_html=True)

    info_card(
        "How Scoring Works",
        "The scoring pipeline runs three models sequentially: <b>Random Forest classifier</b> assigns "
        "interest label + confidence, <b>Random Forest regressor</b> predicts WTP in ₹, and "
        "<b>K-Means clustering</b> assigns the customer persona. Each output feeds into a rule-based "
        "recommended_action that tells your marketing team exactly what to do next.",
        TEAL
    )
