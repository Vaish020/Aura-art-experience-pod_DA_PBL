"""tab_classification.py — Classification with ROC Curves, CV, Business Interpretation"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, LAVENDER, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE,
                        page_header, section_header, info_card)
from aura_data import CLF_FEATURES


def render(df1, df2, arm, wide, clf_models, clf_results, clf_feat_imp,
           X_test_clf, y_test_clf, X_train_clf, y_train_clf):

    page_header(
        "Interest Classification",
        "Three models (Random Forest · XGBoost · Logistic Regression) predict whether a consumer will be "
        "Interested, Maybe, or Not Interested in AURA — with 5-fold cross-validation for robust evaluation.",
        "Classification"
    )

    # ── MODEL COMPARISON TABLE ────────────────────────────────
    section_header("Model Performance Comparison — All Algorithms", GOLD)

    rows = []
    for name, res in clf_results.items():
        cv_str = f"{res['cv_mean']*100:.1f}% ± {res['cv_std']*100:.1f}%" if res.get("cv_mean") else "N/A"
        rows.append({
            "Algorithm":      name,
            "Accuracy":       f"{res['accuracy']*100:.1f}%",
            "Precision":      f"{res['precision']*100:.1f}%",
            "Recall":         f"{res['recall']*100:.1f}%",
            "F1-Score":       f"{res['f1']*100:.1f}%",
            "5-Fold CV":      cv_str,
        })
    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare.set_index("Algorithm"), width="stretch")

    st.download_button(
        "⬇ Download Model Comparison CSV",
        df_compare.to_csv(index=False),
        "aura_model_comparison.csv",
        "text/csv"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CROSS-VALIDATION BOX PLOT ─────────────────────────────
    section_header("5-Fold Cross-Validation Stability", TEAL)
    fig_cv = go.Figure()
    colors = [GOLD, TEAL, ORANGE]
    for (name, res), color in zip(clf_results.items(), colors):
        scores = res.get("cv_scores", [])
        if scores:
            fig_cv.add_trace(go.Box(
                y=[s * 100 for s in scores],
                name=name,
                marker_color=color,
                boxmean="sd",
                hovertemplate=f"<b>{name}</b><br>CV Accuracy: %{{y:.2f}}%<extra></extra>"
            ))
    fig_cv.update_layout(
        title="Cross-Validation Accuracy Distribution (5 folds)",
        yaxis_title="Accuracy (%)", height=320,
        showlegend=True
    )
    st.plotly_chart(fig_cv, width="stretch")

    info_card(
        "Why Cross-Validation Matters",
        "A single train/test split can be lucky or unlucky. 5-fold CV trains on 5 different splits "
        "and averages the result — giving a robust, unbiased estimate of real-world model performance. "
        "Narrow box = stable model. Wide box = model sensitivity to data split.",
        TEAL
    )

    # ── ROC CURVES ────────────────────────────────────────────
    section_header("ROC Curves — All Models (One-vs-Rest)", INDIGO)

    class_names = ["Not Interested", "Maybe", "Interested"]
    model_colors = {
        "Random Forest":      GOLD,
        "XGBoost":            TEAL,
        "Logistic Regression": ORANGE,
    }
    roc_cols = st.columns(len(clf_results))

    for col, (model_name, res) in zip(roc_cols, clf_results.items()):
        with col:
            y_prob = res.get("y_prob")
            if y_prob is None:
                st.info(f"{model_name}: probabilities unavailable")
                continue

            y_bin = label_binarize(y_test_clf, classes=[0, 1, 2])
            color = model_colors.get(model_name, GOLD)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color=MUTED, width=1),
                name="Random Classifier",
                showlegend=False
            ))

            cls_colors = [ROSE, ORANGE, TEAL]
            for i, (cls_name, cls_color) in enumerate(zip(class_names, cls_colors)):
                if i >= y_bin.shape[1]:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=f"{cls_name} (AUC={roc_auc:.2f})",
                    line=dict(color=cls_color, width=2),
                    hovertemplate=f"FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra>{cls_name}</extra>"
                ))

            fig_roc.update_layout(
                title=f"{model_name}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=340,
                legend=dict(font=dict(size=9), x=0.4, y=0.1)
            )
            st.plotly_chart(fig_roc, width="stretch")

    # ── CONFUSION MATRICES WITH BUSINESS INTERPRETATION ───────
    section_header("Confusion Matrices + Business Cost Analysis", ROSE)

    cm_labels = ["Not Interested", "Maybe", "Interested"]
    business_cost = {
        (2, 0): "⚠️ Missed sale — 'Interested' misclassified as 'Not Interested'. Lost customer, wasted no resource but missed revenue.",
        (2, 1): "🟡 Partial miss — 'Interested' sent to nurture instead of direct offer. Delayed conversion.",
        (1, 2): "💸 Over-spend — 'Maybe' treated as 'Interested'. Wasted premium outreach budget.",
        (0, 2): "❌ High cost error — 'Not Interested' receives premium offer. Budget waste + brand irritation.",
        (0, 1): "🟡 Minor — 'Not Interested' added to nurture list. Low cost, low impact.",
        (1, 0): "🟡 Missed nurture — 'Maybe' excluded entirely. Lost potential conversion.",
    }

    for model_name, res in clf_results.items():
        cm = res["cm"]
        with st.expander(f"🔍 {model_name} — Confusion Matrix & Business Impact", expanded=(model_name == "Random Forest")):
            col_cm, col_interp = st.columns([1, 1.3])
            with col_cm:
                fig_cm = go.Figure(go.Heatmap(
                    z=cm,
                    x=[f"Pred: {l}" for l in cm_labels],
                    y=[f"Actual: {l}" for l in cm_labels],
                    colorscale="YlOrRd",
                    showscale=False,
                    text=cm,
                    texttemplate="<b>%{text}</b>",
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
                ))
                fig_cm.update_layout(
                    title=f"{model_name} — Confusion Matrix",
                    height=320,
                    xaxis_tickangle=-20
                )
                st.plotly_chart(fig_cm, width="stretch")

            with col_interp:
                st.markdown(f"<div style='color:{GOLD};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;margin-bottom:12px;'>Business Cost Interpretation</div>", unsafe_allow_html=True)
                total_errors = cm.sum() - np.trace(cm)
                for i in range(len(cm_labels)):
                    for j in range(len(cm_labels)):
                        if i != j and cm[i][j] > 0:
                            note = business_cost.get((i, j), "Off-diagonal error")
                            st.markdown(f"""
                            <div style="background:{SURFACE2};border-radius:6px;padding:10px 14px;
                            margin-bottom:8px;border-left:2px solid {ORANGE}44;">
                                <div style="color:{INK};font-size:12px;font-weight:600;margin-bottom:4px;">
                                    Actual {cm_labels[i]} → Predicted {cm_labels[j]}: <b>{cm[i][j]}</b> cases
                                </div>
                                <div style="color:{MUTED};font-size:11px;line-height:1.5;">{note}</div>
                            </div>
                            """, unsafe_allow_html=True)

                # Per-class precision/recall
                report = res["report"]
                st.markdown(f"<div style='color:{TEAL};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;margin-top:16px;margin-bottom:8px;'>Per-Class Metrics</div>", unsafe_allow_html=True)
                for cls in cm_labels:
                    cls_key = cls
                    if cls_key in report:
                        pr = report[cls_key]["precision"]
                        rc = report[cls_key]["recall"]
                        f1 = report[cls_key]["f1-score"]
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;padding:6px 0;
                        border-bottom:1px solid rgba(255,255,255,0.04);font-size:11px;">
                            <span style="color:{MUTED};">{cls}</span>
                            <span style="color:{INK};">P={pr:.2f} R={rc:.2f} F1={f1:.2f}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE ─────────────────────────────────────
    section_header("Feature Importance (Random Forest) — Business Insight", GOLD)

    top_n = 12
    top_feats = clf_feat_imp.head(top_n)

    business_insights = {
        "session_wtp":              "WTP reveals how much they value the experience — top signal",
        "visit_frequency_intent":   "Intent to visit frequently = committed customer",
        "income_bracket":           "Higher income = higher WTP and conversion",
        "participation_barrier":    "Removing barriers (cost, anxiety) directly drives conversion",
        "art_experience_level":     "Experienced hobbyists need less convincing",
        "creative_self_identity":   "Self-identified creatives convert at 2x rate",
        "social_sharing_propensity":"Sharers become brand ambassadors — high CLV",
        "tech_comfort_score":       "Tech-comfortable users engage with digital booking",
        "instagram_influence_score":"Instagram-influenced = social proof driven",
        "monthly_leisure_spend":    "Leisure budget signals discretionary spending capacity",
        "price_sensitivity_score":  "Price-sensitive segments need intro-offer campaigns",
        "age_group":                "Age shapes art form preference and session timing",
    }

    fig_imp = go.Figure(go.Bar(
        x=top_feats["importance"],
        y=top_feats["feature"],
        orientation="h",
        marker=dict(
            color=top_feats["importance"],
            colorscale="YlOrRd",
            showscale=False
        ),
        text=[f"{v:.3f}" for v in top_feats["importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig_imp.update_layout(
        title="Top 12 Features — Random Forest Importance",
        height=420,
        xaxis_title="Feature Importance",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_imp, width="stretch")

    # Business insight cards
    feat_cols = st.columns(3)
    for i, (_, row) in enumerate(top_feats.head(6).iterrows()):
        col = feat_cols[i % 3]
        insight = business_insights.get(row["feature"], "Key predictor of customer interest")
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
            border-radius:8px;padding:14px;margin-bottom:12px;">
                <div style="color:{GOLD};font-size:11px;font-weight:700;margin-bottom:6px;">
                    #{i+1} {row['feature'].replace('_',' ').title()}
                </div>
                <div style="color:{MUTED};font-size:11px;line-height:1.6;">{insight}</div>
                <div style="color:{TEAL};font-size:12px;font-weight:600;margin-top:8px;">
                    Importance: {row['importance']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    info_card(
        "Classification Insight",
        "Random Forest achieves the highest cross-validated accuracy. <b>WTP and visit frequency intent</b> are the "
        "top predictors — consumers who are willing to pay more and plan to visit frequently are almost certainly "
        "'Interested'. Participation barrier is the third most important feature, confirming that AURA's "
        "launch strategy should focus on removing the 'too expensive' and 'no guidance' barriers first.",
        GOLD
    )

    # Download predictions
    pred_df = pd.DataFrame({
        "actual": y_test_clf,
        "predicted_rf": clf_results["Random Forest"]["y_pred"]
    })
    st.download_button(
        "⬇ Download RF Predictions CSV",
        pred_df.to_csv(index=False),
        "aura_rf_predictions.csv",
        "text/csv"
    )
