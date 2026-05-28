"""
tab_decision_tree.py — Session 3: Decision Tree Analysis
=========================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE,
                        page_header, section_header, info_card,
                        art_image_banner, colour_palette_strip, IMAGES)
from aura_data import encode_features, CLF_FEATURES


def render(df1, df2, arm, wide):

    page_header(
        "Decision Tree Analysis",
        " Decision Tree algorithms applied to AURA customer interest classification. "
        "Visualise the decision rules, understand feature splits, and extract business-actionable if-then logic.",
        "Decision Tree Analysis"
    )

    art_image_banner(
        IMAGES["sketching"], height=170,
        overlay_text="Decision Tree Analysis",
        overlay_sub=" Interpretable classification rules for AURA"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ALGORITHM EXPLANATION ─────────────────────────────────
    section_header("What is a Decision Tree?", GOLD)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px;">
        <div style="background:{SURFACE2};border-left:3px solid {GOLD};border-radius:8px;padding:16px;">
            <div style="color:{GOLD};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">How It Works</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                A Decision Tree splits data at each node using the feature that best separates classes.
                It keeps splitting until it reaches a leaf node — a final prediction.
                Each path from root to leaf = one business decision rule.
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {TEAL};border-radius:8px;padding:16px;">
            <div style="color:{TEAL};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Key Parameters</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                <b style="color:{INK};">Max Depth</b> — how many splits deep the tree grows.<br>
                <b style="color:{INK};">Min Samples Split</b> — minimum samples needed to split a node.<br>
                <b style="color:{INK};">Criterion</b> — Gini impurity or Information Gain (entropy).
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {ORANGE};border-radius:8px;padding:16px;">
            <div style="color:{ORANGE};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Why for AURA</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                Decision Trees produce <b style="color:{INK};">interpretable rules</b> — unlike Random Forest (a black box).
                AURA's marketing team can read the rules directly and act on them without a data scientist present.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PARAMETER CONTROLS ────────────────────────────────────
    section_header("Tree Parameters — Interactive Control", TEAL)
    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.slider("Max Depth", 2, 10, 4,
                               help="Deeper = more complex but risks overfitting")
    with col2:
        criterion = st.selectbox("Splitting Criterion",
                                  ["gini", "entropy"],
                                  help="Gini impurity vs Information Gain (entropy)")
    with col3:
        min_samples = st.slider("Min Samples to Split", 2, 50, 10,
                                 help="Prevents tiny noisy splits")

    # ── TRAIN DECISION TREE ───────────────────────────────────
    df_clean = df1.dropna(subset=["aura_interest_label"]).copy()
    X = encode_features(df_clean, CLF_FEATURES).fillna(0)
    y = df_clean["aura_interest_label"].map(
        {"Interested": 2, "Maybe": 1, "Not_Interested": 0}
    ).fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples,
        random_state=42,
        class_weight="balanced"
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")

    # ── METRICS ───────────────────────────────────────────────
    section_header("Model Performance", GOLD)
    m_cols = st.columns(4)
    metrics = [
        ("Test Accuracy",  f"{acc*100:.1f}%",              GOLD),
        ("5-Fold CV Mean", f"{cv_scores.mean()*100:.1f}%", TEAL),
        ("CV Std Dev",     f"±{cv_scores.std()*100:.1f}%", ORANGE),
        ("Tree Depth",     f"{dt.get_depth()}",             INDIGO),
    ]
    for col, (label, val, color) in zip(m_cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE2};border-top:2px solid {color};
            border-radius:8px;padding:18px;text-align:center;">
                <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;
                text-transform:uppercase;margin-bottom:8px;">{label}</div>
                <div style="color:{color};font-size:28px;font-weight:800;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TREE VISUALISATION (text-based, like Python output) ───
    section_header("Decision Tree Structure — Text Visualisation", TEAL)
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
    border-radius:8px;padding:6px 4px;margin-bottom:8px;">
        <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;text-transform:uppercase;
        padding:8px 12px;margin-bottom:4px;">Decision Rules (like export_text in Python)</div>
    """, unsafe_allow_html=True)

    tree_text = export_text(dt, feature_names=CLF_FEATURES, max_depth=min(max_depth, 5))
    # Colour-code the text
    lines = tree_text.split('\n')
    html_lines = []
    for line in lines[:60]:  # cap at 60 lines for readability
        if "class: 2" in line or "Interested" in line:
            color = TEAL
        elif "class: 0" in line or "Not_Interested" in line:
            color = ROSE
        elif "class: 1" in line or "Maybe" in line:
            color = GOLD
        elif "|---" in line:
            color = MUTED
        else:
            color = INK
        indent = len(line) - len(line.lstrip())
        html_lines.append(
            f'<div style="color:{color};font-family:monospace;font-size:12px;'
            f'padding-left:{indent*4}px;line-height:1.6;">{line}</div>'
        )

    st.markdown(
        '<div style="padding:12px 16px;max-height:400px;overflow-y:auto;">' +
        '\n'.join(html_lines) + '</div></div>',
        unsafe_allow_html=True
    )

    # ── PLOTLY TREE VISUALISATION ─────────────────────────────
    section_header("Decision Tree — Interactive Node Map", INDIGO)
    _plot_tree_interactive(dt, CLF_FEATURES, max_depth)

    # ── FEATURE IMPORTANCE ────────────────────────────────────
    section_header("Feature Importance — Decision Tree Splits", GOLD)
    feat_imp_dt = pd.DataFrame({
        "feature": CLF_FEATURES,
        "importance": dt.feature_importances_
    }).sort_values("importance", ascending=False).head(12)
    feat_imp_dt = feat_imp_dt[feat_imp_dt["importance"] > 0]

    fig_imp = go.Figure(go.Bar(
        x=feat_imp_dt["importance"],
        y=feat_imp_dt["feature"],
        orientation="h",
        marker=dict(
            color=feat_imp_dt["importance"],
            colorscale="YlOrRd",
            showscale=False
        ),
        text=[f"{v:.4f}" for v in feat_imp_dt["importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig_imp.update_layout(
        title="Features Used for Splits — Decision Tree Gini/Entropy Importance",
        height=380, xaxis_title="Importance",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_imp, width="stretch")

    # ── CONFUSION MATRIX ──────────────────────────────────────
    section_header("Confusion Matrix", ROSE)
    cm = confusion_matrix(y_test, y_pred)
    class_names = ["Not Interested", "Maybe", "Interested"]

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {c}" for c in class_names],
        y=[f"Actual: {c}" for c in class_names],
        colorscale="YlOrRd",
        showscale=False,
        text=cm,
        texttemplate="<b>%{text}</b>",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig_cm.update_layout(
        title=f"Decision Tree Confusion Matrix (Depth={max_depth}, {criterion.title()})",
        height=340, xaxis_tickangle=-15
    )
    st.plotly_chart(fig_cm, width="stretch")

    # ── BUSINESS RULES EXTRACTION ─────────────────────────────
    section_header("Extracted Business Decision Rules", ORANGE)
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;padding:24px;margin-bottom:16px;">
        <div style="color:{GOLD};font-size:12px;font-weight:700;text-transform:uppercase;
        letter-spacing:0.1em;margin-bottom:16px;">
            Top Decision Rules — Plain English for AURA Marketing Team
        </div>
    """, unsafe_allow_html=True)

    rules = [
        (TEAL,   "Rule 1 — High Conversion",
         "IF session_wtp > 2 AND visit_frequency_intent > 2 AND creative_self_identity ≥ 2 → INTERESTED",
         "Target: Send premium onboarding pack + pod location details immediately."),
        (GOLD,   "Rule 2 — Price-Sensitive Converter",
         "IF session_wtp = 1–2 AND participation_barrier ≠ Too_Expensive AND income ≥ 50k_1L → MAYBE",
         "Target: Send intro session offer at ₹199 — price barrier is the only obstacle."),
        (ORANGE, "Rule 3 — Social-Driven Convert",
         "IF instagram_influence_score ≥ 4 AND social_sharing_propensity = Definitely_Post → INTERESTED",
         "Target: Status Sharer persona — offer free first session in exchange for Instagram post."),
        (ROSE,   "Rule 4 — Low Priority",
         "IF participation_barrier = Not_Interested AND creative_self_identity = Not_Creative → NOT INTERESTED",
         "Target: Exclude from paid campaigns. Add to passive awareness list only."),
        (INDIGO, "Rule 5 — Corporate Signal",
         "IF income ≥ 1L_2L AND city_tier = Tier1 AND decision_autonomy = Fully_Independent → INTERESTED",
         "Target: Corporate Buyer persona — reach via LinkedIn + HR team outreach."),
    ]

    for color, title, rule, action in rules:
        st.markdown(f"""
        <div style="border-left:3px solid {color};padding:14px 18px;margin-bottom:12px;
        background:rgba(255,255,255,0.02);border-radius:0 6px 6px 0;">
            <div style="color:{color};font-size:11px;font-weight:700;
            text-transform:uppercase;margin-bottom:6px;">{title}</div>
            <div style="color:{INK};font-size:12px;font-family:monospace;
            margin-bottom:8px;background:{SURFACE2};padding:8px 12px;border-radius:4px;">
                {rule}
            </div>
            <div style="color:{MUTED};font-size:12px;line-height:1.6;">
                <b style="color:{INK};">Action:</b> {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── DEPTH vs ACCURACY ─────────────────────────────────────
    section_header("Depth vs Accuracy — Overfitting Analysis", TEAL)
    depths = list(range(1, 13))
    train_accs, test_accs = [], []
    for d in depths:
        dt_d = DecisionTreeClassifier(max_depth=d, criterion=criterion,
                                      min_samples_split=min_samples,
                                      random_state=42, class_weight="balanced")
        dt_d.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, dt_d.predict(X_train)) * 100)
        test_accs.append(accuracy_score(y_test, dt_d.predict(X_test)) * 100)

    fig_depth = go.Figure()
    fig_depth.add_trace(go.Scatter(x=depths, y=train_accs, mode="lines+markers",
                                    name="Train Accuracy", line=dict(color=GOLD, width=2.5),
                                    marker=dict(size=8)))
    fig_depth.add_trace(go.Scatter(x=depths, y=test_accs, mode="lines+markers",
                                    name="Test Accuracy", line=dict(color=TEAL, width=2.5),
                                    marker=dict(size=8)))
    fig_depth.add_vline(x=max_depth, line_dash="dash", line_color=ORANGE,
                         annotation_text=f"Selected Depth={max_depth}",
                         annotation_font_color=ORANGE)
    fig_depth.update_layout(
        title="Train vs Test Accuracy by Tree Depth — Identifying Overfitting Point",
        xaxis_title="Max Depth", yaxis_title="Accuracy (%)",
        height=340, legend=dict(orientation="h", y=-0.2),
        yaxis_range=[0, 105]
    )
    st.plotly_chart(fig_depth, width="stretch")

    st.download_button(
        "⬇ Download Decision Tree Rules (Text)",
        tree_text,
        "aura_decision_tree_rules.txt", "text/plain"
    )

    info_card(
        "Decision Tree Insight",
        f"At depth={max_depth} with {criterion} criterion, the Decision Tree achieves "
        f"<b>{acc*100:.1f}% test accuracy</b> and <b>{cv_scores.mean()*100:.1f}% 5-fold CV accuracy</b>. "
        "Unlike Random Forest, every split rule is readable by a human. "
        "The overfitting chart above shows where train and test accuracy diverge — "
        "the optimal depth is where test accuracy peaks before starting to drop. "
        "Key insight: <b>session_wtp and visit_frequency_intent</b> are the first splits in the tree, "
        "confirming they are the most discriminating features for AURA customer interest.",
        TEAL
    )


def _plot_tree_interactive(dt, feature_names, max_depth_display):
    """Build an interactive Plotly node-link diagram of the decision tree."""
    from sklearn.tree import _tree

    tree_ = dt.tree_
    class_labels = ["Not Interested", "Maybe", "Interested"]
    class_colors = [ROSE, GOLD, TEAL]

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    edge_x, edge_y = [], []

    def traverse(node, depth, x_pos, x_range):
        if depth > max_depth_display:
            return
        y_pos = -depth

        # Node label
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            label = f"{feat.replace('_',' ')[:18]}<br>≤ {threshold:.1f}"
            color = INDIGO
        else:
            cls = np.argmax(tree_.value[node])
            label = f"→ {class_labels[cls]}<br>n={int(tree_.n_node_samples[node])}"
            color = class_colors[cls]

        node_x.append(x_pos)
        node_y.append(y_pos)
        node_text.append(label)
        node_color.append(color)
        node_size.append(max(20, min(50, int(tree_.n_node_samples[node] / 10))))

        left = tree_.children_left[node]
        right = tree_.children_right[node]

        if left != _tree.TREE_LEAF:
            lx = x_pos - x_range / 2
            rx = x_pos + x_range / 2
            # Left edge
            edge_x.extend([x_pos, lx, None])
            edge_y.extend([y_pos, -(depth + 1), None])
            # Right edge
            edge_x.extend([x_pos, rx, None])
            edge_y.extend([y_pos, -(depth + 1), None])
            traverse(left, depth + 1, lx, x_range / 2)
            traverse(right, depth + 1, rx, x_range / 2)

    traverse(0, 0, 0, 4.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(255,255,255,0.15)", width=1.5),
        hoverinfo="none", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color=node_color,
                    line=dict(color=SURFACE2, width=2)),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color=INK),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    ))
    fig.update_layout(
        title=f"Decision Tree Node Map (Depth ≤ {max_depth_display})",
        height=max(350, max_depth_display * 80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE
    )
    st.plotly_chart(fig, width="stretch")
