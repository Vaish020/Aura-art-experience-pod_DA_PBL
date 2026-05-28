"""tab_arm.py — Association Rule Mining: Apriori with full Support/Confidence/Lift display"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE,
                        page_header, section_header, info_card,
                        art_image_banner, colour_palette_strip, IMAGES)
from aura_data import run_arm


def render(df1, df2, arm, wide):

    page_header(
        "Association Rule Mining",
        "Apriori algorithm discovers which art experiences, products, and session types naturally "
        "bundle together — powering AURA's upsell, cross-sell, and experience packaging strategy.",
        "05 — Association Rules (Apriori)"
    )

    art_image_banner(
        IMAGES["paint_brushes"], height=100,
        overlay_text="Association Rule Mining",
        overlay_sub="Apriori · What experiences bundle together?"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── PARAMETER CONTROLS ────────────────────────────────────
    section_header("Algorithm Parameters", GOLD)

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        min_support = st.slider("Min Support", 0.01, 0.3, 0.05, 0.01,
                                 help="Minimum fraction of transactions containing the itemset")
    with col_p2:
        min_confidence = st.slider("Min Confidence", 0.1, 0.9, 0.3, 0.05,
                                    help="P(consequent | antecedent) threshold")
    with col_p3:
        min_lift = st.slider("Min Lift", 1.0, 5.0, 1.0, 0.1,
                              help="Lift > 1 means items co-occur more than by chance")
    with col_p4:
        top_n = st.slider("Top N Rules", 5, 50, 20, 5)

    # ── ALGORITHM EXPLANATION ─────────────────────────────────
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0 24px;">
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {GOLD};border-radius:8px;padding:16px;">
            <div style="color:{GOLD};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:8px;">Support</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.6;">
                P(A ∩ B) — How often A and B appear together in all transactions.
                <br><br><b style="color:{INK};">Support = {min_support:.0%}</b> means at least
                {int(min_support * len(arm)):,} of {len(arm):,} transactions contain this set.
            </div>
        </div>
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {TEAL};border-radius:8px;padding:16px;">
            <div style="color:{TEAL};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:8px;">Confidence</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.6;">
                P(B | A) — Given a customer chose A, how likely are they to also choose B?
                <br><br><b style="color:{INK};">Confidence = {min_confidence:.0%}</b> means
                {min_confidence:.0%} of A-buyers also bought B.
            </div>
        </div>
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {ORANGE};border-radius:8px;padding:16px;">
            <div style="color:{ORANGE};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:8px;">Lift</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.6;">
                Confidence / P(B) — How much more likely is B given A, vs. buying B randomly?
                <br><br><b style="color:{INK};">Lift > 1</b> = positive association. Lift = {min_lift:.1f} threshold set.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── RUN APRIORI ───────────────────────────────────────────
    with st.spinner("Running Apriori algorithm..."):
        rules_df, error_msg = run_arm(arm, min_support, min_confidence, min_lift)

    if error_msg:
        st.warning(f"⚠️ {error_msg}")
        st.info("Showing simulated ARM output for demonstration. Install mlxtend for live results: `pip install mlxtend`")
        rules_df = _generate_demo_rules()

    if rules_df is None or len(rules_df) == 0:
        st.warning("No rules found at these thresholds. Try lowering min_support or min_confidence.")
        return

    rules_top = rules_df.head(top_n).copy()

    # ── RULES TABLE ───────────────────────────────────────────
    section_header(f"Top {top_n} Association Rules by Lift", TEAL)

    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    display_cols = [c for c in display_cols if c in rules_top.columns]

    rules_display = rules_top[display_cols].copy()
    if "support" in rules_display.columns:
        rules_display["support"] = rules_display["support"].apply(lambda x: f"{x:.3f} ({x*100:.1f}%)")
    if "confidence" in rules_display.columns:
        rules_display["confidence"] = rules_display["confidence"].apply(lambda x: f"{x:.3f} ({x*100:.1f}%)")
    if "lift" in rules_display.columns:
        rules_display["lift"] = rules_display["lift"].apply(lambda x: f"{x:.3f}")

    rules_display.columns = ["If Customer Buys...", "They Also Buy...", "Support", "Confidence", "Lift"]
    st.dataframe(rules_display, width="stretch")

    st.download_button(
        "⬇ Download Association Rules CSV",
        rules_top.to_csv(index=False),
        "aura_association_rules.csv", "text/csv"
    )

    # ── SCATTER: SUPPORT vs CONFIDENCE coloured by LIFT ───────
    section_header("Support vs Confidence — Coloured by Lift", ORANGE)

    if all(c in rules_top.columns for c in ["support", "confidence", "lift"]):
        fig_scatter = px.scatter(
            rules_top,
            x="support", y="confidence",
            size="lift",
            color="lift",
            color_continuous_scale="YlOrRd",
            hover_data={"antecedents": True, "consequents": True,
                        "support": ":.3f", "confidence": ":.3f", "lift": ":.3f"},
            title="Association Rules — Support vs Confidence (bubble size = Lift)",
            labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, width="stretch")

    # ── TOP RULES BAR ──────────────────────────────────────────
    section_header("Top Rules by Lift — Visual", GOLD)

    if "lift" in rules_top.columns and "antecedents" in rules_top.columns:
        top_lift = rules_top.nlargest(min(15, len(rules_top)), "lift").copy()
        top_lift["rule_label"] = (
            top_lift["antecedents"].str[:20] + " → " + top_lift["consequents"].str[:20]
        )
        fig_bar = go.Figure(go.Bar(
            x=top_lift["lift"],
            y=top_lift["rule_label"],
            orientation="h",
            marker=dict(
                color=top_lift["lift"],
                colorscale="YlOrRd",
                showscale=False
            ),
            text=[f"Lift: {l:.2f}" for l in top_lift["lift"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Lift: %{x:.3f}<extra></extra>"
        ))
        fig_bar.add_vline(x=1, line_dash="dash", line_color=MUTED,
                          annotation_text="Baseline (Lift=1)", annotation_font_color=MUTED)
        fig_bar.update_layout(
            title="Top Rules by Lift (higher = stronger association)",
            height=max(300, len(top_lift) * 28),
            xaxis_title="Lift",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, width="stretch")

    # ── BUSINESS INTERPRETATION ───────────────────────────────
    section_header("Business Interpretation — AURA Bundling Strategy", TEAL)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;">
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {GOLD};border-radius:8px;padding:20px;">
            <div style="color:{GOLD};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:12px;">🎁 Product Bundling</div>
            <div style="color:{MUTED};font-size:13px;line-height:1.8;">
                Rules with high lift reveal natural product bundles. Customers who buy a
                <b style="color:{INK};">Mandala Kit</b> are strongly associated with also purchasing
                <b style="color:{INK};">Guided Sessions</b> — bundle these at a 10% discount to
                increase basket size.
            </div>
        </div>
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {TEAL};border-radius:8px;padding:20px;">
            <div style="color:{TEAL};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:12px;">🔄 Session Sequencing</div>
            <div style="color:{MUTED};font-size:13px;line-height:1.8;">
                ARM reveals session flow patterns. Customers who start with
                <b style="color:{INK};">Beginner Watercolour</b> frequently progress to
                <b style="color:{INK};">Heritage Art</b> — design a 3-session onboarding journey
                that naturally drives this progression.
            </div>
        </div>
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {ORANGE};border-radius:8px;padding:20px;">
            <div style="color:{ORANGE};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:12px;">🤝 Corporate Upsell</div>
            <div style="color:{MUTED};font-size:13px;line-height:1.8;">
                Corporate session buyers show strong association with
                <b style="color:{INK};">Gift Bundle purchases</b>. Train B2B sales team to
                proactively offer Diwali gift packs to corporate bookers — high confidence upsell.
            </div>
        </div>
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:3px solid {INDIGO};border-radius:8px;padding:20px;">
            <div style="color:{INDIGO};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-bottom:12px;">📱 Digital Cross-Sell</div>
            <div style="color:{MUTED};font-size:13px;line-height:1.8;">
                Social sharers (Instagram-active) who book one session are strongly associated
                with <b style="color:{INK};">subscription interest</b>. Trigger a subscription
                offer in the post-session WhatsApp message for this cohort.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_card(
        "ARM Business Insight",
        "Apriori ARM moves AURA from transactional thinking to <b>ecosystem thinking</b>. "
        "Instead of selling one session at a time, ARM reveals how customers naturally flow through "
        "the AURA experience — enabling cross-sells that feel helpful, not pushy. "
        f"The {len(rules_df)} discovered rules form the foundation of AURA's recommendation engine.",
        TEAL
    )


def _generate_demo_rules():
    """Fallback demo rules if mlxtend is unavailable."""
    data = {
        "antecedents": [
            "Mandala Kit", "Watercolour Session", "Beginner Session",
            "Heritage Art", "Corporate Booking", "Fluid Art", "Mandala Kit",
            "Digital Art Session", "Clay Pottery", "Photography Walk",
        ],
        "consequents": [
            "Guided Session", "Heritage Art Kit", "Mandala Session",
            "Premium Kit Purchase", "Gift Bundle", "Acrylic Session", "Sketchbook Add-on",
            "Social Sharing Pack", "Guided Session", "AURA Subscription",
        ],
        "support":    [0.18, 0.15, 0.22, 0.12, 0.08, 0.13, 0.17, 0.09, 0.11, 0.07],
        "confidence": [0.72, 0.65, 0.58, 0.71, 0.83, 0.61, 0.54, 0.67, 0.59, 0.74],
        "lift":       [3.2,  2.8,  2.1,  3.5,  4.1,  2.4,  2.0,  3.0,  2.6,  3.8],
    }
    return pd.DataFrame(data).sort_values("lift", ascending=False).reset_index(drop=True)
