"""tab_prescriptive.py — Prescriptive: A/B Pod Pricing Simulator + Geographic Heatmap + Launch Strategy"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, LAVENDER, MUTED,
                        SURFACE, SURFACE2, SURFACE3, INK, PALETTE,
                        page_header, section_header, info_card)
from aura_data import name_cluster, CLUSTER_NAMES, CLUSTER_STRATEGY


def render(df1, df2, arm, wide, clf_models, reg_models, km_model, km_scaler, df_clustered, best_k):

    page_header(
        "Prescriptive Analytics",
        "Translate all model outputs into actionable AURA launch decisions — "
        "A/B pod pricing simulator, geographic priority heatmap, segment launch sequencing, "
        "and channel-wise budget allocation.",
        "Prescriptive Strategy"
    )

    # ── A/B POD PRICING SIMULATOR ─────────────────────────────
    section_header("🔬 A/B Session Pricing Simulator", GOLD)

    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid {GOLD}44;border-radius:10px;padding:24px;margin-bottom:8px;">
        <div style="color:{GOLD};font-size:13px;font-weight:700;margin-bottom:4px;">
            Simulate Revenue & Conversion at Different Price Points
        </div>
        <div style="color:{MUTED};font-size:12px;margin-bottom:20px;">
            Compare two pricing scenarios (A and B) — see which maximises revenue per pod per day.
        </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        st.markdown(f"<div style='color:{GOLD};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:10px;'>Scenario A</div>", unsafe_allow_html=True)
        price_a   = st.slider("Price A (₹/session)", 100, 2000, 400, 50, key="pa")
        sessions_a = st.slider("Sessions/pod/day A", 4, 20, 8, 1, key="sa")
        conv_a    = st.slider("Est. Conversion Rate A (%)", 5, 90, 45, 5, key="ca")

    with col_b:
        st.markdown(f"<div style='color:{TEAL};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:10px;'>Scenario B</div>", unsafe_allow_html=True)
        price_b   = st.slider("Price B (₹/session)", 100, 2000, 700, 50, key="pb")
        sessions_b = st.slider("Sessions/pod/day B", 4, 20, 6, 1, key="sb")
        conv_b    = st.slider("Est. Conversion Rate B (%)", 5, 90, 30, 5, key="cb")

    with col_c:
        st.markdown(f"<div style='color:{MUTED};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:10px;'>Pod Settings</div>", unsafe_allow_html=True)
        n_pods    = st.slider("Number of Pods", 1, 20, 3, 1, key="npods")
        pod_cost  = st.slider("Daily Operating Cost/Pod (₹)", 2000, 20000, 8000, 500, key="pcost")
        days      = st.slider("Simulation Days", 7, 365, 30, 7, key="simdays")

    st.markdown("</div>", unsafe_allow_html=True)

    # Calculations
    daily_rev_a  = price_a * sessions_a * (conv_a / 100) * n_pods
    daily_rev_b  = price_b * sessions_b * (conv_b / 100) * n_pods
    daily_cost   = pod_cost * n_pods
    daily_prof_a = daily_rev_a - daily_cost
    daily_prof_b = daily_rev_b - daily_cost
    total_rev_a  = daily_rev_a * days
    total_rev_b  = daily_rev_b * days
    total_prof_a = daily_prof_a * days
    total_prof_b = daily_prof_b * days
    breakeven_a  = daily_cost / (price_a * sessions_a * (conv_a / 100)) if (price_a * sessions_a * (conv_a / 100)) > 0 else 0
    breakeven_b  = daily_cost / (price_b * sessions_b * (conv_b / 100)) if (price_b * sessions_b * (conv_b / 100)) > 0 else 0

    # Results
    winner = "A" if total_prof_a > total_prof_b else "B"
    win_color = GOLD if winner == "A" else TEAL

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{win_color}11,{SURFACE2});
    border:1px solid {win_color}44;border-radius:10px;padding:20px;margin-top:16px;margin-bottom:20px;
    text-align:center;">
        <div style="color:{win_color};font-size:24px;font-weight:900;">
            Scenario {winner} Wins
        </div>
        <div style="color:{MUTED};font-size:13px;margin-top:6px;">
            Higher {days}-day profit by ₹{abs(total_prof_a - total_prof_b):,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    res_cols = st.columns(2)
    for col, (scenario, price, sessions, conv, rev, prof, be, color) in zip(
        res_cols,
        [("A", price_a, sessions_a, conv_a, total_rev_a, total_prof_a, breakeven_a, GOLD),
         ("B", price_b, sessions_b, conv_b, total_rev_b, total_prof_b, breakeven_b, TEAL)]
    ):
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
            border-top:3px solid {color};border-radius:8px;padding:20px;">
                <div style="color:{color};font-size:13px;font-weight:700;margin-bottom:16px;">
                    Scenario {scenario}
                </div>
            """, unsafe_allow_html=True)
            metrics = [
                ("Price", f"₹{price:,}/session"),
                ("Daily Revenue", f"₹{price * sessions * conv/100 * n_pods:,.0f}"),
                ("Daily Profit", f"₹{price * sessions * conv/100 * n_pods - daily_cost:,.0f}"),
                (f"{days}-Day Revenue", f"₹{rev:,.0f}"),
                (f"{days}-Day Profit", f"₹{prof:,.0f}"),
                ("Breakeven Util.", f"{be*100:.1f}%"),
            ]
            for label, value in metrics:
                text_color = TEAL if "Profit" in label and float(value.replace("₹","").replace(",","")) > 0 else INK
                if "Profit" in label and "-" in value:
                    text_color = ROSE
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:8px 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span style="color:{MUTED};font-size:12px;">{label}</span>
                    <span style="color:{text_color};font-size:13px;font-weight:600;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Revenue comparison chart over time
    day_range = list(range(1, days + 1))
    cum_prof_a = [daily_prof_a * d for d in day_range]
    cum_prof_b = [daily_prof_b * d for d in day_range]

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(x=day_range, y=cum_prof_a, mode="lines", name="Scenario A",
                                  line=dict(color=GOLD, width=2.5),
                                  hovertemplate="Day %{x}<br>Cumulative Profit: ₹%{y:,.0f}<extra>A</extra>"))
    fig_sim.add_trace(go.Scatter(x=day_range, y=cum_prof_b, mode="lines", name="Scenario B",
                                  line=dict(color=TEAL, width=2.5),
                                  hovertemplate="Day %{x}<br>Cumulative Profit: ₹%{y:,.0f}<extra>B</extra>"))
    fig_sim.add_hline(y=0, line_dash="dash", line_color=ROSE, line_width=1,
                      annotation_text="Break-even", annotation_font_color=ROSE)
    fig_sim.update_layout(
        title=f"Cumulative Profit Trajectory — {days} Days · {n_pods} Pods",
        xaxis_title="Day", yaxis_title="Cumulative Profit (₹)",
        height=340, legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_sim, width="stretch")

    # ── GEOGRAPHIC PRIORITY HEATMAP ───────────────────────────
    section_header("Geographic Priority — City-Level Interest Heatmap", TEAL)

    city_col = "city" if "city" in df1.columns else None
    if city_col:
        city_interest = df1.groupby(city_col).agg(
            total=("aura_interest_label", "count"),
            interested=(
                "aura_interest_label",
                lambda x: (x == "Interested").sum()
            )
        ).reset_index()
        city_interest["pct_interested"] = (city_interest["interested"] / city_interest["total"] * 100).round(1)
        city_interest["avg_wtp"] = df1.groupby(city_col)["session_wtp_numeric"].mean().values

        city_interest = city_interest.sort_values("pct_interested", ascending=False)

        fig_city = go.Figure()
        fig_city.add_trace(go.Bar(
            x=city_interest[city_col],
            y=city_interest["pct_interested"],
            name="% Interested",
            marker_color=GOLD,
            text=[f"{p:.1f}%" for p in city_interest["pct_interested"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Interested: %{y:.1f}%<extra></extra>"
        ))
        fig_city.add_trace(go.Scatter(
            x=city_interest[city_col],
            y=city_interest["avg_wtp"],
            mode="markers+lines",
            name="Avg WTP (₹)",
            yaxis="y2",
            marker=dict(color=TEAL, size=10, symbol="diamond"),
            line=dict(color=TEAL, width=2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Avg WTP: ₹%{y:,.0f}<extra></extra>"
        ))
        fig_city.update_layout(
            title="City-Level Interest % and Average WTP — Pod Location Priority",
            yaxis=dict(title="% Interested", range=[0, 100]),
            yaxis2=dict(title="Avg WTP (₹)", overlaying="y", side="right"),
            height=380,
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_city, width="stretch")

        # Priority table
        city_interest["priority"] = city_interest.apply(
            lambda r: "🔴 Launch First" if r["pct_interested"] > 50 and r["avg_wtp"] > 600
            else ("🟡 Phase 2" if r["pct_interested"] > 35 else "🟢 Phase 3"), axis=1
        )
        st.dataframe(
            city_interest[[city_col, "total", "interested", "pct_interested", "avg_wtp", "priority"]]
            .rename(columns={city_col: "City", "total": "Surveyed", "interested": "Interested",
                             "pct_interested": "% Interested", "avg_wtp": "Avg WTP (₹)", "priority": "Launch Priority"}),
            width="stretch"
        )

    # ── SEGMENT LAUNCH SEQUENCE ───────────────────────────────
    section_header("Segment Launch Sequence — 3-Phase Strategy", ORANGE)

    phases = [
        (GOLD,   "Phase 1 — Revenue First (Month 1–2)",
         ["💼 Corporate Buyer", "🖌️ Serious Hobbyist"],
         "Highest WTP segments. Corporate buyers deliver B2B revenue with Diwali gift packages. "
         "Serious hobbyists convert at premium price points with monthly unlimited passes. "
         "Goal: achieve break-even within 45 days."),
        (TEAL,   "Phase 2 — Volume & Virality (Month 3–4)",
         ["📸 Status Sharer", "🎨 Weekend Creative"],
         "Largest segments. Status Sharers generate organic Instagram content at near-zero marketing cost. "
         "Weekend Creatives fill off-peak morning slots. "
         "Goal: 80% pod utilisation, 500+ UGC posts, 3-city footprint."),
        (ORANGE, "Phase 3 — Mass Market (Month 5+)",
         ["👨‍👩‍👧 Curious Parent", "🎓 Student Explorer", "🌿 Wellness Seeker"],
         "Price-sensitive segments accessed via partnerships — school holiday camps, college tie-ups, "
         "yoga studio collabs. Lower per-session revenue but high volume and brand equity. "
         "Goal: 1,000+ sessions/month across all pods."),
    ]

    for color, title, segments, desc in phases:
        st.markdown(f"""
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:4px solid {color};border-radius:8px;padding:20px;margin-bottom:16px;">
            <div style="color:{color};font-size:12px;font-weight:800;text-transform:uppercase;
            letter-spacing:0.08em;margin-bottom:10px;">{title}</div>
            <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
                {"".join(f'<span style="background:{color}22;border:1px solid {color}44;color:{color};font-size:11px;padding:4px 10px;border-radius:20px;font-weight:600;">{s}</span>' for s in segments)}
            </div>
            <div style="color:{MUTED};font-size:13px;line-height:1.7;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── CHANNEL BUDGET ALLOCATION ──────────────────────────────
    section_header("Recommended Marketing Channel Mix", INDIGO)

    channels = {
        "Instagram Reels / Stories": 35,
        "Mall & POD Activation":     20,
        "Corporate / LinkedIn":      18,
        "WhatsApp Community":        10,
        "College Partnerships":       8,
        "Google Ads (local)":         9,
    }

    col_pie, col_table = st.columns([1, 1.2])
    with col_pie:
        fig_ch = go.Figure(go.Pie(
            labels=list(channels.keys()),
            values=list(channels.values()),
            hole=0.55,
            marker_colors=PALETTE[:len(channels)],
            textinfo="label+percent",
            textfont=dict(size=10),
            hovertemplate="<b>%{label}</b><br>%{value}%<extra></extra>"
        ))
        fig_ch.update_layout(
            height=320, showlegend=False,
            title="Marketing Budget Mix"
        )
        st.plotly_chart(fig_ch, width="stretch")

    with col_table:
        st.markdown(f"<div style='color:{GOLD};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:12px;'>Channel Rationale</div>", unsafe_allow_html=True)
        rationale = {
            "Instagram Reels / Stories": "Primary discovery channel — Status Sharers create free UGC",
            "Mall & POD Activation":     "In-person trial removes barrier; highest single-session conversion",
            "Corporate / LinkedIn":      "B2B outreach to HR teams — highest revenue per deal",
            "WhatsApp Community":        "Retention and word-of-mouth for parents and hobbyists",
            "College Partnerships":      "Volume play — Student Explorers at low CAC",
            "Google Ads (local)":        "Capture intent searches in Tier-1 cities",
        }
        for ch, pct in channels.items():
            st.markdown(f"""
            <div style="padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="color:{INK};font-size:12px;font-weight:600;">{ch}</span>
                    <span style="color:{GOLD};font-size:12px;font-weight:700;">{pct}%</span>
                </div>
                <div style="color:{MUTED};font-size:11px;">{rationale[ch]}</div>
            </div>
            """, unsafe_allow_html=True)

    info_card(
        "Prescriptive Recommendation",
        "Data tells us: launch in <b>Bengaluru and Mumbai first</b> (highest interest % + WTP). "
        "Price Phase-1 sessions at <b>₹700–₹1,200</b> targeting Corporate Buyers and Serious Hobbyists. "
        "Use the A/B simulator above to stress-test break-even before committing pod locations. "
        "Phase-2 activation of Status Sharers via Instagram will generate organic content that reduces "
        "paid CAC by an estimated 40–60% by Month 4.",
        GOLD
    )
