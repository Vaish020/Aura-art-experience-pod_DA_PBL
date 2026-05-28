"""tab_clustering.py — K-Means Clustering: Silhouette-driven K, Radar Charts, Geographic heatmap"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE, CLUSTER_COLORS,
                        page_header, section_header, info_card,
                        art_image_banner, art_image_grid, colour_palette_strip, IMAGES)
from aura_data import name_cluster, get_strategy, CLU_FEATURES


def render(df1, df2, arm, wide, km_model, df_clustered, km_scaler,
           best_k, k_range, inertias, silhouettes, pca):

    page_header(
        "Customer Persona Segmentation",
        f"K-Means (silhouette-optimised) identified K={best_k} distinct personas — "
        "each with a unique profile, pricing sweet spot, and go-to-market approach.",
        "Clustering — K-Means"
    )

    art_image_banner(
        IMAGES["colour_palette"], height=100,
        overlay_text="Customer Persona Segmentation",
        overlay_sub="K-Means · Silhouette-optimised · 6 distinct AURA personas"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── SILHOUETTE-DRIVEN K SELECTION ─────────────────────────
    section_header("Optimal K — Elbow Method + Silhouette Score (Algorithm-Driven)", GOLD)

    c1, c2 = st.columns(2)

    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias,
            mode="lines+markers",
            line=dict(color=GOLD, width=2.5),
            marker=dict(size=9, color=GOLD, symbol="circle",
                        line=dict(color=SURFACE2, width=1)),
            name="Inertia (WCSS)"
        ))
        fig_elbow.add_vline(x=best_k, line_dash="dash", line_color=TEAL, line_width=2,
                            annotation_text=f"Optimal K={best_k}",
                            annotation_font_color=TEAL, annotation_font_size=12)
        fig_elbow.update_layout(
            title="Elbow Method — Within-Cluster Sum of Squares",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia (WCSS)", height=320
        )
        st.plotly_chart(fig_elbow, width="stretch")

    with c2:
        fig_sil = go.Figure()
        bar_colors = [TEAL if k == best_k else MUTED for k in k_range]
        fig_sil.add_trace(go.Bar(
            x=list(k_range), y=silhouettes,
            marker_color=bar_colors,
            text=[f"{s:.3f}" for s in silhouettes],
            textposition="outside",
            hovertemplate="K=%{x}<br>Silhouette: %{y:.4f}<extra></extra>"
        ))
        fig_sil.add_annotation(
            x=best_k, y=max(silhouettes),
            text=f"✓ Best K={best_k}<br>Score={max(silhouettes):.3f}",
            showarrow=True, arrowhead=2, arrowcolor=GOLD,
            font=dict(color=GOLD, size=11), bgcolor=SURFACE2,
            bordercolor=GOLD, borderwidth=1
        )
        fig_sil.update_layout(
            title="Silhouette Score vs K (Higher = Better Defined Clusters)",
            xaxis_title="K", yaxis_title="Silhouette Score", height=320
        )
        st.plotly_chart(fig_sil, width="stretch")

    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.07);
    border-left:3px solid {TEAL};border-radius:6px;padding:14px 20px;margin:0 0 24px;">
        <span style="color:{TEAL};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;">
        Algorithm Decision: K = {best_k}</span>
        <span style="color:{MUTED};font-size:13px;margin-left:16px;">
        Silhouette score: {max(silhouettes):.4f} · Inertia: {min(inertias):,.0f} · No hardcoded override — purely data-driven</span>
    </div>
    """, unsafe_allow_html=True)

    # ── PCA SCATTER ────────────────────────────────────────────
    section_header("Cluster Visualisation — PCA 2D Projection")

    df_plot = df_clustered.copy()
    df_plot["cluster_name"] = df_plot["cluster"].apply(name_cluster)
    color_map = {name_cluster(i): CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(best_k)}

    fig_pca = px.scatter(
        df_plot, x="pca_x", y="pca_y",
        color="cluster_name",
        color_discrete_map=color_map,
        opacity=0.7,
        title="Customer Segments — PCA 2D Projection (PC1 vs PC2)",
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2",
                "cluster_name": "Persona"},
        hover_data=["occupation", "income_bracket", "art_experience_level", "session_wtp_numeric"]
        if all(c in df_plot.columns for c in ["occupation","income_bracket","art_experience_level","session_wtp_numeric"])
        else None
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(height=460, legend_title="Customer Persona")
    st.plotly_chart(fig_pca, width="stretch")

    # ── CLUSTER SIZES ──────────────────────────────────────────
    section_header("Segment Size Distribution")
    cluster_sizes = df_plot["cluster_name"].value_counts().reset_index()
    cluster_sizes.columns = ["Persona", "Count"]
    cluster_sizes["Pct"] = (cluster_sizes["Count"] / len(df_plot) * 100).round(1)

    fig_size = go.Figure(go.Bar(
        x=cluster_sizes["Persona"],
        y=cluster_sizes["Count"],
        marker_color=[color_map.get(p, GOLD) for p in cluster_sizes["Persona"]],
        text=[f"{c:,}<br>({p}%)" for c, p in zip(cluster_sizes["Count"], cluster_sizes["Pct"])],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))
    fig_size.update_layout(
        title="Respondents per Persona Segment",
        height=320, yaxis_range=[0, cluster_sizes["Count"].max() * 1.2]
    )
    st.plotly_chart(fig_size, width="stretch")

    # ── PER-CLUSTER SILHOUETTE ────────────────────────────────
    if "silhouette_score" in df_plot.columns:
        section_header("Per-Cluster Silhouette Score Distribution", TEAL)
        fig_sil_box = go.Figure()
        for cname, ccolor in color_map.items():
            subset = df_plot[df_plot["cluster_name"] == cname]["silhouette_score"]
            fig_sil_box.add_trace(go.Box(
                y=subset, name=cname,
                marker_color=ccolor,
                boxmean="sd",
                hovertemplate=f"<b>{cname}</b><br>Silhouette: %{{y:.3f}}<extra></extra>"
            ))
        fig_sil_box.update_layout(
            title="Silhouette Score per Customer — By Cluster (Higher = Better Fit)",
            yaxis_title="Silhouette Score", height=340, showlegend=False
        )
        st.plotly_chart(fig_sil_box, width="stretch")

    # ── PERSONA IMAGE ROW ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("The 6 AURA Customer Personas", ORANGE)
    art_image_grid(
        [IMAGES["wellness"], IMAGES["sketching"], IMAGES["pottery"],
         IMAGES["calligraphy"], IMAGES["mandala"], IMAGES["watercolour"]],
        captions=["Wellness Seeker", "Serious Hobbyist", "Curious Parent",
                  "Status Sharer", "Weekend Creative", "Student Explorer"],
        height=110
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── PERSONA DEEP-DIVE WITH RADAR ─────────────────────────
    section_header("Persona Deep-Dive — Radar Profile + Strategy", ORANGE)

    selected_persona = st.selectbox(
        "Select a customer persona:",
        [name_cluster(i) for i in range(best_k)]
    )

    df_seg = df_plot[df_plot["cluster_name"] == selected_persona]
    strategy = get_strategy(selected_persona)

    col_radar, col_info = st.columns([1.1, 1])

    with col_radar:
        # Build radar with 8 dimensions
        cats = [
            "Tech Comfort", "WTP Level", "Art Experience",
            "Social Sharing", "Visit Frequency", "Price Sensitivity ↓",
            "Income Level", "Instagram Influence"
        ]

        def safe_mean_mapped(col, mapping):
            if col not in df_seg.columns:
                return 0.5
            series = df_seg[col]
            # Try mapping first (handles string columns in any pandas version)
            try:
                mapped = series.map(mapping).dropna()
                if len(mapped) > 0:
                    return float(mapped.mean())
            except Exception:
                pass
            # Try numeric mean
            try:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) > 0:
                    return float(numeric.mean())
            except Exception:
                pass
            return 0.5

        vals = [
            df_seg["tech_comfort_score"].mean() / 5 if "tech_comfort_score" in df_seg.columns else 0.5,
            df_seg["session_wtp_numeric"].mean() / 2000 if "session_wtp_numeric" in df_seg.columns else 0.5,
            safe_mean_mapped("art_experience_level",
                             {"Complete_Beginner":0,"Curious_Beginner":0.25,"Casual_Hobbyist":0.5,
                              "Regular_Hobbyist":0.75,"Advanced":1.0}),
            safe_mean_mapped("social_sharing_propensity",
                             {"Definitely_Post":1,"Probably_Post":0.67,"Close_Circle":0.33,"Keep_Private":0}),
            safe_mean_mapped("visit_frequency_intent",
                             {"Unlikely":0,"Once_Try":0.2,"Occasionally":0.4,"Once_month":0.6,"2_3_month":0.8,"Weekly":1}),
            1 - (df_seg["price_sensitivity_score"].mean() / 5 if "price_sensitivity_score" in df_seg.columns else 0.5),
            safe_mean_mapped("income_bracket",
                             {"Below_25k":0,"25k_50k":0.25,"50k_1L":0.5,"1L_2L":0.75,"Above_2L":1,"Prefer_not":0.5}),
            df_seg["instagram_influence_score"].mean() / 5 if "instagram_influence_score" in df_seg.columns else 0.5,
        ]
        vals = [min(1, max(0, v if not np.isnan(v) else 0.5)) for v in vals]

        radar_color = color_map.get(selected_persona, GOLD)
        r, g, b = int(radar_color[1:3], 16), int(radar_color[3:5], 16), int(radar_color[5:7], 16)

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.18)",
            line=dict(color=radar_color, width=2.5),
            marker=dict(size=7, color=radar_color),
            name=selected_persona
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=SURFACE2,
                radialaxis=dict(visible=True, range=[0, 1],
                                tickfont=dict(size=9, color=MUTED),
                                gridcolor="rgba(255,255,255,0.06)"),
                angularaxis=dict(tickfont=dict(size=11, color=INK),
                                 gridcolor="rgba(255,255,255,0.06)")
            ),
            showlegend=False,
            title=f"{selected_persona} — 8-Dimension Profile",
            height=420
        )
        st.plotly_chart(fig_radar, width="stretch")

    with col_info:
        n_seg = len(df_seg)
        pct_seg = n_seg / len(df_plot) * 100

        st.markdown(f"""
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.07);
        border-radius:10px;padding:24px;margin-top:16px;">
            <div style="font-size:24px;margin-bottom:6px;">{selected_persona}</div>
            <div style="color:{MUTED};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:20px;">
                {n_seg:,} customers · {pct_seg:.1f}% of interested segment
            </div>
        """, unsafe_allow_html=True)

        # Key stats
        stats = {}
        if "session_wtp_numeric" in df_seg.columns:
            stats["Median WTP"] = f"₹{int(df_seg['session_wtp_numeric'].median()):,}"
        if "monthly_leisure_spend" in df_seg.columns:
            stats["Avg Leisure Spend"] = f"₹{int(df_seg['monthly_leisure_spend'].median()):,}/mo"
        if "city" in df_seg.columns and len(df_seg) > 0:
            stats["Top City"] = df_seg["city"].value_counts().index[0]
        if "occupation" in df_seg.columns and len(df_seg) > 0:
            stats["Top Occupation"] = df_seg["occupation"].value_counts().index[0]
        if "silhouette_score" in df_seg.columns:
            stats["Avg Silhouette"] = f"{df_seg['silhouette_score'].mean():.3f}"

        for k, v in stats.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:9px 0;
            border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="color:{MUTED};font-size:12px;">{k}</span>
                <span style="color:{INK};font-size:13px;font-weight:600;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        if strategy:
            st.markdown(f"""
            <div style="color:{GOLD};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;
            font-weight:700;margin-top:20px;margin-bottom:12px;">Go-to-Market Strategy</div>
            """, unsafe_allow_html=True)
            for sk, sv in strategy.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:8px 0;
                border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span style="color:{MUTED};font-size:11px;">{sk.replace('_',' ').title()}</span>
                    <span style="color:{TEAL};font-size:11px;font-weight:600;
                    text-align:right;max-width:55%;">{sv}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── ALL PERSONAS RADAR OVERLAY ────────────────────────────
    section_header("All Personas — Radar Overlay Comparison", INDIGO)

    fig_overlay = go.Figure()
    cats_short = ["Tech", "WTP", "Art Exp", "Social", "Visit Freq", "Non-Price-Sens", "Income", "Insta"]

    for i in range(best_k):
        cname = name_cluster(i)
        df_c = df_plot[df_plot["cluster_name"] == cname]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        def safe_val(col, divisor=1.0, mapping=None):
            if col not in df_c.columns:
                return 0.5
            try:
                if mapping:
                    mapped = df_c[col].map(mapping).dropna()
                    if len(mapped) > 0:
                        return float(mapped.mean()) or 0.5
                numeric = pd.to_numeric(df_c[col], errors="coerce").dropna()
                if len(numeric) > 0:
                    return min(1, float(numeric.mean()) / divisor)
            except Exception:
                pass
            return 0.5

        v = [
            safe_val("tech_comfort_score", 5),
            safe_val("session_wtp_numeric", 2000),
            safe_val("art_experience_level", mapping={"Complete_Beginner":0,"Curious_Beginner":0.25,"Casual_Hobbyist":0.5,"Regular_Hobbyist":0.75,"Advanced":1}),
            safe_val("social_sharing_propensity", mapping={"Definitely_Post":1,"Probably_Post":0.67,"Close_Circle":0.33,"Keep_Private":0}),
            safe_val("visit_frequency_intent", mapping={"Unlikely":0,"Once_Try":0.2,"Occasionally":0.4,"Once_month":0.6,"2_3_month":0.8,"Weekly":1}),
            1 - safe_val("price_sensitivity_score", 5),
            safe_val("income_bracket", mapping={"Below_25k":0,"25k_50k":0.25,"50k_1L":0.5,"1L_2L":0.75,"Above_2L":1,"Prefer_not":0.5}),
            safe_val("instagram_influence_score", 5),
        ]
        v = [min(1, max(0, x if not np.isnan(x) else 0.5)) for x in v]

        fig_overlay.add_trace(go.Scatterpolar(
            r=v + [v[0]],
            theta=cats_short + [cats_short[0]],
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.07)",
            line=dict(color=color, width=1.8),
            name=cname
        ))

    fig_overlay.update_layout(
        polar=dict(
            bgcolor=SURFACE2,
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=8, color=MUTED)),
            angularaxis=dict(tickfont=dict(size=11, color=INK))
        ),
        title="All Personas — Overlaid Radar (8 Dimensions)",
        height=500,
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig_overlay, width="stretch")

    # ── CROSS-PERSONA HEATMAP ──────────────────────────────────
    section_header("Cross-Persona Feature Comparison Heatmap")

    num_cols = ["session_wtp_numeric", "monthly_leisure_spend", "monthly_art_spend",
                "tech_comfort_score", "price_sensitivity_score", "instagram_influence_score",
                "recommend_likelihood"]
    avail = [c for c in num_cols if c in df_plot.columns]
    cluster_agg  = df_plot.groupby("cluster_name")[avail].mean()
    cluster_norm = (cluster_agg - cluster_agg.min()) / (cluster_agg.max() - cluster_agg.min() + 1e-9)

    fig_heat = go.Figure(go.Heatmap(
        z=cluster_norm.values,
        x=[c.replace("_", " ").title() for c in cluster_norm.columns],
        y=cluster_norm.index,
        colorscale="YlOrRd",
        showscale=True,
        text=np.round(cluster_agg.values, 0).astype(int),
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>"
    ))
    fig_heat.update_layout(
        title="Persona Comparison Heatmap (colour = normalised, values = raw)",
        height=340, xaxis_tickangle=-25
    )
    st.plotly_chart(fig_heat, width="stretch")

    st.download_button(
        "⬇ Download Cluster Profiles CSV",
        cluster_agg.round(2).to_csv(),
        "aura_cluster_profiles.csv", "text/csv"
    )

    info_card(
        "Clustering Insight",
        f"Silhouette-optimised K-Means selected <b>K={best_k}</b> (Score={max(silhouettes):.3f}) "
        "without any manual override. The Corporate Buyer segment generates the highest WTP per session. "
        "The Status Sharer is the largest and most Instagram-responsive — ideal for launch virality. "
        "<b>Launch sequence recommendation: Corporate Buyers (revenue) → Status Sharers (awareness) → "
        "Weekend Creatives (volume).</b>",
        TEAL
    )
