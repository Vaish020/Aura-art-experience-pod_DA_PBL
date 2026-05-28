"""
tab_hierarchical.py — Sessions 5 & 6: RFM Analysis + Hierarchical Clustering
==============================================================================
"Implementation in Python — K-Means and Hierarchical Clustering"
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, LAVENDER, MUTED,
                        SURFACE, SURFACE2, INK, PALETTE, CLUSTER_COLORS,
                        page_header, section_header, info_card,
                        art_image_banner, colour_palette_strip, IMAGES)
from aura_data import encode_features, CLU_FEATURES, name_cluster


def render(df1, df2, arm, wide):

    page_header(
        "RFM Analysis + Hierarchical Clustering",
        " RFM (Recency, Frequency, Monetary) segmentation followed by "
        "Hierarchical Clustering using Ward's linkage. Compares with K-Means to show "
        "when each method is preferred.",
        "RFM + Hierarchical Clustering"
    )

    art_image_banner(
        IMAGES["pottery"], height=100,
        overlay_text="RFM Analysis + Hierarchical Clustering",
        overlay_sub="Recency · Frequency · Monetary · Dendrogram"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── RFM SECTION ───────────────────────────────────────────
    section_header("Part 1 — RFM Analysis", GOLD)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;">
        <div style="background:{SURFACE2};border-left:3px solid {GOLD};border-radius:8px;padding:16px;">
            <div style="color:{GOLD};font-size:18px;font-weight:800;margin-bottom:6px;">R — Recency</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                How recently did the customer interact with art/wellness activities?
                Proxied by <b style="color:{INK};">visit_frequency_intent</b> score.
                Higher = more recent engagement mindset.
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {TEAL};border-radius:8px;padding:16px;">
            <div style="color:{TEAL};font-size:18px;font-weight:800;margin-bottom:6px;">F — Frequency</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                How often do they engage with experience-based activities?
                Proxied by <b style="color:{INK};">online_exp_purchase_freq</b> + subscription_count.
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {ORANGE};border-radius:8px;padding:16px;">
            <div style="color:{ORANGE};font-size:18px;font-weight:800;margin-bottom:6px;">M — Monetary</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                How much are they willing/able to spend?
                Proxied by <b style="color:{INK};">session_wtp_numeric</b> + monthly_leisure_spend.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build RFM scores
    df_rfm = df1.copy()

    # Recency proxy: visit_frequency_intent mapped to 1-5
    vfi_map = {"Unlikely": 1, "Once_Try": 2, "Occasionally": 3,
               "Once_month": 4, "2_3_month": 5, "Weekly": 5}
    df_rfm["R"] = df_rfm["visit_frequency_intent"].map(vfi_map).fillna(3)

    # Frequency proxy: online_exp_purchase_freq + subscription_count
    oepf_map = {"Rarely": 1, "Occasionally": 2, "Once_month": 3, "2_3_month": 4, "Weekly": 5}
    sub_map = {"0": 0, "1_2": 1, "3_5": 2, "6_plus": 3}
    df_rfm["F_raw"] = df_rfm["online_exp_purchase_freq"].map(oepf_map).fillna(2)
    df_rfm["F_sub"] = df_rfm["subscription_count"].map(sub_map).fillna(1)
    df_rfm["F"] = ((df_rfm["F_raw"] + df_rfm["F_sub"]) / 2).clip(1, 5)

    # Monetary proxy: WTP + leisure
    wtp_col = "session_wtp_numeric"
    if wtp_col in df_rfm.columns:
        wtp_norm = df_rfm[wtp_col].fillna(df_rfm[wtp_col].median())
        df_rfm["M"] = pd.qcut(wtp_norm, 5, labels=[1, 2, 3, 4, 5]).astype(float)
    else:
        df_rfm["M"] = 3.0

    # RFM Score
    df_rfm["RFM_Score"] = df_rfm["R"] + df_rfm["F"] + df_rfm["M"]

    # RFM Segment labels
    def rfm_segment(row):
        score = row["RFM_Score"]
        if score >= 12:
            return "Champions"
        elif score >= 10:
            return "Loyal Customers"
        elif score >= 8:
            return "Potential Loyalists"
        elif score >= 6:
            return "At Risk"
        else:
            return "Lost / Uninterested"

    df_rfm["RFM_Segment"] = df_rfm.apply(rfm_segment, axis=1)

    seg_colors = {
        "Champions": TEAL,
        "Loyal Customers": GOLD,
        "Potential Loyalists": ORANGE,
        "At Risk": ROSE,
        "Lost / Uninterested": MUTED,
    }

    # RFM Distribution
    c1, c2 = st.columns(2)
    with c1:
        seg_counts = df_rfm["RFM_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_rfm_pie = go.Figure(go.Pie(
            labels=seg_counts["Segment"],
            values=seg_counts["Count"],
            hole=0.55,
            marker_colors=[seg_colors.get(s, MUTED) for s in seg_counts["Segment"]],
            textinfo="label+percent",
            textfont=dict(size=10)
        ))
        fig_rfm_pie.update_layout(
            title="RFM Segment Distribution", height=320, showlegend=False
        )
        st.plotly_chart(fig_rfm_pie, width="stretch")

    with c2:
        # RFM scatter: R vs M, coloured by F
        sample = df_rfm.sample(min(500, len(df_rfm)), random_state=42)
        fig_rfm_sc = go.Figure()
        for seg, color in seg_colors.items():
            sub = sample[sample["RFM_Segment"] == seg]
            if len(sub) == 0:
                continue
            fig_rfm_sc.add_trace(go.Scatter(
                x=sub["R"], y=sub["M"],
                mode="markers",
                name=seg,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate=f"<b>{seg}</b><br>R=%{{x}}<br>M=%{{y}}<extra></extra>"
            ))
        fig_rfm_sc.update_layout(
            title="RFM Plot — Recency vs Monetary (coloured by Segment)",
            xaxis_title="Recency Score", yaxis_title="Monetary Score",
            height=320, legend=dict(font=dict(size=9))
        )
        st.plotly_chart(fig_rfm_sc, width="stretch")

    # RFM segment stats
    section_header("RFM Segment Profiles + AURA Strategy", TEAL)
    rfm_stats = df_rfm.groupby("RFM_Segment").agg(
        Count=("RFM_Score", "count"),
        Avg_R=("R", "mean"),
        Avg_F=("F", "mean"),
        Avg_M=("M", "mean"),
        Avg_RFM=("RFM_Score", "mean")
    ).round(2).reset_index()

    strategies = {
        "Champions":            ("₹700–₹1,200 premium", "Monthly unlimited pass + early access", "LinkedIn + Email"),
        "Loyal Customers":      ("₹500–₹700 standard",  "Loyalty points + referral reward",      "WhatsApp + Instagram"),
        "Potential Loyalists":  ("₹299 intro offer",    "2nd session discount",                   "Instagram Reels"),
        "At Risk":              ("₹199 win-back offer", "Re-engagement campaign",                  "Email + SMS"),
        "Lost / Uninterested":  ("Awareness only",      "No paid outreach",                        "Organic social only"),
    }

    for _, row in rfm_stats.iterrows():
        seg = row["RFM_Segment"]
        color = seg_colors.get(seg, MUTED)
        strat = strategies.get(seg, ("—", "—", "—"))
        st.markdown(f"""
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
        border-left:4px solid {color};border-radius:8px;padding:16px;margin-bottom:10px;
        display:grid;grid-template-columns:1.5fr 1fr 1fr 1fr 1fr 1fr;gap:12px;align-items:center;">
            <div>
                <div style="color:{color};font-size:13px;font-weight:700;">{seg}</div>
                <div style="color:{MUTED};font-size:11px;">{int(row['Count']):,} customers</div>
            </div>
            <div style="text-align:center;">
                <div style="color:{MUTED};font-size:10px;text-transform:uppercase;">Price</div>
                <div style="color:{INK};font-size:11px;font-weight:600;">{strat[0]}</div>
            </div>
            <div style="text-align:center;">
                <div style="color:{MUTED};font-size:10px;text-transform:uppercase;">Avg R</div>
                <div style="color:{GOLD};font-size:13px;font-weight:700;">{row['Avg_R']:.1f}</div>
            </div>
            <div style="text-align:center;">
                <div style="color:{MUTED};font-size:10px;text-transform:uppercase;">Avg F</div>
                <div style="color:{TEAL};font-size:13px;font-weight:700;">{row['Avg_F']:.1f}</div>
            </div>
            <div style="text-align:center;">
                <div style="color:{MUTED};font-size:10px;text-transform:uppercase;">Avg M</div>
                <div style="color:{ORANGE};font-size:13px;font-weight:700;">{row['Avg_M']:.1f}</div>
            </div>
            <div style="text-align:center;">
                <div style="color:{MUTED};font-size:10px;text-transform:uppercase;">Channel</div>
                <div style="color:{INK};font-size:11px;">{strat[2]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.download_button(
        "⬇ Download RFM Segments CSV",
        df_rfm[["RFM_Segment", "R", "F", "M", "RFM_Score"]].to_csv(index=False),
        "aura_rfm_segments.csv", "text/csv"
    )

    # ── HIERARCHICAL CLUSTERING ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Part 2 — Hierarchical Clustering", INDIGO)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;">
        <div style="background:{SURFACE2};border-left:3px solid {INDIGO};border-radius:8px;padding:16px;">
            <div style="color:{INDIGO};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">How It Works</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                Hierarchical clustering builds a <b style="color:{INK};">dendrogram</b> — a tree of merges.
                It starts with every customer as its own cluster and merges the closest two at each step.
                No need to pre-specify K — you cut the dendrogram at any level.
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {LAVENDER};border-radius:8px;padding:16px;">
            <div style="color:{LAVENDER};font-size:11px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">K-Means vs Hierarchical</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                <b style="color:{INK};">K-Means:</b> Fast, scalable, requires K upfront, spherical clusters.<br>
                <b style="color:{INK};">Hierarchical:</b> No K needed, produces dendrogram, better for small datasets, finds arbitrary shapes.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Parameters
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        linkage_method = st.selectbox("Linkage Method",
                                       ["ward", "complete", "average", "single"],
                                       help="Ward minimises total within-cluster variance — usually best")
    with col_h2:
        n_hier_clusters = st.slider("Number of Clusters (cut level)", 2, 8, 4)
    with col_h3:
        sample_size = st.slider("Sample Size for Dendrogram", 50, 300, 150,
                                 help="Hierarchical clustering is O(n²) — sample for speed")

    # Prepare data
    df_int = df1[df1["aura_interest_label"] == "Interested"].copy()
    if len(df_int) < 50:
        df_int = df1.copy()

    X_hier = encode_features(df_int, CLU_FEATURES).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hier)

    # Sample for dendrogram
    sample_idx = np.random.RandomState(42).choice(len(X_scaled), min(sample_size, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]

    # Linkage
    Z = linkage(X_sample, method=linkage_method, metric="euclidean")

    # ── DENDROGRAM ────────────────────────────────────────────
    section_header("Dendrogram — Hierarchical Merge Tree", INDIGO)

    # Build dendrogram using scipy + convert to plotly
    from scipy.cluster.hierarchy import dendrogram as sp_dendrogram
    ddata = sp_dendrogram(Z, no_plot=True, truncate_mode="lastp", p=30)

    fig_dendro = go.Figure()
    icoord = np.array(ddata["icoord"])
    dcoord = np.array(ddata["dcoord"])

    for xs, ys in zip(icoord, dcoord):
        fig_dendro.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=TEAL, width=1.5),
            hoverinfo="none",
            showlegend=False
        ))

    # Mark cut level
    cut_height = sorted(Z[:, 2])[-n_hier_clusters + 1] if n_hier_clusters > 1 else Z[-1, 2] / 2
    fig_dendro.add_hline(y=cut_height, line_dash="dash", line_color=GOLD, line_width=2,
                          annotation_text=f"Cut → {n_hier_clusters} clusters",
                          annotation_font_color=GOLD)

    fig_dendro.update_layout(
        title=f"Dendrogram — {linkage_method.title()} Linkage (n={sample_size} sample)",
        xaxis=dict(visible=False),
        yaxis_title="Distance (merge cost)",
        height=400
    )
    st.plotly_chart(fig_dendro, width="stretch")

    # ── CLUSTER ASSIGNMENT ────────────────────────────────────
    cluster_labels_full = fcluster(
        linkage(X_scaled[:500] if len(X_scaled) > 500 else X_scaled,
                method=linkage_method),
        t=n_hier_clusters, criterion="maxclust"
    )

    df_hier = df_int.iloc[:len(cluster_labels_full)].copy()
    df_hier["hier_cluster"] = cluster_labels_full

    # PCA for visualisation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled[:len(cluster_labels_full)])
    df_hier["pca_x"] = X_pca[:, 0]
    df_hier["pca_y"] = X_pca[:, 1]

    section_header("Hierarchical Cluster Visualisation — PCA 2D", TEAL)

    fig_hpca = go.Figure()
    for c in sorted(df_hier["hier_cluster"].unique()):
        sub = df_hier[df_hier["hier_cluster"] == c]
        color = CLUSTER_COLORS[(c - 1) % len(CLUSTER_COLORS)]
        fig_hpca.add_trace(go.Scatter(
            x=sub["pca_x"], y=sub["pca_y"],
            mode="markers",
            name=f"Cluster {c}",
            marker=dict(color=color, size=5, opacity=0.7),
            hovertemplate=f"Cluster {c}<br>PCA1=%{{x:.2f}}<br>PCA2=%{{y:.2f}}<extra></extra>"
        ))
    fig_hpca.update_layout(
        title=f"Hierarchical Clustering — {n_hier_clusters} Clusters ({linkage_method.title()} linkage)",
        xaxis_title="PC1", yaxis_title="PC2",
        height=420, legend_title="Cluster"
    )
    st.plotly_chart(fig_hpca, width="stretch")

    # ── K-MEANS vs HIERARCHICAL COMPARISON ────────────────────
    section_header("K-Means vs Hierarchical — Side-by-Side Comparison", ORANGE)

    # Compute silhouette for both
    from sklearn.cluster import KMeans
    km_comp = KMeans(n_clusters=n_hier_clusters, random_state=42, n_init=10)
    km_labels = km_comp.fit_predict(X_scaled[:len(cluster_labels_full)])

    sil_km = silhouette_score(X_scaled[:len(cluster_labels_full)], km_labels)
    sil_hi = silhouette_score(X_scaled[:len(cluster_labels_full)], cluster_labels_full)

    comp_cols = st.columns(2)
    for col, (method, sil, color) in zip(comp_cols, [
        ("K-Means", sil_km, GOLD),
        (f"Hierarchical ({linkage_method})", sil_hi, TEAL)
    ]):
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
            border-top:3px solid {color};border-radius:8px;padding:20px;text-align:center;">
                <div style="color:{MUTED};font-size:11px;text-transform:uppercase;
                font-weight:700;margin-bottom:10px;">{method}</div>
                <div style="color:{color};font-size:36px;font-weight:800;">{sil:.3f}</div>
                <div style="color:{MUTED};font-size:11px;margin-top:6px;">Silhouette Score</div>
                <div style="color:{INK};font-size:12px;margin-top:10px;">
                    {'Better defined clusters' if sil > 0.3 else 'Moderate cluster separation'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
    border-left:3px solid {ORANGE};border-radius:8px;padding:16px;margin-top:16px;">
        <div style="color:{ORANGE};font-size:11px;font-weight:700;text-transform:uppercase;
        margin-bottom:10px;">When to Use Each Method</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div style="color:{MUTED};font-size:12px;line-height:1.8;">
                <b style="color:{GOLD};">Use K-Means when:</b><br>
                • Dataset is large (1,000+ rows)<br>
                • You have a K estimate from domain knowledge<br>
                • Speed matters<br>
                • Clusters are roughly spherical
            </div>
            <div style="color:{MUTED};font-size:12px;line-height:1.8;">
                <b style="color:{TEAL};">Use Hierarchical when:</b><br>
                • Dataset is smaller (≤500 rows)<br>
                • You don't know K in advance<br>
                • You want a full merge tree (dendrogram)<br>
                • Clusters may have irregular shapes
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_card(
        "Hierarchical Clustering Insight",
        f"Using <b>{linkage_method} linkage</b>, hierarchical clustering produces "
        f"a silhouette score of <b>{sil_hi:.3f}</b> vs K-Means <b>{sil_km:.3f}</b> "
        f"at K={n_hier_clusters}. "
        "The dendrogram shows the merge distances — a large jump in merge cost "
        "is where the natural cluster boundary lies. "
        "For AURA's dataset of 2,000 respondents, K-Means is preferred for deployment, "
        "but hierarchical clustering confirms the cluster structure and validates K.",
        TEAL
    )
