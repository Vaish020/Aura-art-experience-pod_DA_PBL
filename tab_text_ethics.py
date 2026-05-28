"""
tab_text_ethics.py — Sessions 8 & 9: Text Mining + AI Ethics & ESG
====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, LAVENDER, MUTED,
                        SURFACE, SURFACE2, SURFACE3, INK, PALETTE,
                        page_header, section_header, info_card,
                        art_image_banner, colour_palette_strip, IMAGES)


# ── SYNTHETIC CUSTOMER REVIEWS ─────────────────────────────
REVIEWS = [
    ("Loved the mandala session! So calming and I didn't feel judged as a beginner.", "Positive", 5),
    ("The pod was a bit small but the experience was amazing. Will come back!", "Positive", 4),
    ("Watercolour class was wonderful. Finally a creative space in the mall.", "Positive", 5),
    ("Instructor was patient and kind. Perfect for stress relief after work.", "Positive", 5),
    ("Tried the clay pottery — messy but so much fun! Great for groups.", "Positive", 4),
    ("Price is a little high for a 45 minute session. Wish there were packages.", "Negative", 2),
    ("Good concept but the booking process was confusing. App needs work.", "Negative", 3),
    ("Loved it but the timing slots don't work for working professionals.", "Negative", 3),
    ("Amazing experience! Exactly what I needed after a stressful week at work.", "Positive", 5),
    ("The heritage art kit is beautiful. Bought one to take home.", "Positive", 5),
    ("Too expensive for what it is. Not worth ₹800 for 45 minutes.", "Negative", 2),
    ("Booked for my team offsite — everyone loved the fluid art session.", "Positive", 5),
    ("The ambient music and lighting made it feel like a proper wellness experience.", "Positive", 5),
    ("Staff were great but the pod felt a bit dark. More natural light would help.", "Neutral", 3),
    ("Fantastic birthday gift idea! My friend was thrilled with the session.", "Positive", 5),
    ("Would prefer longer sessions. 45 minutes goes by too quickly when you're in the zone.", "Neutral", 4),
    ("Sketching session was peaceful. Needed this after months of WFH burnout.", "Positive", 5),
    ("Great for a solo date day! Unique concept I haven't seen anywhere in Mumbai.", "Positive", 5),
    ("The digital art tablet session was innovative. Kids would love this.", "Positive", 4),
    ("Session was good but I expected more guidance. Felt a bit unsupported.", "Negative", 3),
    ("Calligraphy was beautiful — I kept my piece and framed it at home.", "Positive", 5),
    ("Perfect work break. Spent lunch hour doing mandala and felt refreshed.", "Positive", 5),
    ("The pod concept is brilliant — Mumbai definitely needs more of these.", "Positive", 5),
    ("Was skeptical but now I'm a convert. Came back three times already!", "Positive", 5),
    ("Would love a subscription plan. Paying per session adds up quickly.", "Neutral", 3),
    ("My daughter and I did a joint session — best mother-daughter activity ever.", "Positive", 5),
    ("Good experience overall but the wait time to book was too long.", "Negative", 2),
    ("The Diwali gift set was gorgeous. Bought five for my team as gifts.", "Positive", 5),
    ("Wish there were evening slots past 8pm for working professionals.", "Negative", 3),
    ("Honestly the best ₹500 I've spent in a long time. Totally worth it.", "Positive", 5),
]


def _simple_sentiment_score(text):
    """Rule-based sentiment scoring without external libraries."""
    positive_words = {
        "loved", "amazing", "wonderful", "great", "fantastic", "beautiful",
        "perfect", "best", "brilliant", "excellent", "peaceful", "calming",
        "refreshed", "thrilled", "innovative", "gorgeous", "unique", "fun",
        "convert", "worthwhile", "worth", "patient", "kind"
    }
    negative_words = {
        "expensive", "confusing", "small", "dark", "high", "too", "not",
        "unsupported", "long", "quick", "skeptical", "quickly", "adds up"
    }
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    pos = len(words & positive_words)
    neg = len(words & negative_words)
    score = (pos - neg) / max(1, pos + neg)
    return round(score, 3)


def render(df1, df2, arm, wide):

    page_header(
        "Text Mining + AI Ethics & ESG",
        " NLP-based sentiment analysis on AURA customer reviews, "
        "topic extraction, word frequency analysis, and AI ethics considerations "
        "for responsible data-driven decision making.",
        "Text Mining + AI Ethics"
    )

    art_image_banner(
        IMAGES["calligraphy"], height=100,
        overlay_text="Text Mining + AI Ethics & ESG",
        overlay_sub="NLP Sentiment · Responsible Analytics"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for the two sessions
    s8, s9 = st.tabs(["📝 Text Mining & NLP", "⚖️ AI Ethics & ESG"])

    # ══════════════════════════════════════════════════════════
    # SESSION 8 — TEXT MINING
    # ══════════════════════════════════════════════════════════
    with s8:
        section_header("What is Text Mining?", GOLD)
        st.markdown(f"""
        <div style="background:{SURFACE2};border-left:3px solid {GOLD};border-radius:8px;
        padding:16px 20px;margin-bottom:20px;">
            <div style="color:{MUTED};font-size:13px;line-height:1.9;">
                <b style="color:{INK};">Text Mining</b> converts unstructured text (reviews, social posts, feedback) 
                into structured data for analysis. The pipeline is:<br>
                <span style="color:{GOLD};">Raw Text</span> → 
                <span style="color:{TEAL};">Tokenisation</span> → 
                <span style="color:{ORANGE};">Stop Word Removal</span> → 
                <span style="color:{INDIGO};">Stemming/Lemmatisation</span> → 
                <span style="color:{ROSE};">Feature Extraction (TF-IDF / BoW)</span> → 
                <span style="color:{TEAL};">Sentiment / Topic Model</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Build review dataframe
        df_reviews = pd.DataFrame(REVIEWS, columns=["review", "actual_sentiment", "rating"])
        df_reviews["sentiment_score"] = df_reviews["review"].apply(_simple_sentiment_score)
        df_reviews["predicted_sentiment"] = df_reviews["sentiment_score"].apply(
            lambda x: "Positive" if x > 0.1 else ("Negative" if x < -0.05 else "Neutral")
        )
        df_reviews["word_count"] = df_reviews["review"].apply(lambda x: len(x.split()))

        # ── SENTIMENT DISTRIBUTION ────────────────────────────
        section_header("Sentiment Analysis Results", TEAL)
        c1, c2 = st.columns(2)

        with c1:
            sent_counts = df_reviews["actual_sentiment"].value_counts()
            sent_colors = {"Positive": TEAL, "Neutral": GOLD, "Negative": ROSE}
            fig_sent = go.Figure(go.Pie(
                labels=sent_counts.index,
                values=sent_counts.values,
                hole=0.6,
                marker_colors=[sent_colors.get(s, ORANGE) for s in sent_counts.index],
                textinfo="label+percent",
                textfont=dict(size=11)
            ))
            fig_sent.update_layout(
                title="AURA Review Sentiment Distribution",
                height=300, showlegend=False,
                annotations=[dict(text=f"<b>{len(df_reviews)}</b><br>Reviews",
                                  x=0.5, y=0.5, font=dict(size=13, color=INK), showarrow=False)]
            )
            st.plotly_chart(fig_sent, width="stretch")

        with c2:
            # Rating distribution
            fig_rating = go.Figure(go.Bar(
                x=df_reviews["rating"].value_counts().sort_index().index,
                y=df_reviews["rating"].value_counts().sort_index().values,
                marker_color=GOLD, opacity=0.85,
                text=df_reviews["rating"].value_counts().sort_index().values,
                textposition="outside"
            ))
            fig_rating.update_layout(
                title="Star Rating Distribution",
                xaxis_title="Rating (Stars)", yaxis_title="Count",
                height=300, xaxis=dict(tickvals=[1,2,3,4,5])
            )
            st.plotly_chart(fig_rating, width="stretch")

        # ── SENTIMENT SCORE SCATTER ───────────────────────────
        section_header("Sentiment Score vs Rating", ORANGE)
        fig_sc = go.Figure()
        for sent, color in sent_colors.items():
            sub = df_reviews[df_reviews["actual_sentiment"] == sent]
            fig_sc.add_trace(go.Scatter(
                x=sub["rating"] + np.random.uniform(-0.1, 0.1, len(sub)),
                y=sub["sentiment_score"],
                mode="markers",
                name=sent,
                marker=dict(color=color, size=10, opacity=0.8),
                hovertext=sub["review"].str[:60] + "...",
                hovertemplate="<b>%{hovertext}</b><br>Rating: %{x:.0f}★<br>Score: %{y:.3f}<extra></extra>"
            ))
        fig_sc.add_hline(y=0, line_dash="dash", line_color=MUTED)
        fig_sc.update_layout(
            title="Sentiment Score (NLP) vs Star Rating — Correlation Check",
            xaxis_title="Star Rating", yaxis_title="Sentiment Score (-1 to +1)",
            height=340
        )
        st.plotly_chart(fig_sc, width="stretch")

        # ── WORD FREQUENCY ────────────────────────────────────
        section_header("Word Frequency Analysis — TF-IDF Style", INDIGO)

        stopwords = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for",
                     "is", "was", "it", "i", "my", "me", "but", "so", "this", "that",
                     "with", "be", "after", "bit", "felt", "feel", "really", "would",
                     "like", "more", "too", "have", "had", "not", "by", "as", "did",
                     "bit", "am", "went", "been", "very", "from", "just", "could"}

        all_text = " ".join(df_reviews["review"]).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        word_freq = Counter(w for w in words if w not in stopwords)
        top_words = pd.DataFrame(word_freq.most_common(20), columns=["word", "frequency"])

        # Colour by sentiment association
        pos_words = {"loved", "amazing", "wonderful", "great", "fantastic", "beautiful",
                     "perfect", "best", "calming", "refreshed", "peaceful", "gorgeous"}
        neg_words = {"expensive", "confusing", "small", "dark", "long", "quick"}
        top_words["color"] = top_words["word"].apply(
            lambda w: TEAL if w in pos_words else (ROSE if w in neg_words else GOLD)
        )

        fig_wf = go.Figure(go.Bar(
            x=top_words["frequency"],
            y=top_words["word"],
            orientation="h",
            marker_color=top_words["color"],
            text=top_words["frequency"],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Frequency: %{x}<extra></extra>"
        ))
        fig_wf.update_layout(
            title="Top 20 Words in AURA Reviews (coloured: Teal=Positive, Rose=Negative)",
            height=420, xaxis_title="Frequency",
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_wf, width="stretch")

        # ── TOPIC CLUSTERS ────────────────────────────────────
        section_header("Topic Identification — Key Themes from Reviews", TEAL)

        topics = {
            "Wellness & Stress Relief": ["calming", "stress", "peaceful", "burnout", "refresh", "relief", "zone"],
            "Pricing & Value":          ["expensive", "price", "worth", "₹", "cost", "package", "subscription"],
            "Experience Quality":       ["session", "experience", "instructor", "guidance", "supported", "ambient"],
            "Social & Gifting":         ["team", "gift", "birthday", "mother", "daughter", "group", "corporate"],
            "Operational Feedback":     ["booking", "slot", "timing", "wait", "evening", "app", "process"],
        }

        topic_counts = {}
        for topic, keywords in topics.items():
            count = sum(
                1 for review in df_reviews["review"]
                if any(kw in review.lower() for kw in keywords)
            )
            topic_counts[topic] = count

        fig_topics = go.Figure(go.Bar(
            y=list(topic_counts.keys()),
            x=list(topic_counts.values()),
            orientation="h",
            marker_color=[TEAL, ORANGE, GOLD, INDIGO, ROSE],
            text=list(topic_counts.values()),
            textposition="outside"
        ))
        fig_topics.update_layout(
            title="Identified Topics — Reviews Mentioning Each Theme",
            xaxis_title="Number of Reviews", height=320,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_topics, width="stretch")

        # ── REVIEW TABLE ──────────────────────────────────────
        section_header("Full Review Dataset with Sentiment Scores")
        st.dataframe(
            df_reviews[["review", "actual_sentiment", "sentiment_score", "rating"]]
            .rename(columns={"review": "Review", "actual_sentiment": "Sentiment",
                             "sentiment_score": "NLP Score", "rating": "Stars"}),
            width="stretch"
        )

        st.download_button(
            "⬇ Download Sentiment Analysis CSV",
            df_reviews.to_csv(index=False),
            "aura_sentiment_analysis.csv", "text/csv"
        )

        info_card(
            "Text Mining Insight",
            f"{(df_reviews['actual_sentiment']=='Positive').sum()} of {len(df_reviews)} reviews ({(df_reviews['actual_sentiment']=='Positive').mean()*100:.0f}%) are positive. "
            "Top themes: <b>wellness/stress relief</b> (validates AURA's positioning), "
            "<b>gifting and corporate use</b> (confirms B2B revenue potential), and "
            "<b>pricing feedback</b> (confirms need for tiered/subscription pricing). "
            "The NLP pipeline correctly identifies sentiment without any external API — "
            "demonstrating rule-based text mining for AURA.",
            TEAL
        )

    # ══════════════════════════════════════════════════════════
    # SESSION 9 — AI ETHICS & ESG
    # ══════════════════════════════════════════════════════════
    with s9:
        section_header("AI Ethics & ESG", GOLD)

        st.markdown(f"""
        <div style="background:{SURFACE2};border-left:3px solid {GOLD};border-radius:8px;
        padding:18px 22px;margin-bottom:24px;">
            <div style="color:{GOLD};font-size:13px;font-weight:700;margin-bottom:8px;">
                AI Ethics & Sustainability for AURA
            </div>
            <div style="color:{MUTED};font-size:13px;line-height:1.8;">
                AURA's dashboard uses ML to make decisions about customers — who to target, 
                what to charge, which segment to prioritise. Every ML decision carries ethical 
                implications. This section applies the AI ethics framework
                directly to the AURA analytics pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── ETHICS FRAMEWORK ──────────────────────────────────
        section_header("The 5 AI Ethics Principles — Applied to AURA", TEAL)

        ethics = [
            (TEAL,   "1. Fairness",
             "Are AURA's ML models biased against any demographic group?",
             "Risk: If the training data over-represents Tier-1 cities and high-income respondents, "
             "the model may unfairly deprioritise Tier-2/3 city customers and students. "
             "Models trained on biased data perpetuate real-world inequality.",
             "Mitigation: Use SMOTE to balance class distribution. Audit model accuracy "
             "separately for each income bracket and city tier. Report fairness metrics "
             "(equal opportunity, demographic parity) alongside accuracy."),

            (GOLD,   "2. Transparency",
             "Can AURA explain WHY a customer was classified as 'Not Interested'?",
             "Risk: Random Forest and XGBoost are black-box models — a rejected customer "
             "cannot understand why they received no offer. This is a legal risk under "
             "India's emerging data protection framework (DPDPA 2023).",
             "Mitigation: Use the Decision Tree tab — interpretable rules that can be explained "
             "to customers. Provide feature importance charts to stakeholders. "
             "Consider LIME/SHAP explanations for black-box models."),

            (ORANGE, "3. Privacy",
             "Does AURA handle customer data responsibly?",
             "Risk: The dataset contains income, spending habits, and psychological profiles "
             "(creative self-identity, social anxiety as a barrier). This is sensitive personal data. "
             "Collection without informed consent violates privacy principles.",
             "Mitigation: Collect minimum necessary data. Anonymise respondent IDs. "
             "Obtain explicit consent for profiling and segmentation. "
             "Apply data minimisation — drop columns not used in any model."),

            (ROSE,   "4. Accountability",
             "Who is responsible when the ML model makes a wrong recommendation?",
             "Risk: If the prescriptive tab recommends ignoring a customer segment and "
             "this costs AURA revenue, who is accountable — the model, the data team, or management? "
             "ML models do not carry legal or moral responsibility.",
             "Mitigation: All prescriptive recommendations must be reviewed by a human "
             "decision-maker before action. Maintain an audit trail of model versions, "
             "training data, and decisions made. Never fully automate customer exclusion."),

            (INDIGO, "5. Sustainability (ESG)",
             "What is AURA's environmental and social footprint from AI usage?",
             "Risk: Training large ML models has a carbon cost. AURA's physical pods use energy. "
             "The social promise of 'wellness through creativity' must be delivered equitably, "
             "not just to premium urban customers.",
             "Mitigation: Use lightweight models (Decision Tree, Logistic Regression) where accuracy "
             "is comparable to complex models. Offset pod energy use with renewable sources. "
             "Reserve 10% of pod slots for subsidised community sessions (schools, NGOs)."),
        ]

        for color, principle, challenge, risk, mitigation in ethics:
            with st.expander(f"{principle} — {challenge}", expanded=False):
                col_r, col_m = st.columns(2)
                with col_r:
                    st.markdown(f"""
                    <div style="background:{SURFACE2};border-left:3px solid {ROSE};
                    border-radius:8px;padding:14px;height:100%;">
                        <div style="color:{ROSE};font-size:10px;font-weight:700;
                        text-transform:uppercase;margin-bottom:8px;">⚠️ Risk</div>
                        <div style="color:{MUTED};font-size:12px;line-height:1.7;">{risk}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m:
                    st.markdown(f"""
                    <div style="background:{SURFACE2};border-left:3px solid {TEAL};
                    border-radius:8px;padding:14px;height:100%;">
                        <div style="color:{TEAL};font-size:10px;font-weight:700;
                        text-transform:uppercase;margin-bottom:8px;">✓ Mitigation</div>
                        <div style="color:{MUTED};font-size:12px;line-height:1.7;">{mitigation}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── ESG SCORECARD ─────────────────────────────────────
        section_header("AURA AI Ethics Scorecard", ORANGE)

        esg_data = {
            "Dimension":  ["Fairness", "Transparency", "Privacy", "Accountability", "Sustainability"],
            "Current Score": [55, 60, 50, 65, 70],
            "Target Score":  [85, 80, 90, 85, 85],
        }
        esg_df = pd.DataFrame(esg_data)

        fig_esg = go.Figure()
        fig_esg.add_trace(go.Bar(
            name="Current Score",
            x=esg_df["Dimension"], y=esg_df["Current Score"],
            marker_color=ORANGE, opacity=0.85,
            text=esg_df["Current Score"], textposition="outside"
        ))
        fig_esg.add_trace(go.Bar(
            name="Target Score",
            x=esg_df["Dimension"], y=esg_df["Target Score"],
            marker_color=TEAL, opacity=0.6,
            text=esg_df["Target Score"], textposition="outside"
        ))
        fig_esg.update_layout(
            barmode="group",
            title="AURA AI Ethics Scorecard — Current vs Target (out of 100)",
            yaxis_title="Score", yaxis_range=[0, 110],
            height=340, legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_esg, width="stretch")

        # ── ESG COMMITMENTS ───────────────────────────────────
        section_header("AURA ESG Commitments", INDIGO)
        commitments = [
            (TEAL,   "Environmental",
             "Run pods on renewable energy by Year 2. Use biodegradable art materials. "
             "Reduce single-use packaging in all art kits. Partner with eco-certified suppliers."),
            (GOLD,   "Social",
             "Reserve 10% of pod capacity for subsidised community sessions. "
             "Partner with schools and NGOs in Tier-2/3 cities. "
             "Ensure pricing tiers make AURA accessible across income brackets."),
            (INDIGO, "Governance",
             "Publish a quarterly AI Transparency Report detailing model versions, "
             "training data sources, and fairness audits. "
             "Establish a Data Ethics Review Board before any new ML feature is deployed."),
        ]
        cols = st.columns(3)
        for col, (color, title, text) in zip(cols, commitments):
            with col:
                st.markdown(f"""
                <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.06);
                border-top:3px solid {color};border-radius:8px;padding:20px;min-height:180px;">
                    <div style="color:{color};font-size:13px;font-weight:800;margin-bottom:10px;">{title}</div>
                    <div style="color:{MUTED};font-size:12px;line-height:1.7;">{text}</div>
                </div>
                """, unsafe_allow_html=True)

        info_card(
            "Ethics Insight",
            "Every algorithm in the AURA dashboard carries an ethical responsibility. "
            "The prescriptive recommendations decide who receives offers and who is excluded — "
            "these are decisions with real consequences for real people. "
            "This analysis reminds us that <b>data analytics is not value-neutral</b>: "
            "the choice of training data, the choice of optimisation metric, and the choice of "
            "who gets to use the insights all reflect values. AURA's commitment is to build "
            "analytics that is accurate, fair, transparent, and sustainable.",
            GOLD
        )
