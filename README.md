# AURA India — Art Experience Pod Analytics Dashboard

> A comprehensive data-driven feasibility study for **AURA** —
> India's first subscription-based art experience pod network.
> Full ML analytics pipeline from descriptive to prescriptive.

---

## 🚀 Live App

[aura-art-experience-pod-group.streamlit.app](https://aura-art-experience-pod-group-bmsvkz6fujxlefgly8ouws.streamlit.app/)

---

## 📊 Dashboard — 13 Analytical Tabs

| Tab | Analysis | Key Algorithms |
|---|---|---|
| 🏠 Executive Summary | Overview | KPI cards · Model comparison · Business context |
| 📊 Overview | Descriptive | Demographics · City treemap · Word cloud · Art preferences |
| 🔍 Diagnostic | Diagnostic | Conversion funnel · Correlation heatmap · Barrier analysis |
| 🌳 Decision Tree | Classification | Gini/Entropy · Node map · Business rules extraction |
| 🔗 Association Rules | Market Basket | Apriori · Support · Confidence · Lift |
| 🧩 Clustering | Segmentation | K-Means (silhouette-driven) · PCA · Radar charts |
| 🌿 RFM + Hierarchical | Segmentation | RFM · Dendrogram · Ward linkage · K-Means comparison |
| 📈 Regression | Prediction | RF · Gradient Boosting · Linear · 5-fold CV · Live WTP predictor |
| 📉 Time Series + ARIMA | Forecasting | Decomposition · MLR forecast · ARIMA(p,d,q) · Stationarity |
| 💬 Text Mining + Ethics | NLP + Ethics | Sentiment analysis · Word frequency · AI ethics · ESG scorecard |
| 🤖 Classification Models | Prediction | RF + XGBoost + LR · 5-fold CV · ROC curves · Confusion matrix |
| 🚀 Predict New | Prescriptive | Upload CSV → interest + WTP + persona + action |
| 💡 Prescriptive Strategy | Strategy | A/B pricing simulator · City heatmap · Launch sequence |

---

## 🔑 Key Algorithm Implementations

### K-Means — Pure Silhouette-Driven K
```python
# No manual override — algorithm decides K
best_k = K_range[int(np.argmax(silhouettes))]
```

### 5-Fold Cross-Validation on All Models
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
```

### Decision Tree with Business Rule Extraction
- Interactive depth / criterion / min-samples controls
- Colour-coded text tree + Plotly node map
- Depth vs accuracy overfitting chart

### RFM Segmentation
- Recency · Frequency · Monetary built from survey features
- Champions → Loyal → Potential → At Risk → Lost

### ARIMA Time Series
- Decomposition: trend + seasonality + residual
- Interactive p, d, q sliders
- Stationarity check via rolling mean/std

### Text Mining
- Rule-based NLP sentiment on customer reviews
- Word frequency · Topic detection · Sentiment vs rating

---

## 📁 File Structure

```
app.py                   # Main entry — 13 tabs
aura_theme.py            # Art-inspired design system · Unsplash images · CSS
aura_data.py             # Data loading · Feature engineering · Model training
tab_summary.py           # Executive Summary
tab_overview.py          # Descriptive Analysis
tab_diagnostic.py        # Diagnostic Analysis
tab_decision_tree.py     # Decision Tree
tab_arm.py               # Association Rules (Apriori)
tab_clustering.py        # K-Means Clustering
tab_hierarchical.py      # RFM + Hierarchical Clustering
tab_regression.py        # Regression + Live WTP Predictor
tab_timeseries.py        # Time Series + ARIMA
tab_text_ethics.py       # Text Mining + AI Ethics + ESG
tab_classification.py    # Classification Models (RF · XGB · LR)
tab_predict.py           # New Customer Scoring
tab_prescriptive.py      # Prescriptive Strategy + A/B Simulator
requirements.txt         # Python dependencies
runtime.txt              # Python 3.12
.python-version          # Python 3.12 pin
aura_survey1_n2000.csv   # Main survey — 2,000 respondents
aura_survey2_n1314.csv   # Deep profile — 1,314 respondents
aura_arm_transactions.csv # ARM binary basket matrix
aura_combined_wide.csv   # Merged wide dataset
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/aura-analytics.git
cd aura-analytics
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit>=1.35.0
pandas>=2.2.2
numpy>=1.26.4
plotly>=5.22.0
scikit-learn>=1.5.0
xgboost>=2.0.3
mlxtend>=0.23.1
imbalanced-learn>=0.12.3
scipy>=1.13.0
```

---

## 🎨 AURA Brand Palette

| Colour | Hex | Use |
|---|---|---|
| Gold | `#f0c040` | Primary accent · KPIs · Headers |
| Teal | `#3ecfa8` | Positive signals · Clusters |
| Orange | `#f07840` | Warnings · Secondary |
| Rose | `#e8507a` | Negative · Alerts |
| Indigo | `#8878f0` | ROC curves · Correlation |
| Dark BG | `#0a0608` | Canvas background |

---

*AURA India · Art Experience Pod · Data Analytics Dashboard · 2026*
