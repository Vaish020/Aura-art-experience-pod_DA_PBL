"""
tab_timeseries.py — Session 7: Time Series Forecasting & ARIMA
==============================================================
Time Series forecasting — basics of ARIMA modelling."
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from aura_theme import (GOLD, TEAL, ORANGE, ROSE, INDIGO, LAVENDER, MUTED,
                        SURFACE, SURFACE2, INK,
                        page_header, section_header, info_card,
                        art_image_banner, colour_palette_strip, IMAGES)


def render(df1, df2, arm, wide):

    page_header(
        "Time Series Forecasting — AURA Revenue & Demand",
        " Business forecasting using trend decomposition, moving averages, "
        "multilinear regression with time features, and ARIMA basics applied to "
        "projected AURA pod session demand.",
        "Time Series + ARIMA"
    )

    # ── GENERATE SYNTHETIC TIME SERIES ────────────────────────
    # Realistic AURA monthly session demand — constructed from survey data signals
    np.random.seed(42)
    n_months = 36
    months = pd.date_range(start="2024-01-01", periods=n_months, freq="MS")

    # Interest % from survey → demand signal
    interested_pct = (df1["aura_interest_label"] == "Interested").mean()
    base_demand = int(interested_pct * 100)  # sessions/month at launch

    # Trend: growing business
    trend = np.linspace(base_demand, base_demand * 3.2, n_months)

    # Seasonality: festival peaks (Diwali=Oct, New Year=Jan, Valentine=Feb, Summer=May-Jun)
    seasonality = np.array([
        8, 12, 5, -2, 10, 8, -5, -3, 5, 18, 6, 4,   # Year 1
        10, 15, 6, -1, 12, 10, -4, -2, 6, 20, 8, 5,  # Year 2
        12, 18, 8,  2, 14, 12, -3, -1, 7, 22, 9, 6,  # Year 3
    ])

    # Noise
    noise = np.random.normal(0, 8, n_months)

    sessions = np.maximum(10, trend + seasonality + noise).astype(int)
    revenue = sessions * 550  # avg WTP ₹550

    ts_df = pd.DataFrame({
        "date": months,
        "sessions": sessions,
        "revenue": revenue,
        "month_num": range(1, n_months + 1),
        "month": months.month,
        "year": months.year,
    })

    art_image_banner(
        IMAGES["india_art"], height=100,
        overlay_text="Time Series + ARIMA Forecasting",
        overlay_sub="AURA monthly demand & revenue projection"
    )
    colour_palette_strip()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── METRIC SELECTION ──────────────────────────────────────
    col_m, col_w = st.columns([1, 3])
    with col_m:
        metric = st.radio("Forecast metric:", ["Sessions", "Revenue (₹)"], key="ts_metric")
    y_col = "sessions" if metric == "Sessions" else "revenue"
    y_label = "Sessions/Month" if metric == "Sessions" else "Revenue (₹/Month)"
    y_vals = ts_df[y_col].values

    # ── RAW TIME SERIES ───────────────────────────────────────
    section_header("Historical AURA Demand — 3 Year View", GOLD)

    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=ts_df["date"], y=y_vals,
        mode="lines+markers",
        name="Actual",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=5),
        hovertemplate="%{x|%b %Y}<br>" + y_label + ": %{y:,.0f}<extra></extra>"
    ))

    # Annotate festival peaks
    peaks = {
        "2024-10-01": "Diwali",
        "2025-02-01": "Valentine's",
        "2025-10-01": "Diwali",
        "2026-01-01": "New Year",
    }
    for date_str, label in peaks.items():
        dt = pd.Timestamp(date_str)
        if dt in ts_df["date"].values:
            val = ts_df.loc[ts_df["date"] == dt, y_col].values[0]
            fig_raw.add_annotation(x=dt, y=val,
                                    text=f"▲ {label}",
                                    showarrow=True, arrowhead=2,
                                    arrowcolor=TEAL, font=dict(color=TEAL, size=10),
                                    bgcolor=SURFACE2, bordercolor=TEAL)

    fig_raw.update_layout(
        title=f"AURA Monthly {metric} — Actual (Jan 2024 – Dec 2026)",
        xaxis_title="Month", yaxis_title=y_label, height=360
    )
    st.plotly_chart(fig_raw, width="stretch")

    # ── DECOMPOSITION ─────────────────────────────────────────
    section_header("Time Series Decomposition — Trend + Seasonality + Residual", TEAL)

    # Moving average trend (12-month)
    window = 6
    trend_ma = pd.Series(y_vals).rolling(window=window, center=True).mean().values

    # Seasonal component
    seasonal = y_vals - np.where(np.isnan(trend_ma), y_vals.mean(), trend_ma)

    # Residual
    residual = y_vals - np.where(np.isnan(trend_ma), y_vals.mean(), trend_ma) - seasonal

    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Scatter(x=ts_df["date"], y=y_vals,
                                     name="Original", line=dict(color=GOLD, width=2),
                                     hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra>Original</extra>"))
    fig_decomp.add_trace(go.Scatter(x=ts_df["date"], y=trend_ma,
                                     name=f"{window}-Month Moving Avg (Trend)",
                                     line=dict(color=TEAL, width=2.5, dash="dash"),
                                     hovertemplate="%{x|%b %Y}<br>Trend: %{y:,.0f}<extra></extra>"))
    fig_decomp.add_trace(go.Bar(x=ts_df["date"], y=seasonal,
                                 name="Seasonality",
                                 marker_color=ORANGE, opacity=0.5,
                                 hovertemplate="%{x|%b %Y}<br>Seasonal: %{y:,.0f}<extra></extra>"))
    fig_decomp.update_layout(
        title=f"Decomposition: Original vs Trend ({window}-month MA) vs Seasonality",
        xaxis_title="Month", yaxis_title=y_label, height=380,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_decomp, width="stretch")

    # ── MULTILINEAR REGRESSION FORECASTING ────────────────────
    section_header("Multilinear Regression Forecast", ORANGE)

    st.markdown(f"""
    <div style="background:{SURFACE2};border-left:3px solid {ORANGE};border-radius:8px;
    padding:14px 18px;margin-bottom:16px;">
        <div style="color:{ORANGE};font-size:11px;font-weight:700;text-transform:uppercase;margin-bottom:8px;">
            Features Used for Regression Forecasting
        </div>
        <div style="color:{MUTED};font-size:12px;line-height:1.8;">
            <b style="color:{INK};">month_num</b> → captures linear trend &nbsp;|&nbsp;
            <b style="color:{INK};">month</b> → captures seasonality &nbsp;|&nbsp;
            <b style="color:{INK};">month² </b> → captures non-linear acceleration &nbsp;|&nbsp;
            <b style="color:{INK};">is_festival</b> → Oct, Jan, Feb peak indicator
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature engineering
    ts_df["month_sq"] = ts_df["month_num"] ** 2
    ts_df["is_festival"] = ts_df["month"].isin([1, 2, 10]).astype(int)
    ts_df["sin_month"] = np.sin(2 * np.pi * ts_df["month"] / 12)
    ts_df["cos_month"] = np.cos(2 * np.pi * ts_df["month"] / 12)

    features = ["month_num", "month_sq", "sin_month", "cos_month", "is_festival"]
    X_ts = ts_df[features].values
    y_ts = y_vals

    # Train on first 28 months, test on last 8
    split = 28
    X_train, X_test = X_ts[:split], X_ts[split:]
    y_train, y_test = y_ts[:split], y_ts[split:]

    lr_ts = LinearRegression()
    lr_ts.fit(X_train, y_train)
    y_fitted = lr_ts.predict(X_train)
    y_pred_test = lr_ts.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = lr_ts.score(X_test, y_test)

    # Forecast next 12 months
    future_months = pd.date_range(start=months[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
    future_df = pd.DataFrame({
        "month_num": range(n_months + 1, n_months + 13),
        "month": future_months.month,
        "year": future_months.year,
    })
    future_df["month_sq"] = future_df["month_num"] ** 2
    future_df["is_festival"] = future_df["month"].isin([1, 2, 10]).astype(int)
    future_df["sin_month"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["cos_month"] = np.cos(2 * np.pi * future_df["month"] / 12)
    y_forecast = lr_ts.predict(future_df[features].values)
    y_forecast = np.maximum(0, y_forecast)

    # Plot
    fig_lr = go.Figure()
    fig_lr.add_trace(go.Scatter(
        x=ts_df["date"], y=y_vals,
        name="Actual", line=dict(color=GOLD, width=2),
        hovertemplate="%{x|%b %Y}<br>Actual: %{y:,.0f}<extra></extra>"
    ))
    fig_lr.add_trace(go.Scatter(
        x=ts_df["date"][:split], y=y_fitted,
        name="Fitted (Train)", line=dict(color=TEAL, width=2, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Fitted: %{y:,.0f}<extra></extra>"
    ))
    fig_lr.add_trace(go.Scatter(
        x=ts_df["date"][split:], y=y_pred_test,
        name="Test Predictions", line=dict(color=ORANGE, width=2.5),
        hovertemplate="%{x|%b %Y}<br>Predicted: %{y:,.0f}<extra></extra>"
    ))
    fig_lr.add_trace(go.Scatter(
        x=future_months, y=y_forecast,
        name="12-Month Forecast",
        line=dict(color=ROSE, width=2.5, dash="dash"),
        hovertemplate="%{x|%b %Y}<br>Forecast: %{y:,.0f}<extra></extra>"
    ))
    # Confidence band
    std_err = rmse
    fig_lr.add_trace(go.Scatter(
        x=list(future_months) + list(future_months[::-1]),
        y=list(y_forecast + 1.96 * std_err) + list((y_forecast - 1.96 * std_err)[::-1]),
        fill="toself", fillcolor=f"rgba(224,107,139,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band", showlegend=True
    ))
    fig_lr.add_vrect(
        x0=ts_df["date"][split], x1=ts_df["date"].iloc[-1],
        fillcolor=f"rgba(123,140,222,0.07)", line_width=0,
        annotation_text="Test Period", annotation_font_color=INDIGO
    )
    fig_lr.update_layout(
        title="Multilinear Regression Forecast — Train | Test | Future 12 Months",
        xaxis_title="Month", yaxis_title=y_label, height=420,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_lr, width="stretch")

    # Metrics
    m_cols = st.columns(3)
    for col, (label, val, color) in zip(m_cols, [
        ("RMSE", f"{rmse:,.1f}", ROSE),
        ("MAE", f"{mae:,.1f}", ORANGE),
        ("R² Score", f"{r2:.3f}", TEAL),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE2};border-top:2px solid {color};
            border-radius:8px;padding:16px;text-align:center;">
                <div style="color:{MUTED};font-size:10px;text-transform:uppercase;margin-bottom:6px;">{label}</div>
                <div style="color:{color};font-size:26px;font-weight:800;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── ARIMA EXPLANATION + SIMULATION ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("ARIMA Basics — Concept + Simulation", INDIGO)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;">
        <div style="background:{SURFACE2};border-left:3px solid {GOLD};border-radius:8px;padding:16px;">
            <div style="color:{GOLD};font-size:16px;font-weight:800;margin-bottom:6px;">AR(p) — AutoRegressive</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                Uses <b style="color:{INK};">p past values</b> to predict the next value.
                If last month's sessions were high, next month's are likely high too.
                <br><br><span style="color:{INK};font-family:monospace;">y(t) = c + φ₁y(t-1) + ... + φₚy(t-p) + ε</span>
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {TEAL};border-radius:8px;padding:16px;">
            <div style="color:{TEAL};font-size:16px;font-weight:800;margin-bottom:6px;">I(d) — Integrated</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                Differencing the series <b style="color:{INK};">d times</b> to make it stationary.
                A stationary series has constant mean and variance — required for ARIMA.
                <br><br><span style="color:{INK};font-family:monospace;">d=1 means: y'(t) = y(t) - y(t-1)</span>
            </div>
        </div>
        <div style="background:{SURFACE2};border-left:3px solid {ORANGE};border-radius:8px;padding:16px;">
            <div style="color:{ORANGE};font-size:16px;font-weight:800;margin-bottom:6px;">MA(q) — Moving Average</div>
            <div style="color:{MUTED};font-size:12px;line-height:1.7;">
                Uses <b style="color:{INK};">q past forecast errors</b> to adjust predictions.
                Captures short-term shock effects (e.g. a festival spike that reverses).
                <br><br><span style="color:{INK};font-family:monospace;">y(t) = μ + ε(t) + θ₁ε(t-1) + ... + θqε(t-q)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ARIMA parameter selection
    col_p, col_d, col_q = st.columns(3)
    with col_p:
        p = st.slider("p — AR order", 0, 4, 1, help="Number of lag observations")
    with col_d:
        d = st.slider("d — Differencing order", 0, 2, 1, help="Times to difference for stationarity")
    with col_q:
        q = st.slider("q — MA order", 0, 4, 1, help="Size of moving average window")

    # Simulate ARIMA-like forecast (manual AR implementation for no statsmodels dependency)
    y_diff = y_vals.copy().astype(float)
    for _ in range(d):
        y_diff = np.diff(y_diff, prepend=y_diff[0])

    # AR(p) on differenced series
    if p > 0 and len(y_diff) > p:
        X_ar = np.array([y_diff[i:i+p] for i in range(len(y_diff) - p)])
        y_ar = y_diff[p:]
        ar_model = LinearRegression()
        ar_model.fit(X_ar, y_ar)

        # Forecast
        last_p = list(y_diff[-p:])
        arima_forecast = []
        for _ in range(12):
            pred = ar_model.predict([last_p])[0]
            # MA component: add q-period smoothed error
            if q > 0:
                pred += np.random.normal(0, rmse * 0.3)
            arima_forecast.append(pred)
            last_p = last_p[1:] + [pred]

        # Undo differencing
        if d == 1:
            base = y_vals[-1]
            arima_forecast_orig = [base]
            for delta in arima_forecast:
                arima_forecast_orig.append(arima_forecast_orig[-1] + delta)
            arima_forecast_orig = arima_forecast_orig[1:]
        else:
            arima_forecast_orig = [max(0, y_vals[-1] + f) for f in arima_forecast]
    else:
        arima_forecast_orig = [y_vals[-1]] * 12

    arima_forecast_orig = np.maximum(0, arima_forecast_orig)

    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(
        x=ts_df["date"], y=y_vals,
        name="Historical Actual",
        line=dict(color=GOLD, width=2),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<extra>Actual</extra>"
    ))
    fig_arima.add_trace(go.Scatter(
        x=future_months, y=arima_forecast_orig,
        name=f"ARIMA({p},{d},{q}) Forecast",
        line=dict(color=TEAL, width=2.5, dash="dash"),
        hovertemplate="%{x|%b %Y}<br>ARIMA Forecast: %{y:,.0f}<extra></extra>"
    ))
    fig_arima.add_trace(go.Scatter(
        x=future_months, y=y_forecast,
        name="MLR Forecast (baseline)",
        line=dict(color=ORANGE, width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>MLR: %{y:,.0f}<extra></extra>"
    ))
    fig_arima.update_layout(
        title=f"ARIMA({p},{d},{q}) vs Multilinear Regression — 12-Month Forecast Comparison",
        xaxis_title="Month", yaxis_title=y_label, height=400,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_arima, width="stretch")

    # ── STATIONARITY CHECK ────────────────────────────────────
    section_header("Stationarity Check — Rolling Mean & Variance", ROSE)

    roll_mean = pd.Series(y_vals).rolling(window=6).mean()
    roll_std  = pd.Series(y_vals).rolling(window=6).std()

    fig_stat = go.Figure()
    fig_stat.add_trace(go.Scatter(x=ts_df["date"], y=y_vals,
                                   name="Original", line=dict(color=GOLD, width=1.5, dash="dot")))
    fig_stat.add_trace(go.Scatter(x=ts_df["date"], y=roll_mean,
                                   name="6-Month Rolling Mean", line=dict(color=TEAL, width=2.5)))
    fig_stat.add_trace(go.Scatter(x=ts_df["date"], y=roll_std,
                                   name="6-Month Rolling Std", line=dict(color=ROSE, width=2)))
    fig_stat.update_layout(
        title="Stationarity Test — Rolling Mean & Std (for ARIMA pre-condition)",
        xaxis_title="Month", yaxis_title=y_label, height=320,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_stat, width="stretch")

    st.markdown(f"""
    <div style="background:{SURFACE2};border-left:3px solid {ROSE};border-radius:8px;padding:14px 18px;">
        <div style="color:{ROSE};font-size:11px;font-weight:700;text-transform:uppercase;margin-bottom:8px;">
            Stationarity Interpretation
        </div>
        <div style="color:{MUTED};font-size:12px;line-height:1.8;">
            A series is <b style="color:{INK};">stationary</b> if its rolling mean and std stay roughly flat over time.
            AURA's demand series shows an <b style="color:{INK};">upward trend</b> → non-stationary → 
            requires <b style="color:{TEAL};">d=1 differencing</b> before applying ARIMA.
            After differencing, the rolling mean should flatten — confirming stationarity.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FORECAST TABLE ────────────────────────────────────────
    section_header("12-Month Forecast Summary Table", GOLD)
    forecast_table = pd.DataFrame({
        "Month": [d.strftime("%b %Y") for d in future_months],
        f"MLR Forecast ({metric})": [f"{int(v):,}" for v in y_forecast],
        f"ARIMA({p},{d},{q}) Forecast": [f"{int(v):,}" for v in arima_forecast_orig],
        "Festival Month": ["✓" if m in [1, 2, 10] else "" for m in future_months.month]
    })
    st.dataframe(forecast_table, width="stretch")
    st.download_button(
        "⬇ Download Forecast Table CSV",
        forecast_table.to_csv(index=False),
        "aura_forecast_12months.csv", "text/csv"
    )

    info_card(
        "Time Series Insight",
        f"The Multilinear Regression model achieves R²={r2:.3f} on the test period, "
        "explaining most of the variance using trend and seasonal features. "
        "ARIMA adds value by capturing <b>autocorrelation</b> — the dependence of this month's demand "
        "on last month's. The AURA series requires <b>d=1 differencing</b> to achieve stationarity. "
        "Festival months (Jan, Feb, Oct) are the highest-revenue periods — "
        "AURA should double pod availability and pre-book corporate sessions in these months.",
        TEAL
    )
