import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# ===== 설정 =====
INTERVAL_MIN = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240
}

# ===== 데이터 불러오기 함수 =====
def get_data(symbol="SOLUSD", interval="15m", limit=500):
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={INTERVAL_MIN[interval]}"
    r = requests.get(url)
    data = r.json()
    key = list(data['result'].keys())[0]
    df = pd.DataFrame(data['result'][key], columns=[
        "time","open","high","low","close","vwap","volume","count"
    ])
    df["time"] = pd.to_datetime(df["time"], unit='s')
    df[["open","high","low","close","vwap","volume"]] = df[["open","high","low","close","vwap","volume"]].astype(float)
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["ATR14"] = (df["high"] - df["low"]).rolling(14).mean()
    return df

# ===== Streamlit 앱 =====
st.set_page_config(page_title="SOL 롱·숏 라이브", layout="wide")

st.title("📈 SOL 롱·숏 라이브 — Kraken")

# ---- 사이드바 ----
with st.sidebar:
    st.header("차트 설정")
    interval = st.selectbox("Interval", list(INTERVAL_MIN.keys()), index=2)
    show_count = st.slider("표시 캔들 개수", 50, 500, 200, 10)
    
    st.markdown("---")
    st.subheader("예측(실험)")
    show_forecast = st.toggle("예측선 표시", value=True)
    fit_len = st.slider("학습 구간(봉)", 50, 300, 180, 10)
    steps_ahead = st.slider("예측 길이(봉)", 5, 60, 20, 5)
    band_k = st.slider("불확실성 대역(ATR 배수)", 0.0, 2.0, 0.8, 0.1)

# ---- 데이터 로드 ----
df = get_data(interval=interval)
plot_df = df.tail(int(show_count)).copy()

# ---- 예측 계산 ----
forecast_df = pd.DataFrame()
if show_forecast and len(plot_df) > fit_len + 5:
    x = np.arange(len(plot_df))
    y = plot_df["close"].to_numpy()
    x_fit = x[-fit_len:]
    y_fit = y[-fit_len:]
    a, b = np.polyfit(x_fit, y_fit, 1)

    step_min = INTERVAL_MIN[interval]
    future_idx = np.arange(x[-1] + 1, x[-1] + 1 + steps_ahead)
    future_time = pd.date_range(
        plot_df["time"].iloc[-1] + pd.Timedelta(minutes=step_min),
        periods=steps_ahead, freq=f"{step_min}min"
    )
    y_pred = a * future_idx + b

    band = float(plot_df["ATR14"].iloc[-1]) * band_k if "ATR14" in plot_df and not np.isnan(plot_df["ATR14"].iloc[-1]) else 0.0
    upper = y_pred + band
    lower = y_pred - band

    forecast_df = pd.DataFrame({"time": future_time, "pred": y_pred, "upper": upper, "lower": lower})

# ---- Plotly 차트 ----
fig = go.Figure(data=[
    go.Candlestick(
        x=plot_df["time"],
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        name="Price"
    ),
    go.Scatter(x=plot_df["time"], y=plot_df["MA20"], mode="lines", name="MA20", line=dict(dash="dot")),
    go.Scatter(x=plot_df["time"], y=plot_df["MA50"], mode="lines", name="MA50", line=dict(dash="dash"))
])

# 예측선 추가
if not forecast_df.empty:
    fig.add_trace(go.Scatter(
        x=forecast_df["time"], y=forecast_df["pred"],
        mode="lines", name="Forecast",
        line=dict(width=2, dash="dot", color="blue")
    ))
    if band_k > 0:
        fig.add_trace(go.Scatter(
            x=forecast_df["time"], y=forecast_df["lower"],
            mode="lines", name="Forecast−", line=dict(width=0.5, dash="dot"),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["time"], y=forecast_df["upper"],
            mode="lines", name="Forecast+",
            line=dict(width=0.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(100,149,237,0.12)",
            showlegend=False
        ))

fig.update_layout(xaxis_rangeslider_visible=False, height=600)
st.plotly_chart(fig, use_container_width=True)
