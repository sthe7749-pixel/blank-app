# ─────────────────────────────────────────────────────────
# SOL Live — Binance 우선, 실패 시 CoinGecko 자동 대체
# 파일: streamlit_app.py
# ─────────────────────────────────────────────────────────
import math, time, requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="SOL Long/Short Live", page_icon="📈", layout="wide")
st.title("📈 SOL 롱·숏 라이브 — Binance ↔ CoinGecko")

# ---------- Sidebar ----------
with st.sidebar:
    symbol = st.text_input("심볼", "SOLUSDT")
    interval = st.selectbox("봉 간격", ["5m","15m","30m","1h","4h","1d"], index=1)
    entry_long  = st.number_input("롱 진입가", 183.0, step=0.1)
    stop_long   = st.number_input("롱 손절가", 180.0, step=0.1)
    entry_short = st.number_input("숏 진입가", 182.0, step=0.1)
    stop_short  = st.number_input("숏 손절가", 184.0, step=0.1)
    lev = st.slider("레버리지(x)", 1, 20, 5)
    pos = st.number_input("포지션 금액(USDT)", 100.0, step=10.0)
    refresh = st.slider("자동 새로고침(초)", 0, 120, 10)
    targets_long  = [190, 200, 210]
    targets_short = [175, 170, 165]

# ---------- Data Sources ----------
def klines_binance(symbol="SOLUSDT", interval="15m", limit=500):
    """Binance Klines → DataFrame"""
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol":symbol, "interval":interval, "limit":limit}, timeout=10)
    r.raise_for_status()
    cols = ["t","o","h","l","c","v","ct","q","n","tb","tq","i"]
    df = pd.DataFrame(r.json(), columns=cols)
    df = df.rename(columns={"ct":"time","c":"close","h":"high","l":"low","v":"volume"})
    for col in ["close","high","low","volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df[["time","close","high","low","volume"]]

def klines_coingecko(symbol="SOLUSDT", interval="15m", limit=500):
    """CoinGecko 종가 기반 대체 소스(SOL 전용)"""
    # 간단 매핑: 현재 SOL만 지원
    coin_id = "solana"
    minutes = {"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}[interval]
    need_min = limit * minutes
    days = max(1, min(90, math.ceil(need_min / (60*24))))  # CoinGecko 최대 90일
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = requests.get(url, params={"vs_currency":"usd","days":days,"interval":"minute"}, timeout=12)
    r.raise_for_status()
    prices = r.json().get("prices", [])  # [[ts, price], ...]
    if not prices:
        raise RuntimeError("CoinGecko: 데이터 없음")
    df = pd.DataFrame(prices, columns=["ms","close"])
    df["time"] = pd.to_datetime(df["ms"], unit="ms")
    # 원하는 봉 간격으로 리샘플
    df = df.set_index("time").resample(interval).last().dropna().reset_index()
    df["high"] = df["close"]; df["low"] = df["close"]; df["volume"] = np.nan
    return df[["time","close","high","low","volume"]]

def load_data(symbol, interval, limit=500):
    """Binance 우선 시도, 실패 시 CoinGecko로 자동 대체"""
    try:
        df = klines_binance(symbol, interval, limit)
        st.caption("데이터 소스: Binance")
        return df
    except Exception as e:
        st.warning(f"Binance 오류: {e} → CoinGecko로 대체합니다.")
        df = klines_coingecko(symbol, interval, limit)
        st.caption("데이터 소스: CoinGecko (종가 기반)")
        return df

# ---------- Indicators ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = up.rolling(n).mean() / (dn.rolling(n).mean() + 1e-9)
    return 100 - (100/(1+rs))
def macd(s, f=12, slow=26, sig=9):
    m = ema(s,f) - ema(s,slow); sg = ema(m,sig); return m, sg, m - sg

def pnl_rows_long(entry, stop, tgts, lev, pos):
    rows=[]
    for t in tgts:
        pct=((t-entry)/entry)*lev*100; rows.append(["롱", t, pct, pos*(pct/100)])
    loss=((stop-entry)/entry)*lev*100; rows.append(["롱", stop, loss, pos*(loss/100)])
    return rows

def pnl_rows_short(entry, stop, tgts, lev, pos):
    rows=[]
    for t in tgts:
        pct=((entry-t)/entry)*lev*100; rows.append(["숏", t, pct, pos*(pct/100)])
    loss=((stop-entry)/entry)*lev*100; rows.append(["숏", stop, -abs(loss), -abs(pos*(loss/100))])
    return rows

# ---------- Render ----------
def render_once():
    df = load_data(symbol, interval, 500)

    # 지표
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    mac, sig, hist = macd(df["close"])
    df["MACD"], df["SIGNAL"], df["HIST"] = mac, sig, hist
    df["RSI14"] = rsi(df["close"])
    price = float(df["close"].iloc[-1]); last = df.iloc[-1]

    # KPI
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("현재가", f"{price:,.3f} USDT")
    c2.metric("RSI(14)", f"{last['RSI14']:.1f}")
    c3.metric("MA20", f"{last['MA20']:.2f}")
    c4.metric("MA50", f"{last['MA50']:.2f}")

    left, right = st.columns([2,1])
    with left:
        st.subheader(f"{symbol} · {interval}")
        st.line_chart(df.set_index("time")[["close","MA20","MA50"]])
        with st.expander("RSI / MACD"):
            st.line_chart(df.set_index("time")[["RSI14"]])
            st.line_chart(df.set_index("time")[["MACD","SIGNAL"]])
            st.bar_chart(df.set_index("time")[["HIST"]])

    with right:
        st.subheader("신호 요약")
        long_score  = int(last["MA20"]>last["MA50"]) + int(last["MACD"]>last["SIGNAL"]) + int(last["RSI14"]>50)
        short_score = int(last["MA20"]<last["MA50"]) + int(last["MACD"]<last["SIGNAL"]) + int(last["RSI14"]<50)
        badge = "🟢 롱 우세" if long_score>short_score else ("🔴 숏 우세" if short_score>long_score else "⚠️ 중립")
        st.write(f"롱 점수: **{long_score}/3** · 숏 점수: **{short_score}/3**")
        st.success(badge)

        st.subheader("손익 시뮬레이션")
        sim = pd.DataFrame(
            pnl_rows_long(entry_long, stop_long, targets_long, lev, pos) +
            pnl_rows_short(entry_short, stop_short, targets_short, lev, pos),
            columns=["방향","목표가","예상 수익률(%)","예상 PnL(USDT)"]
        )
        st.dataframe(sim.style.format({"예상 수익률(%)":"{:.2f}","예상 PnL(USDT)":"{:.2f}"}), use_container_width=True)
        st.caption("※ 정보 제공용. 선물거래는 고위험입니다.")

render_once()
if refresh and refresh>0:
    while True:
        time.sleep(refresh)
        st.experimental_rerun()
