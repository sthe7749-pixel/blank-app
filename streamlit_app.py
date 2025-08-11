# ─────────────────────────────────────────────────────────
# SOL Live — 모바일 친화 디자인 (Streamlit Cloud용)
# 파일명: streamlit_app.py
# 의존성: requirements.txt -> streamlit, pandas, numpy, requests
# ─────────────────────────────────────────────────────────
import time, requests
import pandas as pd
import numpy as np
import streamlit as st

# ---------- Page Setup ----------
st.set_page_config(page_title="SOL Live", page_icon="📈", layout="wide")
st.markdown("""
<style>
/* 모바일 가독성 업 */
:root { --fg:#e5e7eb; --bg:#0b0f1a; --card:#111827; --accent:#22c55e; }
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1200px}
h1,h2,h3{letter-spacing:.2px}
.kpi{background:var(--card);padding:14px 16px;border-radius:16px;border:1px solid rgba(255,255,255,.06)}
.kpi h3{font-size:14px;margin:0;color:#9ca3af}
.kpi .v{font-weight:700;font-size:22px;margin-top:6px}
hr{opacity:.15}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def klines(symbol="SOLUSDT", interval="15m", limit=500):
    """Binance Klines → DataFrame"""
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":limit}, timeout=12)
    r.raise_for_status()
    cols = ["t","o","h","l","c","v","ct","q","n","tb","tq","i"]
    df = pd.DataFrame(r.json(), columns=cols)
    df["c"]=df["c"].astype(float); df["h"]=df["h"].astype(float); df["l"]=df["l"].astype(float); df["v"]=df["v"].astype(float)
    df["ct"]=pd.to_datetime(df["ct"], unit="ms")
    return df.rename(columns={"ct":"time","c":"close","h":"high","l":"low","v":"volume"})

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=(-d.clip(upper=0))
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9)
    return 100-(100/(1+rs))
def macd(s, f=12, slow=26, sig=9):
    m=ema(s,f)-ema(s,slow); sg=ema(m,sig); return m, sg, m-sg

def pnl_rows_long(entry, stop, tgts, lev, pos):
    rows=[]; 
    for t in tgts:
        pct=((t-entry)/entry)*lev*100; rows.append(["롱", t, pct, pos*(pct/100)])
    loss=((stop-entry)/entry)*lev*100; rows.append(["롱", stop, loss, pos*(loss/100)])
    return rows

def pnl_rows_short(entry, stop, tgts, lev, pos):
    rows=[]; 
    for t in tgts:
        pct=((entry-t)/entry)*lev*100; rows.append(["숏", t, pct, pos*(pct/100)])
    loss=((stop-entry)/entry)*lev*100; rows.append(["숏", stop, -abs(loss), -abs(pos*(loss/100))])
    return rows

# ---------- Header ----------
st.title("📈 SOL 롱·숏 라이브")
st.caption("Binance 가격 기반 · 실시간 지표/시그널 · 모바일 최적화 UI")

# ---------- Sidebar (Controls) ----------
with st.sidebar:
    st.subheader("설정")
    symbol   = st.text_input("심볼", "SOLUSDT")
    interval = st.selectbox("봉 간격", ["5m","15m","30m","1h","4h","1d"], index=1)
    refresh  = st.slider("자동 새로고침(초)", 0, 120, 10, help="0이면 자동 새로고침 없음")

    st.markdown("---")
    st.subheader("전략 파라미터")
    entry_long  = st.number_input("롱 진입가", 183.0, step=0.1)
    stop_long   = st.number_input("롱 손절가", 180.0, step=0.1)
    entry_short = st.number_input("숏 진입가", 182.0, step=0.1)
    stop_short  = st.number_input("숏 손절가", 184.0, step=0.1)
    lev = st.slider("레버리지(x)", 1, 20, 5)
    pos = st.number_input("포지션 금액(USDT)", 100.0, step=10.0)

# ---------- Data & Indicators ----------
def render_once():
    df = klines(symbol, interval, 500)
    df["MA20"]=df["close"].rolling(20).mean()
    df["MA50"]=df["close"].rolling(50).mean()
    mac, sig, hist = macd(df["close"])
    df["MACD"], df["SIGNAL"], df["HIST"] = mac, sig, hist
    df["RSI14"]=rsi(df["close"])
    price = float(df["close"].iloc[-1])
    last  = df.iloc[-1]

    # KPI Row
    k1,k2,k3,k4 = st.columns(4)
    for box, title, val in [
        (k1,"현재가", f"{price:,.3f} USDT"),
        (k2,"RSI(14)", f"{last['RSI14']:.1f}"),
        (k3,"MA20", f"{last['MA20']:.2f}"),
        (k4,"MA50", f"{last['MA50']:.2f}"),
    ]:
        with box: st.markdown(f'<div class="kpi"><h3>{title}</h3><div class="v">{val}</div></div>', unsafe_allow_html=True)

    # Charts
    left, right = st.columns([2,1])
    with left:
        st.subheader(f"{symbol} · {interval}")
        st.line_chart(df.set_index("time")[["close","MA20","MA50"]])
        with st.expander("RSI / MACD", expanded=False):
            st.line_chart(df.set_index("time")[["RSI14"]])
            st.line_chart(df.set_index("time")[["MACD","SIGNAL"]])
            st.bar_chart(df.set_index("time")[["HIST"]])

    # Signals + PnL
    with right:
        st.subheader("신호 요약")
        long_score  = int(last["MA20"]>last["MA50"]) + int(last["MACD"]>last["SIGNAL"]) + int(last["RSI14"]>50)
        short_score = int(last["MA20"]<last["MA50"]) + int(last["MACD"]<last["SIGNAL"]) + int(last["RSI14"]<50)
        badge = "🟢 롱 우세" if long_score>short_score else ("🔴 숏 우세" if short_score>long_score else "⚠️ 중립")
        st.markdown(f"**롱 점수:** {long_score}/3 &nbsp;&nbsp; **숏 점수:** {short_score}/3")
        st.success(badge)

        st.subheader("손익 시뮬레이션")
        long_rows  = pnl_rows_long(entry_long,  stop_long,  [190,200,210], lev, pos)
        short_rows = pnl_rows_short(entry_short, stop_short, [175,170,165], lev, pos)
        sim = pd.DataFrame(long_rows+short_rows, columns=["방향","목표가","예상 수익률(%)","예상 PnL(USDT)"])
        st.dataframe(sim.style.format({"예상 수익률(%)":"{:.2f}","예상 PnL(USDT)":"{:.2f}"}), use_container_width=True)
        st.caption("※ 정보 제공용입니다. 선물거래는 고위험입니다.")

render_once()
if refresh and refresh>0:
    while True:
        time.sleep(refresh)
        st.experimental_rerun()
