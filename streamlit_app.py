# ─────────────────────────────────────────────────────────
# SOL Live — Kraken 공개 API (빠름, 키 불필요)
# 파일: streamlit_app.py
# ─────────────────────────────────────────────────────────
import time, math, requests
import pandas as pd
import numpy as np
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="SOL Live (Kraken)", page_icon="📈", layout="wide")
st.title("📈 SOL 롱·숏 라이브 — Kraken")

# ── 공용 설정
INTERVAL_MIN = {"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}

def _session():
    s = requests.Session()
    s.headers.update({"User-Agent":"streamlit-sol-live/1.0"})
    retry = Retry(total=3, backoff_factor=0.8,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

@st.cache_data(ttl=180, show_spinner=False)
def klines_kraken(symbol="SOLUSD", interval="15m", limit=500):
    """Kraken OHLC → DataFrame
       반환 컬럼: time, close, high, low, volume
    """
    kr_interval = INTERVAL_MIN[interval]          # Kraken은 분 단위 정수
    pair = symbol.upper()                          # 예: SOLUSD, SOLUSDT
    url = "https://api.kraken.com/0/public/OHLC"
    r = _session().get(url, params={"pair": pair, "interval": kr_interval}, timeout=10)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(js["error"])
    # 결과 키는 종종 변형됨(예: "SOLUSD" 또는 "SOLUSD.d")
    key = [k for k in js["result"].keys() if k != "last"][0]
    rows = js["result"][key]
    if not rows:
        raise RuntimeError("Kraken: 데이터 없음")
    # [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","vwap","volume","count"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    for c in ["open","high","low","close","vwap","volume"]:
        df[c] = df[c].astype(float)
    df = df.tail(limit).rename(columns={"ts":"time"})  # 최신 limit개 사용
    return df[["time","close","high","low","volume"]]

# ── 보조 지표
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9)
    return 100-(100/(1+rs))
def macd(s, f=12, slow=26, sig=9):
    m=ema(s,f)-ema(s,slow); sg=ema(m,sig); return m, sg, m-sg

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

# ── 사이드바 (모바일 최적)
with st.sidebar:
    symbol   = st.selectbox("심볼 (Kraken)", ["SOLUSD","SOLUSDT"], index=0)
    interval = st.selectbox("봉 간격", list(INTERVAL_MIN.keys()), index=1)
    refresh  = st.slider("자동 새로고침(초)", 0, 300, 60)
    if refresh and refresh < 15:
        st.info("요청 과다 방지를 위해 15초 이상을 권장합니다.")
    st.markdown("---")
    st.subheader("전략 파라미터")
    entry_long  = st.number_input("롱 진입가", 183.0, step=0.1)
    stop_long   = st.number_input("롱 손절가", 180.0, step=0.1)
    entry_short = st.number_input("숏 진입가", 182.0, step=0.1)
    stop_short  = st.number_input("숏 손절가", 184.0, step=0.1)
    lev = st.slider("레버리지(x)", 1, 20, 5)
    pos = st.number_input("포지션 금액(USDT)", 100.0, step=10.0)
    targets_long  = [190,200,210]
    targets_short = [175,170,165]

# ── 렌더
def render_once():
    try:
        df = klines_kraken(symbol, interval, 500)
        st.caption("데이터 소스: Kraken (캐시 3분)")
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        st.stop()

    df["MA20"]=df["close"].rolling(20).mean()
    df["MA50"]=df["close"].rolling(50).mean()
    mac, sig, hist = macd(df["close"])
    df["MACD"], df["SIGNAL"], df["HIST"] = mac, sig, hist
    df["RSI14"]=rsi(df["close"])
    price=float(df["close"].iloc[-1]); last=df.iloc[-1]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("현재가", f"{price:,.3f} USD")
    c2.metric("RSI(14)", f"{last['RSI14']:.1f}")
    c3.metric("MA20", f"{last['MA20']:.2f}")
    c4.metric("MA50", f"{last['MA50']:.2f}")

    left,right = st.columns([2,1])
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
    time.sleep(max(refresh, 15))
    st.experimental_rerun()
