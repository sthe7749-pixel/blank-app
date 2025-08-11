# ─────────────────────────────────────────────────────────
# SOL Live — CoinGecko 전용 (429 방지: 캐시 + 재시도 + 요청 제한)
# 파일: streamlit_app.py
# ─────────────────────────────────────────────────────────
import math, time, requests
import pandas as pd
import numpy as np
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================== 기본 설정 ==================
st.set_page_config(page_title="SOL Live (CoinGecko)", page_icon="📈", layout="wide")
st.title("📈 SOL 롱·숏 라이브 — CoinGecko")

# 봉 간격 매핑
RULE   = {"5m":"5T","15m":"15T","30m":"30T","1h":"1H","4h":"4H","1d":"1D"}
MINUTE = {"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}

# ================== 데이터 소스 (CoinGecko) ==================
def _session():
    s = requests.Session()
    s.headers.update({"User-Agent":"streamlit-sol-live/1.0"})
    retry = Retry(
        total=3, backoff_factor=1.2,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

@st.cache_data(ttl=180, show_spinner=False)  # 3분 캐시
def klines_gecko(symbol="SOL", interval="15m", limit=500):
    coin_id = {"SOL":"solana","SOLUSDT":"solana","SOLUSD":"solana"}.get(symbol.upper())
    if not coin_id:
        raise ValueError("현재는 SOL/SOLUSDT만 지원합니다.")

    # 과요청 방지: 최대 6일만 요청
    need_min = min(limit * MINUTE[interval], 60*24*6)
    days = max(1, int(math.ceil(need_min/(60*24))))

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = _session().get(url, params={"vs_currency":"usd","days":days,"interval":"minute"}, timeout=12)
    r.raise_for_status()

    prices = r.json().get("prices", [])  # [[ms, price], ...]
    if not prices: raise RuntimeError("CoinGecko: 데이터 없음")

    df = pd.DataFrame(prices, columns=["ms","close"])
    df["time"] = pd.to_datetime(df["ms"], unit="ms")
    df = (df.set_index("time").resample(RULE[interval]).last().dropna().reset_index())
    df["high"]   = df["close"]
    df["low"]    = df["close"]
    df["volume"] = np.nan
    return df[["time","close","high","low","volume"]]

# ================== 보조 지표 ==================
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

# ================== 사이드바 ==================
with st.sidebar:
    symbol   = st.text_input("심볼", "SOL").upper()
    interval = st.selectbox("봉 간격", list(RULE.keys()), index=1)

    # 429 방지: 30초 이상 권장, 기본 60초
    refresh  = st.slider("자동 새로고침(초)", 0, 300, 60)
    if refresh and refresh < 30:
        st.info("429 방지를 위해 자동 새로고침은 30초 이상을 권장합니다.")

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

# ================== 렌더 ==================
def render_once():
    try:
        df = klines_gecko(symbol, interval, 500)
        st.caption("데이터 소스: CoinGecko (종가 기반, 캐시 3분)")
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        st.stop()

    # 지표
    df["MA20"]=df["close"].rolling(20).mean()
    df["MA50"]=df["close"].rolling(50).mean()
    mac, sig, hist = macd(df["close"])
    df["MACD"], df["SIGNAL"], df["HIST"] = mac, sig, hist
    df["RSI14"]=rsi(df["close"])

    price=float(df["close"].iloc[-1]); last=df.iloc[-1]

    # KPI
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("현재가", f"{price:,.3f} USDT")
    c2.metric("RSI(14)", f"{last['RSI14']:.1f}")
    c3.metric("MA20", f"{last['MA20']:.2f}")
    c4.metric("MA50", f"{last['MA50']:.2f}")

    # 차트/표
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

# 자동 새로고침 루프 (30초 이상 권장)
if refresh and refresh > 0:
    effective = max(refresh, 30)
    while True:
        time.sleep(effective)
        st.experimental_rerun()
