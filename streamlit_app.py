# ─────────────────────────────────────────────────────────
# SOL Live — Kraken (화이트톤 + 레벨 오버레이 + 신호 요약 + PnL)
# ─────────────────────────────────────────────────────────
import time, requests, math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= UI / Theme =================
st.set_page_config(page_title="SOL Live (Kraken)", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1100px;padding-top:.6rem;padding-bottom:2rem}
h1,h2,h3,h4{letter-spacing:-.02em}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{background:#fff;border:1px solid #eee;border-radius:18px;padding:14px 16px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card h4{margin:0;color:#667085;font-size:.9rem;font-weight:600}
.card .v{margin-top:6px;font-size:1.35rem;font-weight:800;color:#101828}
.badge{display:inline-block;padding:.38rem .6rem;border-radius:999px;font-weight:800;font-size:.85rem}
.badge.long{background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46}
.badge.short{background:#fef2f2;border:1px solid #fecaca;color:#991b1b}
.badge.neutral{background:#f3f4f6;border:1px solid #e5e7eb;color:#374151}
thead th{position:sticky;top:0;background:#fff}
.dataframe{border-radius:14px;border:1px solid #eee}
@media (max-width: 640px){
  .grid{grid-template-columns:repeat(2,1fr)}
  .card .v{font-size:1.15rem}
}
</style>
""", unsafe_allow_html=True)

# ================= Data (Kraken) =================
INTERVAL_MIN = {"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}

def _session():
    s = requests.Session()
    s.headers.update({"User-Agent":"streamlit-sol-live/3.0"})
    retry = Retry(total=3, backoff_factor=0.8,
                  status_forcelist=[429,500,502,503,504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

@st.cache_data(ttl=180, show_spinner=False)
def klines_kraken(symbol="SOLUSD", interval="15m", limit=500):
    url = "https://api.kraken.com/0/public/OHLC"
    r = _session().get(url, params={"pair":symbol, "interval":INTERVAL_MIN[interval]}, timeout=10)
    r.raise_for_status()
    js = r.json()
    if js.get("error"): raise RuntimeError(js["error"])
    key = [k for k in js["result"].keys() if k!="last"][0]
    rows = js["result"][key]
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","vwap","volume","count"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    for c in ["open","high","low","close","vwap","volume"]:
        df[c] = df[c].astype(float)
    df = df.tail(limit).rename(columns={"ts":"time"})
    return df[["time","open","high","low","close","volume"]]

@st.cache_data(ttl=60, show_spinner=False)
def ticker_24h(symbol="SOLUSD"):
    """Kraken Ticker: 현재가/시가로 24h 변화율 근사치"""
    url = "https://api.kraken.com/0/public/Ticker"
    r = _session().get(url, params={"pair":symbol}, timeout=8)
    r.raise_for_status()
    js = r.json()
    if js.get("error"): raise RuntimeError(js["error"])
    key = list(js["result"].keys())[0]
    t = js["result"][key]
    last = float(t["c"][0])
    openp = float(t["o"])
    change = (last-openp)/openp*100 if openp else 0.0
    return last, openp, change

# ================= Indicators =================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9)
    return 100-(100/(1+rs))
def macd(s, f=12, slow=26, sig=9):
    m=ema(s,f)-ema(s,slow); sg=ema(m,sig); return m, sg, m-sg
def atr(high, low, close, n=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(),
                    (high-prev_close).abs(),
                    (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

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

# ================= Sidebar =================
with st.sidebar:
    symbol   = st.selectbox("거래쌍 (Kraken)", ["SOLUSD","SOLUSDT"], index=0)
    interval = st.selectbox("봉 간격", list(INTERVAL_MIN.keys()), index=1)

    refresh  = st.slider("자동 새로고침(초)", 0, 300, 60)
    st.caption("요청 과다 방지: 15초 이상 권장")

    st.markdown("---")
    st.subheader("전략 파라미터")
    entry_long  = st.number_input("롱 진입가", 183.0, step=0.1)
    stop_long   = st.number_input("롱 손절가", 180.0, step=0.1)
    entry_short = st.number_input("숏 진입가", 182.0, step=0.1)
    stop_short  = st.number_input("숏 손절가", 184.0, step=0.1)
    tp_factor   = st.slider("목표가(ATR 배수)", 0.5, 5.0, 2.0, step=0.5)
    lev = st.slider("레버리지(x)", 1, 20, 5)
    pos = st.number_input("포지션 금액(USDT)", 100.0, step=10.0)

# ================= Load & Calc =================
st.subheader("데이터 소스: Kraken (캐시 3분)")

try:
    df = klines_kraken(symbol, interval, 500)
    last_px, open_px, chg24 = ticker_24h(symbol)
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

df["MA20"]=df["close"].rolling(20).mean()
df["MA50"]=df["close"].rolling(50).mean()
m, s, h = macd(df["close"])
df["MACD"], df["SIGNAL"], df["HIST"] = m, s, h
df["RSI14"]=rsi(df["close"])
df["ATR14"]=atr(df["high"], df["low"], df["close"], 14)

price=float(df["close"].iloc[-1]); last=df.iloc[-1]
low24=float(df["low"].tail(96).min())   # 대략 24h(15m 기준 96개) 근사
high24=float(df["high"].tail(96).max())
atr14=float(last["ATR14"]) if not math.isnan(last["ATR14"]) else 0.0

# ATR 기반 제안 레벨
long_tgts  = [round(price + atr14*tp_factor*i, 2) for i in [1,1.5,2]]
short_tgts = [round(price - atr14*tp_factor*i, 2) for i in [1,1.5,2]]

# ================= KPIs =================
st.markdown('<div class="grid">', unsafe_allow_html=True)
kpis = [
    ("현재가", f"{price:,.3f} USD"),
    ("24h 변화", f"{chg24:+.2f}%"),
    ("24h 고가", f"{high24:,.2f}"),
    ("24h 저가", f"{low24:,.2f}"),
    ("RSI(14)", f"{last['RSI14']:.1f}"),
    ("ATR(14)", f"{atr14:.3f}"),
    ("MA20", f"{last['MA20']:.2f}"),
    ("MA50", f"{last['MA50']:.2f}"),
]
# 2행으로 렌더
for title, val in kpis[:4]:
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="v">{val}</div></div>', unsafe_allow_html=True)
for title, val in kpis[4:8]:
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="v">{val}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================= Main Chart (with levels via Altair) =================
base = alt.Chart(df).encode(x='time:T')
price_line = base.mark_line().encode(y='close:Q')
ma20_line  = base.mark_line(strokeDash=[3,3]).encode(y='MA20:Q')
ma50_line  = base.mark_line(strokeDash=[6,3]).encode(y='MA50:Q')

# 수평 레벨: 진입/손절/목표 (롱/숏 공통 표시)
levels = []
if entry_long: levels.append(("Entry Long", entry_long))
if stop_long:  levels.append(("Stop Long",  stop_long))
if entry_short:levels.append(("Entry Short",entry_short))
if stop_short: levels.append(("Stop Short", stop_short))
for t in long_tgts:  levels.append((f"TP L {t}", t))
for t in short_tgts: levels.append((f"TP S {t}", t))
lvl_df = pd.DataFrame(levels, columns=["name","y"]) if levels else pd.DataFrame({"name":[],"y":[]})

rule_layer = alt.Chart(lvl_df).mark_rule().encode(y='y:Q', tooltip=['name:N','y:Q'])
label_layer = alt.Chart(lvl_df).mark_text(align='left', dx=5, dy=-5).encode(y='y:Q', text='name:N')

main_chart = alt.layer(price_line, ma20_line, ma50_line, rule_layer, label_layer)\
    .properties(height=320, title=f"{symbol} · {interval}")\
    .interactive()

left, right = st.columns([2,1], gap="large")
with left:
    st.altair_chart(main_chart, use_container_width=True)
    with st.expander("RSI / MACD / HIST", expanded=False):
        st.altair_chart(alt.Chart(df).mark_line().encode(x='time:T', y='RSI14:Q').properties(height=140), use_container_width=True)
        st.altair_chart(alt.layer(
            alt.Chart(df).mark_line().encode(x='time:T', y='MACD:Q'),
            alt.Chart(df).mark_line().encode(x='time:T', y='SIGNAL:Q')
        ).properties(height=140), use_container_width=True)
        st.altair_chart(alt.Chart(df).mark_bar().encode(x='time:T', y='HIST:Q').properties(height=120), use_container_width=True)

# ================= Signals & PnL =================
with right:
    st.subheader("신호 요약")
    long_score  = int(last["MA20"]>last["MA50"]) + int(last["MACD"]>last["SIGNAL"]) + int(last["RSI14"]>50)
    short_score = int(last["MA20"]<last["MA50"]) + int(last["MACD"]<last["SIGNAL"]) + int(last["RSI14"]<50)
    if long_score>short_score:
        st.markdown('<span class="badge long">🟢 롱 우세</span>', unsafe_allow_html=True)
    elif short_score>long_score:
        st.markdown('<span class="badge short">🔴 숏 우세</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge neutral">⚠️ 중립</span>', unsafe_allow_html=True)
    st.write(f"롱 {long_score}/3 · 숏 {short_score}/3")

    st.subheader("손익 시뮬레이션")
    sim = pd.DataFrame(
        pnl_rows_long(entry_long or price, stop_long or (price-atr14), [*long_tgts], lev, pos) +
        pnl_rows_short(entry_short or price, stop_short or (price+atr14), [*short_tgts], lev, pos),
        columns=["방향","목표가","예상 수익률(%)","예상 PnL(USDT)"]
    )
    def _color(v):
        if isinstance(v,(int,float)):
            if v>0: return "color:#065f46;font-weight:700"
            if v<0: return "color:#991b1b;font-weight:700"
        return ""
    st.dataframe(
        sim.style.format({"예상 수익률(%)":"{:.2f}","예상 PnL(USDT)":"{:.2f}"})
           .applymap(_color, subset=["예상 수익률(%)","예상 PnL(USDT)"]),
        use_container_width=True, height=300
    )
    st.caption("※ 정보 제공용입니다. 선물거래는 고위험입니다.")

# ================= Auto refresh =================
if refresh and refresh>0:
    time.sleep(max(refresh, 15))
    st.experimental_rerun()
