# ─────────────────────────────────────────────────────────
# SOL Live — Kraken + 타이밍 엔진 + 텔레그램 알림 + 로그 저장
# (화이트톤 UI, KPI 카드, 레벨 오버레이, 신호 요약, 안전한 PnL 시뮬레이션, CSV 다운로드)
# ─────────────────────────────────────────────────────────
import os, math, time, datetime as dt, requests
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
.dataframe{border:1px solid #eee;border-radius:14px}
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
    s.headers.update({"User-Agent":"streamlit-sol-live/3.2"})
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
    url = "https://api.kraken.com/0/public/Ticker"
    r = _session().get(url, params={"pair":symbol}, timeout=8)
    r.raise_for_status()
    js = r.json()
    if js.get("error"): raise RuntimeError(js["error"])
    key = list(js["result"].keys())[0]
    t = js["result"][key]
    last = float(t["c"][0]); openp = float(t["o"])
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

# ================= Alerts & Logging Helpers =================
TG_TOKEN   = st.secrets.get("TELEGRAM_TOKEN", "")
TG_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")

def send_telegram(text: str):
    if not (TG_TOKEN and TG_CHAT_ID):
        return False, "No Telegram secrets"
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TG_CHAT_ID, "text": text}, timeout=8)
        return (r.status_code == 200), r.text
    except Exception as e:
        return False, str(e)

if "signal_log" not in st.session_state:
    st.session_state.signal_log = pd.DataFrame(columns=[
        "time","symbol","interval","signal","entry","stop","price","rsi","ma20","ma50"
    ])

def append_signal_log(signal, entry_px, stop_px, price, rsi, ma20, ma50, symbol, interval):
    row = {
        "time": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbol": symbol, "interval": interval, "signal": signal,
        "entry": entry_px, "stop":  stop_px, "price": price,
        "rsi":   rsi, "ma20": ma20, "ma50": ma50
    }
    st.session_state.signal_log = pd.concat(
        [st.session_state.signal_log, pd.DataFrame([row])], ignore_index=True
    )

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
    st.markdown("---")
    st.subheader("알림 & 로깅")
    enable_alerts = st.toggle("텔레그램 알림 켜기", value=False,
                              help="Settings ▸ Secrets에 TELEGRAM_TOKEN/TELEGRAM_CHAT_ID 저장 필요")
    notify_buy  = st.checkbox("BUY 신호 알림", value=True)
    notify_sell = st.checkbox("SELL 신호 알림", value=True)

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
low24=float(df["low"].tail(96).min())   # 약 24h (15m 기준 96개) 근사
high24=float(df["high"].tail(96).max())
atr14=float(last["ATR14"]) if not math.isnan(last["ATR14"]) else 0.0

# ATR 기반 목표가(가이드)
long_tgts  = [round(price + atr14*tp_factor*i, 2) for i in [1,1.5,2]]
short_tgts = [round(price - atr14*tp_factor*i, 2) for i in [1,1.5,2]]

# ================= Timing Engine =================
def score_long(row):
    s  = 0
    s += 1 if row["MA20"] > row["MA50"] else 0
    s += 1 if row["MACD"] > row["SIGNAL"] else 0
    s += 1 if 48 <= row["RSI14"] <= 60 else 0
    return s
def score_short(row):
    s  = 0
    s += 1 if row["MA20"] < row["MA50"] else 0
    s += 1 if row["MACD"] < row["SIGNAL"] else 0
    s += 1 if 40 <= row["RSI14"] <= 52 else 0
    return s

long_ok_trend   = (last["MA20"] > last["MA50"]) and (last["MACD"] > last["SIGNAL"])
short_ok_trend  = (last["MA20"] < last["MA50"]) and (last["MACD"] < last["SIGNAL"])
pullback_ok     = abs(price - last["MA20"]) <= (0.35 * atr14 if atr14>0 else 9999)
rally_ok        = abs(price - last["MA20"]) <= (0.35 * atr14 if atr14>0 else 9999)

long_entry  = long_ok_trend  and pullback_ok
short_entry = short_ok_trend and rally_ok

long_entry_px  = round(last["MA20"], 2) if long_entry else None
long_stop_px   = round(price - 1.2*atr14, 2) if long_entry and atr14>0 else None
short_entry_px = round(last["MA20"], 2) if short_entry else None
short_stop_px  = round(price + 1.2*atr14, 2) if short_entry and atr14>0 else None

# 신호 중복 발송 방지 + 알림/로그
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

new_signal = "BUY" if long_entry else ("SELL" if short_entry else None)
if new_signal and new_signal != st.session_state.last_signal:
    msg = (f"[{symbol}·{interval}] {new_signal} 신호\n"
           f"가격: {price:.3f}\n"
           f"진입: {long_entry_px or short_entry_px}\n"
           f"손절: {long_stop_px or short_stop_px}\n"
           f"RSI: {last['RSI14']:.1f}  MA20:{last['MA20']:.2f}  MA50:{last['MA50']:.2f}")
    if enable_alerts and ((new_signal=="BUY" and notify_buy) or (new_signal=="SELL" and notify_sell)):
        ok, detail = send_telegram(msg)
        if ok: st.toast("텔레그램 전송 완료 ✅", icon="✅")
        else:  st.toast(f"텔레그램 실패: {detail}", icon="⚠️")
    append_signal_log(
        new_signal,
        (long_entry_px if new_signal=="BUY" else short_entry_px),
        (long_stop_px  if new_signal=="BUY" else short_stop_px),
        price, float(last["RSI14"]), float(last["MA20"]), float(last["MA50"]),
        symbol, interval
    )
    st.session_state.last_signal = new_signal

# ================= KPIs =================
st.markdown('<div class="grid">', unsafe_allow_html=True)
kpis = [
    ("현재가", f"{price:,.3f} USD"),
    ("24h 변화", f"{((price-open_px)/open_px*100) if open_px else 0:+.2f}%"),
    ("24h 고가", f"{high24:,.2f}"),
    ("24h 저가", f"{low24:,.2f}"),
    ("RSI(14)", f"{last['RSI14']:.1f}"),
    ("ATR(14)", f"{atr14:.3f}"),
    ("MA20", f"{last['MA20']:.2f}"),
    ("MA50", f"{last['MA50']:.2f}"),
]
for title, val in kpis[:4]:
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="v">{val}</div></div>', unsafe_allow_html=True)
for title, val in kpis[4:8]:
    st.markdown(f'<div class="card"><h4>{title}</h4><div class="v">{val}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================= Main Chart with Levels =================
base = alt.Chart(df).encode(x='time:T')
price_line = base.mark_line().encode(y='close:Q')
ma20_line  = base.mark_line(strokeDash=[3,3]).encode(y='MA20:Q')
ma50_line  = base.mark_line(strokeDash=[6,3]).encode(y='MA50:Q')

levels = []
if long_entry:   levels += [("Entry Long", long_entry_px), ("Stop Long", long_stop_px)]
if short_entry:  levels += [("Entry Short", short_entry_px), ("Stop Short", short_stop_px)]
for t in long_tgts:  levels.append((f"TP L {t}", t))
for t in short_tgts: levels.append((f"TP S {t}", t))
lvl_df = pd.DataFrame(levels, columns=["name","y"]) if levels else pd.DataFrame({"name":[],"y":[]})

rule_layer  = alt.Chart(lvl_df).mark_rule().encode(y='y:Q', tooltip=['name:N','y:Q'])
label_layer = alt.Chart(lvl_df).mark_text(align='left', dx=5, dy=-5).encode(y='y:Q', text='name:N')

signals_df = pd.DataFrame({
    "time":[df["time"].iloc[-1]]*2,
    "y":[long_entry_px if long_entry else np.nan, short_entry_px if short_entry else np.nan],
    "label":["BUY","SELL"]
})
point_layer = alt.Chart(signals_df.dropna()).mark_point(size=90).encode(x='time:T', y='y:Q', shape='label:N')
text_layer  = alt.Chart(signals_df.dropna()).mark_text(dy=-10, fontWeight='bold').encode(x='time:T', y='y:Q', text='label:N')

main_chart = alt.layer(price_line, ma20_line, ma50_line, rule_layer, label_layer, point_layer, text_layer)\
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

# ================= Signals & PnL & Timing =================
with right:
    st.subheader("신호 요약")
    long_score  = score_long(last)
    short_score = score_short(last)
    if long_score>short_score:
        st.markdown('<span class="badge long">🟢 롱 우세</span>', unsafe_allow_html=True)
    elif short_score>long_score:
        st.markdown('<span class="badge short">🔴 숏 우세</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge neutral">⚠️ 중립</span>', unsafe_allow_html=True)
    st.write(f"롱 {long_score}/3 · 숏 {short_score}/3")

    st.subheader("손익 시뮬레이션")
    sim_rows = []
    # 롱 시뮬레이션 (사용자가 진입/손절 입력했을 때만)
    if entry_long > 0:
        for t in long_tgts:
            ret = ((t - entry_long) / entry_long) * lev * 100
            pnl = pos * (ret / 100)
            sim_rows.append(["롱", t, ret, pnl])
        if stop_long:
            ret = ((stop_long - entry_long) / entry_long) * lev * 100
            pnl = pos * (ret / 100)
            sim_rows.append(["롱", stop_long, ret, pnl])
    # 숏 시뮬레이션
    if entry_short > 0:
        for t in short_tgts:
            ret = ((entry_short - t) / entry_short) * lev * 100
            pnl = pos * (ret / 100)
            sim_rows.append(["숏", t, ret, pnl])
        if stop_short:
            ret = -abs(((stop_short - entry_short) / entry_short) * lev * 100)
            pnl = -abs(pos * (((stop_short - entry_short) / entry_short) * lev))
            sim_rows.append(["숏", stop_short, ret, pnl])

    sim = pd.DataFrame(sim_rows, columns=["방향","목표가","예상 수익률(%)","예상 PnL(USDT)"])

    def _color(v):
        if isinstance(v,(int,float)):
            if v>0: return "color:#065f46;font-weight:700"
            if v<0: return "color:#991b1b;font-weight:700"
        return ""

    if not sim.empty:
        st.dataframe(
            sim.style.format({"예상 수익률(%)":"{:.2f}","예상 PnL(USDT)":"{:.2f}"})
               .applymap(_color, subset=["예상 수익률(%)","예상 PnL(USDT)"]),
            use_container_width=True, height=300
        )
    else:
        st.caption("포지션 정보가 없어 시뮬레이션을 표시할 수 없습니다.")

    st.subheader("타이밍 제안")
    if long_entry:
        st.markdown('<span class="badge long">✅ BUY(롱) 타이밍</span>', unsafe_allow_html=True)
        st.write(f"- 제안 진입: **{long_entry_px}**  · 손절: **{long_stop_px}**")
        st.write(f"- 목표(예시): {', '.join(map(str,long_tgts))}")
    elif short_entry:
        st.markdown('<span class="badge short">✅ SELL(숏) 타이밍</span>', unsafe_allow_html=True)
        st.write(f"- 제안 진입: **{short_entry_px}**  · 손절: **{short_stop_px}**")
        st.write(f"- 목표(예시): {', '.join(map(str,short_tgts))}")
    else:
        st.markdown('<span class="badge neutral">⏸ 대기</span>', unsafe_allow_html=True)
        st.caption("조건 미충족: 추세/RSI/되돌림 룰이 일치할 때 신호가 뜹니다.")

    st.subheader("신호 로그")
    if not st.session_state.signal_log.empty:
        st.dataframe(st.session_state.signal_log, use_container_width=True, height=220)
        csv = st.session_state.signal_log.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV 다운로드", csv, file_name="signal_log.csv", mime="text/csv")
    else:
        st.caption("아직 기록된 신호가 없습니다.")

# ================= Auto refresh =================
if refresh and refresh>0:
    time.sleep(max(refresh, 15))
    st.experimental_rerun()
