# ─────────────────────────────────────────────────────────
# SOL Live — Kraken
# Plotly 캔들(자동 확대) + MA/RSI/MACD/ATR
# 타이밍 제안(BUY/SELL) + 손익 시뮬레이션 + 신호 로그 + 텔레그램 알림(선택)
# 선형회귀 예측선(+ ATR 대역) · 보조지표 기본 펼침
# ─────────────────────────────────────────────────────────
import time, math, datetime as dt, requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================== 기본 설정 ==================
st.set_page_config(page_title="SOL Live (Kraken)", page_icon="📈", layout="wide")

INTERVAL_MIN = {"5m":5, "15m":15, "30m":30, "1h":60, "4h":240, "1d":1440}

def _session():
    s = requests.Session()
    s.headers.update({"User-Agent":"streamlit-sol-live/3.6"})
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
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    for c in ["open","high","low","close","vwap","volume"]:
        df[c] = df[c].astype(float)
    df = df.tail(limit)
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

# -------- 지표 --------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    rs=up.rolling(n).mean()/(dn.rolling(n).mean()+1e-9)
    return 100-(100/(1+rs))
def macd(s, f=12, slow=26, sig=9):
    m=ema(s,f)-ema(s,slow); sg=ema(m,sig); return m, sg, m-sg
def atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# -------- 텔레그램(선택) --------
TG_TOKEN   = st.secrets.get("TELEGRAM_TOKEN", "")
TG_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
def send_telegram(text):
    if not (TG_TOKEN and TG_CHAT_ID): return False, "No secrets"
    try:
        url=f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        r=requests.post(url, json={"chat_id":TG_CHAT_ID,"text":text}, timeout=8)
        return (r.status_code==200), r.text
    except Exception as e:
        return False, str(e)

if "signal_log" not in st.session_state:
    st.session_state.signal_log = pd.DataFrame(columns=[
        "time","symbol","interval","signal","entry","stop","price","rsi","ma20","ma50"
    ])
if "last_signal" not in st.session_state:
    st.session_state.last_signal = None

def append_signal_log(signal, entry_px, stop_px, price, rsi_v, ma20, ma50, symbol, interval):
    st.session_state.signal_log = pd.concat([st.session_state.signal_log, pd.DataFrame([{
        "time": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "symbol":symbol, "interval":interval, "signal":signal,
        "entry":entry_px, "stop":stop_px, "price":price,
        "rsi":rsi_v, "ma20":ma20, "ma50":ma50
    }])], ignore_index=True)

# ================== 사이드바 ==================
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
    tp_factor   = st.slider("목표가(ATR 배수)", 0.5, 5.0, 2.0, 0.5)
    lev         = st.slider("레버리지(x)", 1, 20, 5)
    pos         = st.number_input("포지션 금액(USDT)", 100.0, step=10.0)

    st.markdown("---")
    st.subheader("알림 & 로그")
    enable_alerts = st.toggle("텔레그램 알림 켜기", value=False)
    notify_buy  = st.checkbox("BUY 알림", value=True)
    notify_sell = st.checkbox("SELL 알림", value=True)

    st.markdown("---")
    st.subheader("차트/예측")
    show_count = st.slider("표시 봉 수", 100, 500, 250, 50)
    y_pad_pct  = st.slider("Y축 여백(%)", 1, 10, 5)
    show_forecast = st.toggle("예측선 표시", value=True)
    fit_len      = st.slider("학습 구간(봉)", 50, 300, 180, 10)
    steps_ahead  = st.slider("예측 길이(봉)", 5, 60, 20, 5)
    band_k       = st.slider("불확실성 대역(ATR 배수)", 0.0, 2.0, 0.8, 0.1)

st.title("📈 SOL 롱·숏 라이브 — Kraken")

# ================== 데이터 로드/지표 ==================
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
atr14=float(last["ATR14"]) if not math.isnan(last["ATR14"]) else 0.0
low24=float(df["low"].tail(96).min()); high24=float(df["high"].tail(96).max())

# 목표가 가이드
long_tgts  = [round(price + atr14*tp_factor*i, 2) for i in [1,1.5,2]]
short_tgts = [round(price - atr14*tp_factor*i, 2) for i in [1,1.5,2]]

# ================== 타이밍 엔진 ==================
def score_long(row):
    s=0; s+=1 if row["MA20"]>row["MA50"] else 0
    s+=1 if row["MACD"]>row["SIGNAL"] else 0
    s+=1 if 48<=row["RSI14"]<=60 else 0
    return s
def score_short(row):
    s=0; s+=1 if row["MA20"]<row["MA50"] else 0
    s+=1 if row["MACD"]<row["SIGNAL"] else 0
    s+=1 if 40<=row["RSI14"]<=52 else 0
    return s

long_ok   = (last["MA20"]>last["MA50"]) and (last["MACD"]>last["SIGNAL"])
short_ok  = (last["MA20"]<last["MA50"]) and (last["MACD"]<last["SIGNAL"])
pullback  = abs(price-last["MA20"]) <= (0.35*atr14 if atr14>0 else 9999)

long_entry  = long_ok  and pullback
short_entry = short_ok and pullback

long_entry_px  = round(last["MA20"], 2) if long_entry else None
long_stop_px   = round(price - 1.2*atr14, 2) if long_entry and atr14>0 else None
short_entry_px = round(last["MA20"], 2) if short_entry else None
short_stop_px  = round(price + 1.2*atr14, 2) if short_entry and atr14>0 else None

# 신호 알림/로그
new_signal = "BUY" if long_entry else ("SELL" if short_entry else None)
if new_signal and new_signal != st.session_state.last_signal:
    msg = (f"[{symbol}·{interval}] {new_signal} 신호\n"
           f"가격: {price:.3f}\n"
           f"진입: {long_entry_px or short_entry_px}\n"
           f"손절: {long_stop_px or short_stop_px}\n"
           f"RSI: {last['RSI14']:.1f}  MA20:{last['MA20']:.2f}  MA50:{last['MA50']:.2f}")
    if enable_alerts and ((new_signal=="BUY" and notify_buy) or (new_signal=="SELL" and notify_sell)):
        ok, detail = send_telegram(msg)
        st.toast("텔레그램 전송 완료 ✅" if ok else f"텔레그램 실패: {detail}", icon="✅" if ok else "⚠️")
    append_signal_log(new_signal,
        (long_entry_px if new_signal=="BUY" else short_entry_px),
        (long_stop_px  if new_signal=="BUY" else short_stop_px),
        price, float(last["RSI14"]), float(last["MA20"]), float(last["MA50"]),
        symbol, interval)
    st.session_state.last_signal = new_signal

# ================== KPI ==================
k1,k2,k3,k4 = st.columns(4)
k1.metric("현재가", f"{price:,.3f} USD")
k2.metric("24h 변화", f"{((price-open_px)/open_px*100) if open_px else 0:+.2f}%")
k3.metric("24h 고가", f"{high24:,.2f}")
k4.metric("24h 저가", f"{low24:,.2f}")

# ================== 예측(선형회귀 + ATR 대역) ==================
plot_df = df.tail(int(show_count)).copy()
forecast_df = pd.DataFrame()
if show_forecast and len(plot_df) > fit_len + 5:
    x = np.arange(len(plot_df)); y = plot_df["close"].to_numpy()
    x_fit = x[-fit_len:]; y_fit = y[-fit_len:]
    a, b = np.polyfit(x_fit, y_fit, 1)   # y ≈ a*x + b

    step_min = INTERVAL_MIN[interval]
    fut_idx = np.arange(x[-1]+1, x[-1]+1+steps_ahead)
    fut_time = pd.date_range(plot_df["time"].iloc[-1] + pd.Timedelta(minutes=step_min),
                             periods=steps_ahead, freq=f"{step_min}min")
    y_pred = a*fut_idx + b
    band = float(plot_df["ATR14"].iloc[-1]) * band_k if "ATR14" in plot_df and not np.isnan(plot_df["ATR14"].iloc[-1]) else 0.0
    forecast_df = pd.DataFrame({"time":fut_time, "pred":y_pred, "upper":y_pred+band, "lower":y_pred-band})

# ================== 메인 Plotly 캔들 ==================
fig = go.Figure()
fig.add_candlestick(x=plot_df["time"], open=plot_df["open"], high=plot_df["high"],
                    low=plot_df["low"], close=plot_df["close"],
                    increasing_line_color="#0ea5e9", decreasing_line_color="#ef4444",
                    increasing_fillcolor="#0ea5e9", decreasing_fillcolor="#ef4444")
fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["MA20"], mode="lines",
                         name="MA20", line=dict(width=1.4, dash="dot")))
fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["MA50"], mode="lines",
                         name="MA50", line=dict(width=1.4, dash="dash")))
# 수평 레벨
levels = []
if long_entry:   levels += [("Entry Long", long_entry_px), ("Stop Long", long_stop_px)]
if short_entry:  levels += [("Entry Short", short_entry_px), ("Stop Short", short_stop_px)]
for t in long_tgts:  levels.append((f"TP L {t}", t))
for t in short_tgts: levels.append((f"TP S {t}", t))
for name, yv in [lv for lv in levels if lv[1] is not None]:
    fig.add_hline(y=yv, line_width=1, line_dash="dot",
                  annotation_text=name, annotation_position="top left")
# 예측선/대역
if not forecast_df.empty:
    fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["pred"],
                             mode="lines", name="Forecast",
                             line=dict(width=2, dash="dot", color="#1f77b4")))
    if band_k > 0:
        fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["lower"],
                                 mode="lines", line=dict(width=0.5, dash="dot", color="#1f77b4"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["upper"],
                                 mode="lines", line=dict(width=0.5, dash="dot", color="#1f77b4"),
                                 fill="tonexty", fillcolor="rgba(31,119,180,0.12)", showlegend=False))
# 자동 확대(Y padding)
ymin = float(plot_df["low"].min()); ymax = float(plot_df["high"].max())
ypad = (ymax - ymin) * (y_pad_pct/100); ypad = ypad if ypad>0 else max(price*0.01, 0.5)
fig.update_yaxes(range=[ymin-ypad, ymax+ypad])
fig.update_layout(height=420, template="plotly_white",
                  xaxis_rangeslider_visible=False,
                  margin=dict(l=10,r=10,t=30,b=0),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

left, right = st.columns([2,1], gap="large")
with left:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # 보조지표 (기본 펼침)
    with st.expander("RSI / MACD / HIST", expanded=True):
        fr = go.Figure(go.Scatter(x=plot_df["time"], y=plot_df["RSI14"], mode="lines", name="RSI14"))
        fr.update_layout(height=140, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), yaxis_title="RSI14")
        st.plotly_chart(fr, use_container_width=True, config={"displayModeBar": False})

        fm = go.Figure()
        fm.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["MACD"], mode="lines", name="MACD"))
        fm.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["SIGNAL"], mode="lines", name="SIGNAL"))
        fm.add_trace(go.Bar(x=plot_df["time"], y=plot_df["HIST"], name="HIST", opacity=0.4))
        fm.update_layout(height=160, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fm, use_container_width=True, config={"displayModeBar": False})

# ================== Signals & PnL & Timing (오른쪽 컬럼) ==================
with right:
    st.subheader("신호 요약")
    lsc, ssc = score_long(last), score_short(last)
    st.write(f"롱 {lsc}/3 · 숏 {ssc}/3")
    if lsc>ssc: st.success("🟢 롱 우세")
    elif ssc>lsc: st.error("🔴 숏 우세")
    else: st.warning("⚠️ 중립")

    st.subheader("손익 시뮬레이션")
    rows=[]
    if entry_long>0:
        for t in long_tgts:
            ret=((t-entry_long)/entry_long)*lev*100; pnl=pos*(ret/100); rows.append(["롱", t, ret, pnl])
        if stop_long:
            ret=((stop_long-entry_long)/entry_long)*lev*100; pnl=pos*(ret/100); rows.append(["롱", stop_long, ret, pnl])
    if entry_short>0:
        for t in short_tgts:
            ret=((entry_short-t)/entry_short)*lev*100; pnl=pos*(ret/100); rows.append(["숏", t, ret, pnl])
        if stop_short:
            ret=-abs(((stop_short-entry_short)/entry_short)*lev*100); pnl=-abs(pos*((stop_short-entry_short)/entry_short)*lev); rows.append(["숏", stop_short, ret, pnl])
    sim = pd.DataFrame(rows, columns=["방향","목표가","예상 수익률(%)","예상 PnL(USDT)"])
    if not sim.empty:
        sim["예상 수익률(%)"]=sim["예상 수익률(%)"].round(2)
        sim["예상 PnL(USDT)"]=sim["예상 PnL(USDT)"].round(2)
        st.dataframe(sim, use_container_width=True, height=260)
    else:
        st.caption("포지션 정보가 없어 시뮬레이션을 표시할 수 없습니다.")

    st.subheader("타이밍 제안")
    if long_entry:
        st.success(f"✅ BUY(롱) 타이밍 · 제안 진입 {long_entry_px} · 손절 {long_stop_px} · 목표 {', '.join(map(str,long_tgts))}")
    elif short_entry:
        st.error(f"✅ SELL(숏) 타이밍 · 제안 진입 {short_entry_px} · 손절 {short_stop_px} · 목표 {', '.join(map(str,short_tgts))}")
    else:
        st.info("⏸ 대기 — 조건 미충족(추세/RSI/되돌림)")

    st.subheader("신호 로그")
    if st.session_state.signal_log.empty:
        st.caption("아직 기록된 신호가 없습니다.")
    else:
        st.dataframe(st.session_state.signal_log, use_container_width=True, height=200)
        st.download_button("CSV 다운로드",
            st.session_state.signal_log.to_csv(index=False).encode("utf-8-sig"),
            file_name="signal_log.csv", mime="text/csv")

# ================== 자동 새로고침 ==================
if refresh and refresh>0:
    time.sleep(max(refresh, 15))
    st.rerun()
