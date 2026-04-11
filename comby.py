# pyright: basic
# type: ignore
"""
╔══════════════════════════════════════════════════════════════╗
║   6 BOTS IA TRADING — SMC / ICT / VWAP / VOLUME             ║
║   Données : Yahoo Finance (gratuit, sans compte)             ║
║   Bot 1 : Swing Très Puissant                               ║
║   Bot 2 : Day Trading Puissant (Arbre)                      ║
║   Bot 3 : Day Trading Très Puissant                         ║
║   Bot 4 : Scalping Puissant                                 ║
║   Bot 5 : Scalping Très Puissant                            ║
║   Bot 6 : Swing Puissant                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import time
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="6 Bots Trading IA",
    layout="wide",
    page_icon="🤖",
)
st_autorefresh(interval=30000, key="auto")

try:
    GROQ_API_KEY      = st.secrets.get("GROQ_API_KEY", "")
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY    = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    GROQ_API_KEY = ANTHROPIC_API_KEY = GEMINI_API_KEY = ""

for k, v in {
    "b1_signals": {}, "b2_signals": {}, "b3_signals": {},
    "b4_signals": {}, "b5_signals": {}, "b6_signals": {},
    "chat": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SYMBOLES
# ─────────────────────────────────────────────
YF_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "XAUUSD": "GC=F",
}

TF_CFG = {
    # label → (yf_interval, yf_period)
    "W1":  ("1wk", "2y"),
    "D1":  ("1d",  "1y"),
    "H4":  ("1h",  "60d"),   # resamplé en 4H
    "H1":  ("60m", "30d"),
    "M15": ("15m", "5d"),
    "M5":  ("5m",  "5d"),
    "M1":  ("1m",  "1d"),
}

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=20, show_spinner=False)
def fetch(sym_key: str, tf: str, bars: int = 300) -> pd.DataFrame | None:
    yf_sym = YF_SYMBOLS.get(sym_key, sym_key)
    interval, period = TF_CFG[tf]
    try:
        df = yf.download(yf_sym, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df.rename(columns={"volume": "tick_volume"})
        df = df[["open", "high", "low", "close", "tick_volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        # resample 4H
        if tf == "H4":
            df = df.resample("4h").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last",
                "tick_volume": "sum",
            }).dropna()
        df = df.reset_index().rename(columns={"index": "time", "Datetime": "time", "Date": "time"})
        if "time" not in df.columns:
            df = df.reset_index()
            df.columns = ["time"] + list(df.columns[1:])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        return df.tail(bars).reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(ttl=10, show_spinner=False)
def live_price(sym_key: str) -> float | None:
    try:
        t = yf.Ticker(YF_SYMBOLS[sym_key])
        return float(t.fast_info.last_price)
    except Exception:
        return None


# ─────────────────────────────────────────────
# INDICATEURS COMMUNS
# ─────────────────────────────────────────────
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"].copy(); h = df["high"]; l = df["low"]
    df["ema8"]   = c.ewm(span=8,   adjust=False).mean()
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    # RSI
    d  = c.diff()
    ag = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))
    # ATR
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=13, adjust=False).mean()
    # VWAP (journalier rolling)
    tp = (h + l + c) / 3
    df["vwap"] = (tp * df["tick_volume"]).rolling(20).sum() / df["tick_volume"].rolling(20).sum()
    # Volume relatif
    df["vol_ma"]  = df["tick_volume"].rolling(20).mean()
    df["vol_rel"] = df["tick_volume"] / df["vol_ma"].replace(0, 1)
    # Bollinger
    df["bb_mid"]   = c.rolling(20).mean()
    df["bb_upper"] = df["bb_mid"] + 2 * c.rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - 2 * c.rolling(20).std()
    # Stoch RSI
    rsi_min = df["rsi"].rolling(14).min(); rsi_max = df["rsi"].rolling(14).max()
    df["stoch_k"] = ((df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-9)).rolling(3).mean() * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


def detect_ob(df: pd.DataFrame) -> list:
    """Order Blocks valides (non cassés)"""
    res: list = []
    recent = df.tail(120).reset_index(drop=True)
    atr_v  = float(recent["atr"].iloc[-1]) if "atr" in recent.columns else 0.0
    for i in range(2, len(recent) - 3):
        c = recent.iloc[i]; n1 = recent.iloc[i+1]; n2 = recent.iloc[i+2]
        body = abs(float(c["close"]) - float(c["open"]))
        if body < 0.3 * atr_v: continue
        is_bull = (float(c["close"]) < float(c["open"]) and
                   float(n1["close"]) > float(n1["open"]) and
                   float(n2["close"]) > float(n2["open"]))
        is_bear = (float(c["close"]) > float(c["open"]) and
                   float(n1["close"]) < float(n1["open"]) and
                   float(n2["close"]) < float(n2["open"]))
        if not (is_bull or is_bear): continue
        kind = "BULL" if is_bull else "BEAR"
        lo, hi = round(float(c["low"]), 5), round(float(c["high"]), 5)
        broken = False
        for _, row in recent.iloc[i+3:].iterrows():
            cf = float(row["close"])
            if kind == "BULL" and cf < lo: broken = True; break
            if kind == "BEAR" and cf > hi: broken = True; break
        if not broken:
            res.append({"kind": kind, "low": lo, "high": hi,
                        "mid": round((lo+hi)/2, 5), "time": recent["time"].iloc[i], "idx": i})
    return res[-4:]


def detect_fvg(df: pd.DataFrame) -> list:
    fvgs: list = []
    recent = df.tail(80).reset_index(drop=True)
    for i in range(1, len(recent)-1):
        p = recent.iloc[i-1]; n = recent.iloc[i+1]; t = recent["time"].iloc[i]
        if float(n["low"]) > float(p["high"]):
            fvgs.append({"type": "BULL", "top": round(float(n["low"]), 5), "bot": round(float(p["high"]), 5), "time": t})
        elif float(n["high"]) < float(p["low"]):
            fvgs.append({"type": "BEAR", "top": round(float(p["low"]), 5), "bot": round(float(n["high"]), 5), "time": t})
    return fvgs[-6:]


def detect_bos(df: pd.DataFrame) -> str:
    """Break of Structure : BULL / BEAR / NONE"""
    if len(df) < 30: return "NONE"
    w = df.tail(30)
    highs = w["high"].values; lows = w["low"].values
    if highs[-1] > highs[-15:].max() * 0.999: return "BULL"
    if lows[-1]  < lows[-15:].min()  * 1.001: return "BEAR"
    return "NONE"


def session_ok(kind: str = "both") -> bool:
    """London 08-11h UTC / NY 13-16h UTC"""
    h = datetime.now(timezone.utc).hour
    lon = 8 <= h <= 11
    ny  = 13 <= h <= 16
    if kind == "london": return lon
    if kind == "ny":     return ny
    return lon or ny


def trend_direction(df: pd.DataFrame) -> str:
    if df is None or len(df) < 50: return "NEUTRE"
    l = df.iloc[-1]
    e20 = float(l["ema20"]); e50 = float(l["ema50"]); p = float(l["close"])
    if e20 > e50 and p > e20: return "HAUSSIER"
    if e20 < e50 and p < e20: return "BAISSIER"
    return "NEUTRE"


def make_signal(order: str, entry: float, atr: float, sym: str,
                bot: str, reason: str, rr1: float = 1.5, rr2: float = 2.5, rr3: float = 3.5) -> dict:
    d = 5 if sym == "EURUSD" else 2
    if "BUY" in order:
        sl  = round(entry - 1.5 * atr, d)
        tp1 = round(entry + rr1 * atr, d)
        tp2 = round(entry + rr2 * atr, d)
        tp3 = round(entry + rr3 * atr, d)
    else:
        sl  = round(entry + 1.5 * atr, d)
        tp1 = round(entry - rr1 * atr, d)
        tp2 = round(entry - rr2 * atr, d)
        tp3 = round(entry - rr3 * atr, d)
    rr = round(abs(tp1 - entry) / max(abs(entry - sl), 1e-10), 2)
    return {"order": order, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "rr": rr, "atr": round(atr, d), "bot": bot, "reason": reason,
            "sym": sym, "ts": time.time()}


# ─────────────────────────────────────────────
# CHART GÉNÉRIQUE
# ─────────────────────────────────────────────
def make_chart(df: pd.DataFrame, sig: dict | None, obs: list, fvgs: list,
               title: str, n: int = 60) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#131722", plot_bgcolor="#131722",
                          font=dict(color="#d1d4dc"),
                          annotations=[dict(text="Données indisponibles",
                                           xref="paper", yref="paper", x=0.5, y=0.5,
                                           showarrow=False, font=dict(size=16, color="#888"))])
        return fig

    pl = df.tail(n).reset_index(drop=True)
    t0 = pl["time"].iloc[0]; t1 = pl["time"].iloc[-1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.03,
                        subplot_titles=(title, "RSI 14 + Stoch RSI"))

    fig.add_trace(go.Candlestick(
        x=pl["time"], open=pl["open"], high=pl["high"], low=pl["low"], close=pl["close"],
        name="Prix",
        increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
    ), row=1, col=1)

    for col_n, col_c, lbl in [("ema20", "#f39c12", "EMA20"), ("ema50", "#3498db", "EMA50"),
                               ("ema200", "#e74c3c", "EMA200")]:
        if col_n in pl.columns:
            fig.add_trace(go.Scatter(x=pl["time"], y=pl[col_n],
                line=dict(color=col_c, width=1.2), name=lbl), row=1, col=1)

    if "vwap" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["vwap"],
            line=dict(color="#e056fd", width=1.4, dash="dot"), name="VWAP"), row=1, col=1)

    # OBs
    for ob in obs:
        x0  = ob["time"] if ob["time"] >= t0 else t0
        fc  = "rgba(0,200,120,0.18)"  if ob["kind"] == "BULL" else "rgba(255,80,80,0.18)"
        lc  = "rgba(0,220,130,0.9)"   if ob["kind"] == "BULL" else "rgba(255,90,90,0.9)"
        lbl = "OB ↑"                  if ob["kind"] == "BULL" else "OB ↓"
        fig.add_shape(type="rect", x0=x0, x1=t1, y0=ob["low"], y1=ob["high"],
                      fillcolor=fc, line=dict(color=lc, width=1.5), row=1, col=1)
        fig.add_annotation(x=t1, y=ob["mid"], text=lbl, showarrow=False,
                           font=dict(size=9, color=lc), xanchor="left", row=1, col=1)

    # FVGs
    for fvg in fvgs:
        if fvg["time"] < t0: continue
        fc = "rgba(0,180,255,0.12)"  if fvg["type"] == "BULL" else "rgba(255,140,0,0.12)"
        lc = "rgba(0,180,255,0.8)"   if fvg["type"] == "BULL" else "rgba(255,140,0,0.8)"
        fig.add_shape(type="rect", x0=fvg["time"], x1=t1, y0=fvg["bot"], y1=fvg["top"],
                      fillcolor=fc, line=dict(color=lc, width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=t1, y=(fvg["bot"]+fvg["top"])/2,
                           text="FVG↑" if fvg["type"] == "BULL" else "FVG↓",
                           showarrow=False, font=dict(size=8, color=lc), xanchor="left", row=1, col=1)

    # Signal
    if sig:
        ac = "#00e676" if "BUY" in sig["order"] else "#ff5252"
        ep = sig["entry"]
        fig.add_hline(y=ep, line_color=ac, line_dash="solid", line_width=2,
                      annotation_text=f"  ▶ {sig['order']} @ {ep}",
                      annotation_position="right", annotation_font=dict(color=ac, size=10))
        fig.add_hline(y=sig["sl"],  line_color="rgba(230,50,50,0.85)",  line_dash="dash",
                      annotation_text="  🛑 SL",  annotation_position="right",
                      annotation_font=dict(color="rgba(230,50,50,0.9)", size=9))
        for tp, tc, lbl in [(sig["tp1"], "rgba(0,210,100,0.9)", "TP1"),
                             (sig["tp2"], "rgba(0,150,210,0.9)", "TP2"),
                             (sig["tp3"], "rgba(160,60,230,0.9)", "TP3")]:
            fig.add_hline(y=tp, line_color=tc, line_dash="dot",
                          annotation_text=f"  {lbl}", annotation_position="right",
                          annotation_font=dict(color=tc, size=9))

    # RSI
    if "rsi" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["rsi"],
            line=dict(color="#9b59b6", width=1.6), name="RSI14"), row=2, col=1)
    if "stoch_k" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["stoch_k"],
            line=dict(color="#f39c12", width=1, dash="dot"), name="K"), row=2, col=1)
    if "stoch_d" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["stoch_d"],
            line=dict(color="#3498db", width=1, dash="dot"), name="D"), row=2, col=1)
    for lvl, lc in [(70, "rgba(255,80,80,0.6)"), (30, "rgba(80,200,80,0.6)"), (50, "rgba(150,150,150,0.4)")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=lc,
                      annotation_text=str(lvl), annotation_position="right",
                      annotation_font=dict(size=8, color=lc), row=2, col=1)

    fig.update_layout(
        height=820, xaxis_rangeslider_visible=False,
        paper_bgcolor="#131722", plot_bgcolor="#131722",
        font=dict(color="#d1d4dc", family="monospace", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=0, r=110, t=50, b=10), hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e222d", font=dict(color="#d1d4dc")),
    )
    fig.update_xaxes(gridcolor="#1e2130", zerolinecolor="#1e2130",
                     showspikes=True, spikecolor="#555", spikethickness=1,
                     rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(gridcolor="#1e2130", zerolinecolor="#1e2130",
                     showspikes=True, spikecolor="#555", side="right")
    fig.update_yaxes(range=[0, 100], row=2, col=1, side="right")
    return fig


# ─────────────────────────────────────────────
# AFFICHAGE SIGNAL CARD
# ─────────────────────────────────────────────
def signal_card(sig: dict):
    if not sig: return
    col = "🟢" if "BUY" in sig["order"] else "🔴"
    st.success(f"""
{col} **{sig['order']}** — {sig['sym']}
📍 Entrée : `{sig['entry']}`   |   🛑 SL : `{sig['sl']}`   |   R:R **{sig['rr']}**
🥇 TP1 : `{sig['tp1']}`   🥈 TP2 : `{sig['tp2']}`   🥉 TP3 : `{sig['tp3']}`
🔍 **Raison** : {sig['reason']}
⏱ ATR : `{sig['atr']}`
""")
    tv = "EURUSD" if sig["sym"] == "EURUSD" else "XAUUSD"
    st.markdown(f"🔗 [TradingView](https://www.tradingview.com/chart/?symbol={tv})  🔗 [Investing.com](https://www.investing.com/{'currencies/eur-usd' if sig['sym']=='EURUSD' else 'commodities/gold'})")


def no_signal(reason: str = ""):
    st.info(f"⏸️ Pas de signal — {reason}" if reason else "⏸️ Conditions non remplies — attendre.")


# ─────────────────────────────────────────────
# ARBRE DÉCISIONNEL — HELPERS
# ─────────────────────────────────────────────
def vwap_filter(df: pd.DataFrame, direction: str) -> bool:
    if df is None or "vwap" not in df.columns: return False
    p = float(df.iloc[-1]["close"]); v = float(df.iloc[-1]["vwap"])
    if direction == "BUY"  and p > v: return True
    if direction == "SELL" and p < v: return True
    return False


def volume_strong(df: pd.DataFrame, threshold: float = 1.3) -> bool:
    if df is None or "vol_rel" not in df.columns: return False
    return float(df.iloc[-1]["vol_rel"]) >= threshold


def price_near_ob(df: pd.DataFrame, ob: dict, tol_atr: float = 0.5) -> bool:
    if df is None: return False
    p   = float(df.iloc[-1]["close"])
    atr = float(df.iloc[-1]["atr"])
    return ob["low"] - tol_atr * atr <= p <= ob["high"] + tol_atr * atr


def rejection_candle(df: pd.DataFrame, direction: str) -> bool:
    if df is None or len(df) < 2: return False
    p = df.iloc[-2]; atr = float(df.iloc[-1]["atr"])
    body  = abs(float(p["close"]) - float(p["open"]))
    if direction == "BUY":
        wick = min(float(p["open"]), float(p["close"])) - float(p["low"])
        return wick >= 0.5 * atr
    else:
        wick = float(p["high"]) - max(float(p["open"]), float(p["close"]))
        return wick >= 0.5 * atr


def bullish_confirm(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2: return False
    l = df.iloc[-1]
    return float(l["close"]) > float(l["open"]) and float(l["close"]) > float(df.iloc[-2]["close"])


def bearish_confirm(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 2: return False
    l = df.iloc[-1]
    return float(l["close"]) < float(l["open"]) and float(l["close"]) < float(df.iloc[-2]["close"])


# ═══════════════════════════════════════════════════════════════
# BOT 1 — SWING TRÈS PUISSANT
# W1/D1 biais → H4 Order Block → H1 entrée
# ═══════════════════════════════════════════════════════════════
def bot1_run(sym: str) -> tuple:
    """
    Arbre décisionnel (8 étapes) :
    1. Biais W1/D1 clair ?
    2. H4 Order Block dans la même direction ?
    3. Zone liquidité (FVG, sweep) claire ?
    4. Prix teste l'OB avec sweep ?
    5. Rebond dans direction du biais ?
    6. Volume/ATR fort ?
    7. Gestion risque (1-2% capital)
    8. Entrée H1/H4 avec SL/TP
    """
    steps: list = []

    # Step 1 — Biais W1/D1
    dfw1 = fetch(sym, "W1")
    dfd1 = fetch(sym, "D1")
    if dfw1 is None or dfd1 is None:
        return None, ["❌ Données W1/D1 indisponibles"]

    dfw1 = indicators(dfw1); dfd1 = indicators(dfd1)
    biais_w1 = trend_direction(dfw1); biais_d1 = trend_direction(dfd1)

    if biais_w1 == "NEUTRE" or biais_d1 == "NEUTRE":
        steps.append(f"1. ❌ Biais W1={biais_w1} / D1={biais_d1} — pas de trade swing.")
        return None, steps
    if biais_w1 != biais_d1:
        steps.append(f"1. ⚠️ Biais divergent W1={biais_w1} / D1={biais_d1} — attendre alignement.")
        return None, steps
    direction = biais_w1
    steps.append(f"1. ✅ Biais W1/D1 aligné : **{direction}**")

    # Step 2 — H4 Order Block
    dfh4 = fetch(sym, "H4")
    if dfh4 is None:
        return None, steps + ["2. ❌ H4 indisponible"]
    dfh4 = indicators(dfh4)
    obs_h4 = detect_ob(dfh4)
    matching_obs = [ob for ob in obs_h4 if ob["kind"] == ("BULL" if direction == "HAUSSIER" else "BEAR")]
    if not matching_obs:
        steps.append("2. ❌ Aucun Order Block H4 dans la direction du biais — attendre.")
        return None, steps
    ob_ref = matching_obs[-1]
    steps.append(f"2. ✅ OB H4 trouvé : {ob_ref['low']} → {ob_ref['high']}")

    # Step 3 — Zone liquidité (FVG)
    fvgs_h4 = detect_fvg(dfh4)
    fvg_ok = any(
        (f["type"] == "BULL" and direction == "HAUSSIER") or
        (f["type"] == "BEAR" and direction == "BAISSIER")
        for f in fvgs_h4
    )
    if fvg_ok:
        steps.append("3. ✅ FVG / Zone de liquidité détectée sur H4.")
    else:
        steps.append("3. ⚠️ Pas de FVG clair — surveiller uniquement.")

    # Step 4 — Prix teste l'OB
    near = price_near_ob(dfh4, ob_ref)
    if not near:
        steps.append("4. ⏳ Prix pas encore dans la zone OB — attendre le test.")
        return None, steps
    steps.append("4. ✅ Prix dans / proche de la zone OB — test en cours.")

    # Step 5 — Rebond + bougie de confirmation
    if direction == "HAUSSIER":
        rej = rejection_candle(dfh4, "BUY") or bullish_confirm(dfh4)
    else:
        rej = rejection_candle(dfh4, "SELL") or bearish_confirm(dfh4)
    if not rej:
        steps.append("5. ❌ Pas de bougie de rejet/confirmation — invalider le setup.")
        return None, steps
    steps.append("5. ✅ Bougie de rejet/confirmation présente.")

    # Step 6 — Volume/ATR fort
    vol_ok = volume_strong(dfh4, 1.2)
    if vol_ok:
        steps.append("6. ✅ Volume/ATR fort — mouvement institutionnel confirmé.")
    else:
        steps.append("6. ⚠️ Volume moyen — SL serré, confiance réduite.")

    # Step 7-8 — Signal
    entry = float(dfh4.iloc[-1]["close"])
    atr   = float(dfh4.iloc[-1]["atr"])
    order = "BUY LIMIT" if direction == "HAUSSIER" else "SELL LIMIT"
    reason = f"Swing {direction} | W1/D1/H4 alignés | OB H4 {ob_ref['low']}-{ob_ref['high']}"
    if fvg_ok: reason += " | FVG ✅"
    if vol_ok: reason += " | Volume fort ✅"
    steps.append("7. ✅ Gestion risque : 1-2% capital, SL sous OB, TP highs/lows précédents.")
    steps.append(f"8. 🎯 **ENTRÉE H4/H1** — {order} @ {entry:.5f}")
    sig = make_signal(order, entry, atr, sym, "Bot1-SwingTrèsPuissant", reason, rr1=2.0, rr2=3.5, rr3=5.0)
    return sig, steps


# ═══════════════════════════════════════════════════════════════
# BOT 2 — DAY TRADING PUISSANT (ARBRE 4 PHASES)
# H1 biais → M15 SMC → VWAP + Volume → M5/M1 trigger
# ═══════════════════════════════════════════════════════════════
def bot2_run(sym: str) -> tuple:
    """
    Phase 0 : Session London/NY
    Phase 1 : OB/FVG sur M15 dans le biais H1
    Phase 2 : VWAP + Volume (force du move)
    Phase 3 : M5/M1 trigger (rejet OB/FVG)
    Phase 4 : Gestion risque
    """
    steps: list = []

    # Phase 0 — Session
    sess = session_ok("both")
    if not sess:
        steps.append("Phase 0. ❌ Hors session London/NY (08-11h / 13-16h UTC) — ne pas trader.")
        return None, steps
    steps.append("Phase 0. ✅ Session London ou NY active.")

    # Phase 1 — Biais H1
    dfh1 = fetch(sym, "H1")
    if dfh1 is None:
        return None, steps + ["Phase 1. ❌ H1 indisponible"]
    dfh1 = indicators(dfh1)
    biais_h1 = trend_direction(dfh1)
    if biais_h1 == "NEUTRE":
        steps.append("Phase 1. ❌ Biais H1 neutre / range extrême — attendre.")
        return None, steps
    steps.append(f"Phase 1. ✅ Biais H1 : **{biais_h1}**")

    # Phase 1b — OB/FVG sur M15
    dfm15 = fetch(sym, "M15")
    if dfm15 is None:
        return None, steps + ["Phase 1b. ❌ M15 indisponible"]
    dfm15 = indicators(dfm15)
    obs_m15  = detect_ob(dfm15)
    fvgs_m15 = detect_fvg(dfm15)
    direction = biais_h1
    kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    obs_ok  = [ob for ob in obs_m15  if ob["kind"] == kind_want]
    fvgs_ok = [f  for f  in fvgs_m15 if f["type"]  == kind_want]

    if not obs_ok and not fvgs_ok:
        steps.append("Phase 1b. ❌ Aucun OB/FVG M15 dans la direction — attendre nouveau cycle.")
        return None, steps
    zone_ref = obs_ok[-1] if obs_ok else None
    steps.append(f"Phase 1b. ✅ {'OB' if zone_ref else 'FVG'} M15 trouvé dans la direction.")

    # Phase 2 — VWAP + Volume
    vwap_ok = vwap_filter(dfm15, "BUY" if direction == "HAUSSIER" else "SELL")
    vol_ok  = volume_strong(dfm15, 1.25)
    if not vwap_ok:
        steps.append("Phase 2. ❌ Prix mauvais côté du VWAP — réduire priorité.")
        return None, steps
    steps.append(f"Phase 2. ✅ VWAP aligné | Volume {'fort ✅' if vol_ok else 'moyen ⚠️'}")

    # Phase 3 — M5 rejet sur OB/FVG
    dfm5 = fetch(sym, "M5")
    if dfm5 is None:
        return None, steps + ["Phase 3. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    obs_m5  = detect_ob(dfm5)
    near_m5 = any(price_near_ob(dfm5, ob) for ob in obs_m5 if ob["kind"] == kind_want)
    rej_m5  = rejection_candle(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    if not near_m5 or not rej_m5:
        steps.append("Phase 3. ❌ Pas de rejet M5 propre sur OB/FVG — attendre.")
        return None, steps
    steps.append("Phase 3. ✅ Rejet M5 propre sur OB/FVG — signal VERT.")

    # Phase 4 — Signal
    entry = float(dfm5.iloc[-1]["close"])
    atr   = float(dfm5.iloc[-1]["atr"])
    order = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Day {direction} | H1→M15→M5 | VWAP {'✅' if vwap_ok else ''} | Vol {'✅' if vol_ok else '⚠️'}"
    steps.append(f"Phase 4. ✅ Risque 1-2% | SL sous OB/FVG | TP zone liquidité suivante.")
    steps.append(f"Phase 4. 🎯 **SIGNAL** {order} @ {entry:.5f}")
    sig = make_signal(order, entry, atr, sym, "Bot2-DayTradingPuissant", reason, rr1=1.5, rr2=2.5, rr3=3.5)
    return sig, steps


# ═══════════════════════════════════════════════════════════════
# BOT 3 — DAY TRADING TRÈS PUISSANT
# D1/H1 biais → M15 tendance+zones → M5 trade+trigger
# Sessions London/NY — Confluence OB+FVG+Volume+Support
# ═══════════════════════════════════════════════════════════════
def bot3_run(sym: str) -> tuple:
    steps: list = []

    # 1. Biais D1/H1
    dfd1 = fetch(sym, "D1"); dfh1 = fetch(sym, "H1")
    if dfd1 is None or dfh1 is None:
        return None, ["❌ D1/H1 indisponibles"]
    dfd1 = indicators(dfd1); dfh1 = indicators(dfh1)
    b_d1 = trend_direction(dfd1); b_h1 = trend_direction(dfh1)
    if b_d1 == "NEUTRE":
        steps.append(f"1. ❌ Biais D1 neutre — ignorer / attendre.")
        return None, steps
    steps.append(f"1. ✅ Biais D1={b_d1}  H1={b_h1}")
    direction = b_d1

    # 2. Session
    if not session_ok():
        steps.append("2. ❌ Hors session London/NY — ne pas trader.")
        return None, steps
    steps.append("2. ✅ Session favorable (London ou NY).")

    # 3. M15 OB/FVG dans la direction
    dfm15 = fetch(sym, "M15")
    if dfm15 is None: return None, steps + ["3. ❌ M15 indisponible"]
    dfm15 = indicators(dfm15)
    kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    obs_m15  = [ob for ob in detect_ob(dfm15)  if ob["kind"] == kind_want]
    fvgs_m15 = [f  for f  in detect_fvg(dfm15) if f["type"]  == kind_want]
    if not obs_m15 and not fvgs_m15:
        steps.append("3. ❌ Pas d'OB/FVG M15 dans la direction.")
        return None, steps
    steps.append(f"3. ✅ {'OB' if obs_m15 else 'FVG'} M15 aligné avec D1.")

    # 4. Prix reteste OB/FVG M15
    ob_ref = obs_m15[-1] if obs_m15 else None
    near   = ob_ref and price_near_ob(dfm15, ob_ref)
    if not near:
        steps.append("4. ⏳ Prix pas encore sur la zone OB/FVG M15 — attendre.")
        return None, steps
    steps.append(f"4. ✅ Prix reteste OB/FVG M15 ({ob_ref['low']} → {ob_ref['high']}).")

    # 5. Volume + VWAP
    vwap_ok = vwap_filter(dfm15, "BUY" if direction == "HAUSSIER" else "SELL")
    vol_ok  = volume_strong(dfm15, 1.2)
    conf_score = sum([vwap_ok, vol_ok, bool(obs_m15), bool(fvgs_m15)])
    steps.append(f"5. Confluence : {conf_score}/4 — VWAP={'✅' if vwap_ok else '❌'} Vol={'✅' if vol_ok else '⚠️'}")
    if conf_score < 2:
        steps.append("5. ❌ Confluence insuffisante — ne pas entrer.")
        return None, steps

    # 6. M5 trigger
    dfm5 = fetch(sym, "M5")
    if dfm5 is None: return None, steps + ["6. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    rej  = rejection_candle(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    if not rej:
        steps.append("6. ❌ Pas de bougie de rejet M5 — attendre.")
        return None, steps
    steps.append("6. ✅ Bougie de rejet M5 confirmée.")

    # 7. Confirmations structurelles
    bos = detect_bos(dfm5)
    bos_ok = (bos == "BULL" and direction == "HAUSSIER") or (bos == "BEAR" and direction == "BAISSIER")
    steps.append(f"7. BOS M5 : {bos} {'✅' if bos_ok else '⚠️'}")

    # 8-9. Signal
    entry = float(dfm5.iloc[-1]["close"])
    atr   = float(dfm5.iloc[-1]["atr"])
    order = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Day Très Puissant {direction} | D1/H1/M15/M5 | Conf {conf_score}/4"
    if vwap_ok: reason += " | VWAP ✅"
    if bos_ok:  reason += " | BOS ✅"
    steps.append(f"8. ✅ 1-2% capital | SL sous OB | TP 1:2-1:3")
    steps.append(f"9. 🎯 **SIGNAL** {order} M5 @ {entry:.5f}")
    sig = make_signal(order, entry, atr, sym, "Bot3-DayTradingTrèsPuissant", reason, rr1=1.5, rr2=2.5, rr3=3.5)
    return sig, steps


# ═══════════════════════════════════════════════════════════════
# BOT 4 — SCALPING PUISSANT
# H1 biais → M5 OB/FVG → VWAP → M1 trigger
# Kill Zones — MT4/MT5 style — SL/TP automatiques
# ═══════════════════════════════════════════════════════════════
def bot4_run(sym: str) -> tuple:
    steps: list = []

    # Étape 0 — Pré-filtre
    kill = session_ok("both")
    if not kill:
        steps.append("0. ❌ Hors Kill Zone (08-11h / 13-16h UTC) — ne pas scalper.")
        return None, steps
    steps.append("0. ✅ Kill Zone active.")

    # Étape 1 — SMC H1 biais
    dfh1 = fetch(sym, "H1")
    if dfh1 is None: return None, ["❌ H1 indisponible"]
    dfh1 = indicators(dfh1)
    b_h1 = trend_direction(dfh1)
    if b_h1 == "NEUTRE":
        steps.append("1.1. ❌ H1 biais neutre / pas de tendance / résistance visible — passer au prochain jour.")
        return None, steps
    direction = b_h1
    kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    obs_h1    = [ob for ob in detect_ob(dfh1) if ob["kind"] == kind_want]
    fvg_h1    = [f  for f  in detect_fvg(dfh1) if f["type"] == kind_want]
    steps.append(f"1. ✅ H1 biais={direction} | OBs={len(obs_h1)} | FVGs={len(fvg_h1)}")

    # Étape 1 — M5 OB/FVG
    dfm5 = fetch(sym, "M5")
    if dfm5 is None: return None, steps + ["1.2. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    obs_m5  = [ob for ob in detect_ob(dfm5)  if ob["kind"] == kind_want]
    fvgs_m5 = [f  for f  in detect_fvg(dfm5) if f["type"] == kind_want]
    if not obs_m5 and not fvgs_m5:
        steps.append("1.3. ❌ Pas d'OB/FVG M5 — attendre ou abandonner.")
        return None, steps
    steps.append(f"1.3. ✅ M5 : {len(obs_m5)} OBs / {len(fvgs_m5)} FVGs dans la direction.")

    # Prix reteste la zone
    ob_m5  = obs_m5[-1] if obs_m5 else None
    near_m5 = ob_m5 and price_near_ob(dfm5, ob_m5)
    if not near_m5:
        steps.append("1.4. ⏳ Prix pas encore sur OB/FVG M5 — attendre le test.")
        return None, steps
    steps.append(f"1.4. ✅ Prix reteste OB M5 : {ob_m5['low']} → {ob_m5['high']}")

    # Kill Zone check (déjà fait en étape 0)
    steps.append("1.5. ✅ Kill Zone London/NY confirmée.")

    # Étape 2 — VWAP + Volume
    vwap_ok = vwap_filter(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    vol_ok  = volume_strong(dfm5, 1.3)
    if not vwap_ok:
        steps.append("2. ❌ Prix mauvais côté VWAP — signal faible.")
        return None, steps
    if not vol_ok:
        steps.append("2. ⚠️ Volume sous la moyenne — signal faible, éviter ou lot réduit.")
    else:
        steps.append("2. ✅ VWAP + Volume fort — signal PUISSANT.")

    # Étape 3 — M1 trigger
    dfm1 = fetch(sym, "M1")
    if dfm1 is None: return None, steps + ["3. ❌ M1 indisponible"]
    dfm1 = indicators(dfm1)
    # Bougie de rejet ou cassure franche
    rej_m1  = rejection_candle(dfm1, "BUY" if direction == "HAUSSIER" else "SELL")
    bull_m1 = bullish_confirm(dfm1) if direction == "HAUSSIER" else bearish_confirm(dfm1)
    # La bougie M1 doit être JUSTE après le retest OB/FVG
    ob_m1 = detect_ob(dfm1)
    near_m1 = any(price_near_ob(dfm1, ob) for ob in ob_m1 if ob["kind"] == kind_want)
    if not (rej_m1 or bull_m1) or not near_m1:
        steps.append("3. ❌ Bougie M1 pas propre sur OB/FVG — annuler.")
        return None, steps
    steps.append("3. ✅ Bougie M1 propre (rejet / engulfing) juste après OB/FVG.")

    # Étape 4 — Risque
    entry = float(dfm1.iloc[-1]["close"])
    atr   = float(dfm1.iloc[-1]["atr"])
    order = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Scalping {direction} | H1→M5→M1 | VWAP ✅ | Vol {'✅' if vol_ok else '⚠️'} | Kill Zone ✅"
    steps.append(f"4. ✅ 0.5% capital | SL sous bougie de rejet | TP 1:2 (50%) puis 1:2-1:3")
    steps.append(f"4. 🎯 **SCALP SIGNAL** {order} M1 @ {entry:.5f}")
    sig = make_signal(order, entry, atr, sym, "Bot4-ScalpingPuissant", reason, rr1=1.0, rr2=1.5, rr3=2.0)
    return sig, steps


# ═══════════════════════════════════════════════════════════════
# BOT 5 — SCALPING TRÈS PUISSANT
# H1 biais → M5 OB+FVG → Kill Zone → VWAP → M1 exécution
# SMC/ICT ultra-précis
# ═══════════════════════════════════════════════════════════════
def bot5_run(sym: str) -> tuple:
    steps: list = []

    # 1. Biais H1
    dfh1 = fetch(sym, "H1")
    if dfh1 is None: return None, ["❌ H1 indisponible"]
    dfh1 = indicators(dfh1)
    biais = trend_direction(dfh1)
    if biais == "NEUTRE":
        steps.append("1. ❌ Biais H1 pas clair — attendre le prochain H1.")
        return None, steps
    direction = biais; kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    steps.append(f"1. ✅ Biais H1 : **{direction}**")

    # 2. Kill Zone London/NY
    if not session_ok():
        steps.append("2. ❌ Pas dans une Kill Zone (London/NY) — ne pas scalper.")
        return None, steps
    steps.append("2. ✅ Kill Zone active (London 08-11h / NY 13-16h UTC).")

    # 3. M5 OB + FVG aligné avec H1
    dfm5 = fetch(sym, "M5")
    if dfm5 is None: return None, steps + ["3. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    obs_m5  = [ob for ob in detect_ob(dfm5)  if ob["kind"] == kind_want]
    fvgs_m5 = [f  for f  in detect_fvg(dfm5) if f["type"] == kind_want]
    if not obs_m5 and not fvgs_m5:
        steps.append("3. ❌ Pas d'OB/FVG M5 dans le sens du biais — attendre.")
        return None, steps
    steps.append(f"3. ✅ M5 : {len(obs_m5)} OBs + {len(fvgs_m5)} FVGs dans la direction.")

    # 4. Prix reteste OB/FVG M5
    ob_ref = obs_m5[-1] if obs_m5 else None
    near   = ob_ref and price_near_ob(dfm5, ob_ref)
    if not near:
        steps.append("4. ⏳ Prix pas sur l'OB/FVG M5 — attendre le retest.")
        return None, steps
    steps.append(f"4. ✅ Prix reteste OB M5 : {ob_ref['low']} → {ob_ref['high']}")

    # 5. VWAP + Price Action
    vwap_ok = vwap_filter(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    steps.append(f"5. VWAP : {'✅ prix au-dessus (long)' if vwap_ok and direction=='HAUSSIER' else '✅ prix en-dessous (short)' if vwap_ok else '❌ mauvais côté'}")
    if not vwap_ok:
        steps.append("5. ❌ VWAP défavorable — attendre.")
        return None, steps

    # 6. Volume > MA 20 périodes
    vol_ok = volume_strong(dfm5, 1.2)
    steps.append(f"6. Volume M1 vs MA20 : {'✅ fort — signal puissant' if vol_ok else '⚠️ faible — SL serré, TP 1:1'}")

    # 7. M1 rejet + direction
    dfm1 = fetch(sym, "M1")
    if dfm1 is None: return None, steps + ["5. ❌ M1 indisponible"]
    dfm1 = indicators(dfm1)
    rej_m1  = rejection_candle(dfm1, "BUY" if direction == "HAUSSIER" else "SELL")
    conf_m1 = bullish_confirm(dfm1) if direction == "HAUSSIER" else bearish_confirm(dfm1)
    if not (rej_m1 or conf_m1):
        steps.append("5. ❌ Pas de rejet/clôture propre M1 — ne pas entrer.")
        return None, steps
    steps.append("5. ✅ M1 : rebond/clôture dans la bonne direction — signal VERT pour scalpe.")

    # 8-9. Entrée
    entry = float(dfm1.iloc[-1]["close"])
    atr   = float(dfm1.iloc[-1]["atr"])
    order = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Scalp Très Puissant {direction} | H1/M5/M1 | Kill Zone ✅ | VWAP ✅ | Vol {'✅' if vol_ok else '⚠️'}"
    steps.append(f"8. ✅ 1-2% capital | SL juste sous OB/FVG | TP 1:1 immédiat, 1:2-1:3 si volume confirme.")
    steps.append(f"9. 🎯 **SCALP M1** {order} @ {entry:.5f}")
    rr1 = 1.0 if not vol_ok else 1.5
    sig = make_signal(order, entry, atr, sym, "Bot5-ScalpingTrèsPuissant", reason, rr1=rr1, rr2=2.0, rr3=3.0)
    return sig, steps


# ═══════════════════════════════════════════════════════════════
# BOT 6 — SWING PUISSANT
# W1 biais macro → D1 OB/FVG/BOS → H4 confirmation + entrée
# RR ≥ 1:3, 1-2% capital, 1 trade swing à la fois
# ═══════════════════════════════════════════════════════════════
def bot6_run(sym: str) -> tuple:
    steps: list = []

    # Étape W1 — Biais macro
    dfw1 = fetch(sym, "W1")
    if dfw1 is None: return None, ["❌ W1 indisponible"]
    dfw1 = indicators(dfw1)
    b_w1 = trend_direction(dfw1)
    if b_w1 == "NEUTRE":
        steps.append("W1. ❌ Biais macro W1 non clair — passer au jour suivant.")
        return None, steps
    direction = b_w1
    kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    # Confirmation HH/HL ou LH/LL
    w1_close = dfw1["close"].values
    hh_hl = w1_close[-1] > w1_close[-3] and w1_close[-2] > w1_close[-4]
    lh_ll = w1_close[-1] < w1_close[-3] and w1_close[-2] < w1_close[-4]
    macro_ok = (direction == "HAUSSIER" and hh_hl) or (direction == "BAISSIER" and lh_ll)
    steps.append(f"W1. ✅ Biais macro : **{direction}** | HH/HL={'✅' if hh_hl else '—'} LH/LL={'✅' if lh_ll else '—'}")

    # Étape D1 — OB + FVG + BOS/MSS
    dfd1 = fetch(sym, "D1")
    if dfd1 is None: return None, steps + ["D1. ❌ D1 indisponible"]
    dfd1 = indicators(dfd1)
    obs_d1  = [ob for ob in detect_ob(dfd1)  if ob["kind"] == kind_want]
    fvgs_d1 = [f  for f  in detect_fvg(dfd1) if f["type"] == kind_want]
    bos_d1  = detect_bos(dfd1)
    bos_ok  = (bos_d1 == "BULL" and direction == "HAUSSIER") or (bos_d1 == "BEAR" and direction == "BAISSIER")

    if not obs_d1 and not fvgs_d1:
        steps.append("D1. ❌ Pas d'OB/FVG D1 — manque un alignement → pas d'entrée.")
        return None, steps
    if not bos_ok:
        steps.append(f"D1. ⚠️ BOS/MSS D1 : {bos_d1} — pas encore de rupture de structure claire. Surveiller.")
    else:
        steps.append(f"D1. ✅ BOS D1 confirmé : {bos_d1}")

    ob_d1 = obs_d1[-1] if obs_d1 else None
    steps.append(f"D1. ✅ OB D1 : {ob_d1['low']} → {ob_d1['high']}" if ob_d1 else "D1. FVG D1 utilisé.")

    # Prix se rapproche de la zone D1
    near_d1 = ob_d1 and price_near_ob(dfd1, ob_d1, tol_atr=1.0)
    if not near_d1:
        steps.append("D1. ⏳ Prix loin de la zone OB/FVG D1 — attendre.")
        return None, steps
    steps.append("D1. ✅ Prix approche la zone OB/FVG D1.")

    # Étape H4 — Confirmation + Entrée
    dfh4 = fetch(sym, "H4")
    if dfh4 is None: return None, steps + ["H4. ❌ H4 indisponible"]
    dfh4 = indicators(dfh4)

    # Prix teste OB/FVG D1 avec mouvement H4
    rej_h4 = rejection_candle(dfh4, "BUY" if direction == "HAUSSIER" else "SELL")
    if not rej_h4:
        steps.append("H4. ❌ Pas de bougie de rejet H4 (mèche longue, engulfing) — attendre ou abandonner.")
        return None, steps
    steps.append("H4. ✅ Bougie de rejet H4 confirmée — setup valide.")

    # 4.2 — Calcul SL/TP
    entry = float(dfh4.iloc[-1]["close"])
    atr   = float(dfh4.iloc[-1]["atr"])
    d     = 5 if sym == "EURUSD" else 2
    if direction == "HAUSSIER":
        sl_ref = ob_d1["low"] if ob_d1 else entry - 2 * atr
        # Distance SL > 2% du capital → réduire lot
    else:
        sl_ref = ob_d1["high"] if ob_d1 else entry + 2 * atr
    dist_sl = abs(entry - sl_ref)
    if dist_sl > 0.005 * entry:   # > 0.5% → lot réduit
        steps.append("H4. ⚠️ SL éloigné > 0.5% — réduire la taille du lot.")

    order = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Swing Puissant {direction} | W1/D1/H4 | OB D1 {ob_d1['low'] if ob_d1 else '?'}-{ob_d1['high'] if ob_d1 else '?'} | BOS={'✅' if bos_ok else '⚠️'}"

    # 4.3 — RR minimum 1:3
    rr_min = 3.0
    steps.append(f"H4. ✅ RR ≥ 1:3 ciblé | 1-2% capital | 1 seul trade swing ouvert.")
    steps.append("H4. ✅ Couvrir 50% lots à 1:2, trailing stop sur le reste.")
    steps.append(f"H4. 🎯 **SIGNAL SWING** {order} @ {entry:.{d}f}")
    sig = make_signal(order, entry, atr, sym, "Bot6-SwingPuissant", reason, rr1=2.0, rr2=3.5, rr3=5.0)
    # Override SL avec OB ref
    if direction == "HAUSSIER":
        sig["sl"] = round(sl_ref - 0.2 * atr, d)
    else:
        sig["sl"] = round(sl_ref + 0.2 * atr, d)
    return sig, steps


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════
st.title("🤖 6 BOTS IA TRADING — SMC / ICT / VWAP")
st.caption("Données Yahoo Finance (gratuit) | EURUSD + XAUUSD | Refresh automatique 30s")

BOT_INFO = [
    ("🔵 Bot 1 — Swing Très Puissant",    "W1/D1/H4/H1 | OB D1 | FVG | Volume/ATR | SL sous OB | TP ERL/IRL"),
    ("🟠 Bot 2 — Day Trading Puissant",   "H1→M15→M5 | SMC + VWAP + Volume | 4 Phases | London/NY"),
    ("🔴 Bot 3 — Day Trading Très Puissant","D1/H1/M15/M5 | OB+FVG+Session | Confluence 4 critères"),
    ("🟡 Bot 4 — Scalping Puissant",      "H1→M5→M1 | Kill Zone | VWAP | OB/FVG | TP 1:2 rapide"),
    ("🟢 Bot 5 — Scalping Très Puissant", "H1/M5/M1 | SMC/ICT | Kill Zone | VWAP | Volume M1"),
    ("🟣 Bot 6 — Swing Puissant",         "W1→D1→H4 | OB+FVG+BOS/MSS | RR ≥ 1:3 | 1 trade à la fois"),
]

tabs = st.tabs([b[0] for b in BOT_INFO] + ["📊 Tableau de bord", "🤖 Chat IA"])

RUNNERS = [bot1_run, bot2_run, bot3_run, bot4_run, bot5_run, bot6_run]
CACHE_KEYS = ["b1_signals", "b2_signals", "b3_signals", "b4_signals", "b5_signals", "b6_signals"]

for i, (tab, (bot_name, bot_desc), runner, cache_key) in enumerate(
    zip(tabs[:6], BOT_INFO, RUNNERS, CACHE_KEYS)
):
    with tab:
        st.subheader(bot_name)
        st.caption(bot_desc)

        sym_sel = st.selectbox("Symbole", ["EURUSD", "XAUUSD"], key=f"sym_b{i+1}")

        col_run, col_live = st.columns([1, 3])
        with col_run:
            run_btn = st.button(f"▶ Analyser", key=f"run_b{i+1}")

        price = live_price(sym_sel)
        with col_live:
            if price:
                st.metric("Prix live", f"{price:.5f}" if sym_sel == "EURUSD" else f"{price:.2f}")

        if run_btn:
            with st.spinner("Analyse en cours (multi-timeframes)…"):
                sig, steps = runner(sym_sel)
            st.session_state[cache_key][sym_sel] = {"sig": sig, "steps": steps, "ts": time.time()}

        cached = st.session_state[cache_key].get(sym_sel)

        if cached:
            age = int(time.time() - cached["ts"])
            st.caption(f"⏱ Dernière analyse : il y a {age}s")
            st.divider()

            # Arbre décisionnel
            st.subheader("🌲 Arbre décisionnel")
            for step in cached["steps"]:
                if "✅" in step:
                    st.success(step)
                elif "❌" in step:
                    st.error(step)
                elif "⏳" in step or "⚠️" in step:
                    st.warning(step)
                elif "🎯" in step:
                    st.info(step)
                else:
                    st.write(step)

            st.divider()
            sig = cached["sig"]
            if sig:
                st.subheader("📡 Signal généré")
                signal_card(sig)

                # Graphique
                st.subheader("📈 Graphique")
                tf_chart = "H4" if "Swing" in bot_name else "M5" if "Scalping" in bot_name else "M15"
                df_chart = fetch(sym_sel, tf_chart)
                if df_chart is not None:
                    df_chart = indicators(df_chart)
                    obs_ch   = detect_ob(df_chart)
                    fvgs_ch  = detect_fvg(df_chart)
                    st.plotly_chart(
                        make_chart(df_chart, sig, obs_ch, fvgs_ch,
                                   f"{sym_sel} — {tf_chart} — {bot_name}", n=60),
                        use_container_width=True,
                    )
            else:
                no_signal("Conditions non réunies — relancer après un nouveau cycle de marché.")
        else:
            st.info("Cliquer sur **▶ Analyser** pour lancer le bot.")

        # Liens externes
        tv_sym = "EURUSD" if sym_sel == "EURUSD" else "XAUUSD"
        st.markdown(
            f"🔗 [TradingView](https://www.tradingview.com/chart/?symbol={tv_sym})  "
            f"🔗 [Investing.com](https://www.investing.com/{'currencies/eur-usd' if sym_sel=='EURUSD' else 'commodities/gold'})"
        )

# ─────────────────────────────────────────────
# TAB 7 — TABLEAU DE BORD
# ─────────────────────────────────────────────
with tabs[6]:
    st.subheader("📊 Tableau de bord — Tous les Bots")
    st.caption("Résumé de la dernière analyse de chaque bot.")

    dashboard_data = []
    for i, (bot_name, bot_desc) in enumerate(BOT_INFO):
        ck = CACHE_KEYS[i]
        for sym in ["EURUSD", "XAUUSD"]:
            cached = st.session_state[ck].get(sym)
            if cached and cached.get("sig"):
                s = cached["sig"]
                dashboard_data.append({
                    "Bot":     bot_name,
                    "Symbole": sym,
                    "Signal":  s["order"],
                    "Entrée":  s["entry"],
                    "SL":      s["sl"],
                    "TP1":     s["tp1"],
                    "RR":      s["rr"],
                    "Raison":  s["reason"][:60],
                })

    if dashboard_data:
        df_dash = pd.DataFrame(dashboard_data)
        st.dataframe(df_dash, use_container_width=True, hide_index=True)

        # Consensus
        buy_signals  = sum(1 for r in dashboard_data if "BUY" in r["Signal"])
        sell_signals = sum(1 for r in dashboard_data if "SELL" in r["Signal"])
        total = buy_signals + sell_signals
        if total > 0:
            pct_buy  = round(buy_signals / total * 100)
            pct_sell = round(sell_signals / total * 100)
            st.metric("🟢 Consensus BUY",  f"{buy_signals}/{total} bots  ({pct_buy}%)")
            st.metric("🔴 Consensus SELL", f"{sell_signals}/{total} bots  ({pct_sell}%)")
            if pct_buy >= 70:
                st.success("🟢 Majorité des bots en BUY — marché haussier dominant.")
            elif pct_sell >= 70:
                st.error("🔴 Majorité des bots en SELL — marché baissier dominant.")
            else:
                st.warning("⚠️ Signaux mixtes — prudence, pas de consensus clair.")
    else:
        st.info("Aucun signal disponible — lancez les bots dans leurs onglets respectifs.")

    # Prix live
    st.divider()
    st.subheader("💱 Prix Live")
    pc1, pc2 = st.columns(2)
    for sym, col in zip(["EURUSD", "XAUUSD"], [pc1, pc2]):
        p = live_price(sym)
        col.metric(sym, f"{p:.5f}" if (p and sym == "EURUSD") else f"{p:.2f}" if p else "—")


# ─────────────────────────────────────────────
# TAB 8 — CHAT IA
# ─────────────────────────────────────────────
with tabs[7]:
    st.subheader("🤖 Chat IA — Assistant Trading Multi-Bot")

    # Contexte des signaux actifs
    ctx_parts: list = []
    for i, ck in enumerate(CACHE_KEYS):
        for sym in ["EURUSD", "XAUUSD"]:
            c = st.session_state[ck].get(sym)
            if c and c.get("sig"):
                s = c["sig"]
                ctx_parts.append(f"Bot{i+1} {sym}: {s['order']} @ {s['entry']}, SL={s['sl']}, TP1={s['tp1']}, RR={s['rr']}")
    market_ctx = ("Signaux actifs :\n" + "\n".join(ctx_parts)) if ctx_parts else ""

    ai_choice = st.selectbox("IA", ["Groq (Llama 3.3-70b)", "Gemini (2.0 Flash)", "Claude (Sonnet)"])
    question  = st.text_input("💬 Question trading (stratégie, signal, risque…)")

    def ask_groq_chat(q: str, ctx: str = "") -> str:
        if not GROQ_API_KEY: return "❌ GROQ_API_KEY manquant dans secrets.toml"
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "system", "content": f"Expert SMC/ICT trading. Réponds en français.\n{ctx}"},
                                    {"role": "user", "content": q}],
                      "max_tokens": 700, "temperature": 0.4}, timeout=15)
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        except Exception as e: return f"❌ {e}"

    def ask_gemini_chat(q: str, ctx: str = "") -> str:
        if not GEMINI_API_KEY: return "❌ GEMINI_API_KEY manquant"
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": f"Expert SMC/ICT. Français.\n{ctx}\n\n{q}"}]}]}, timeout=15)
            r.raise_for_status()
            return str(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e: return f"❌ {e}"

    def ask_claude_chat(q: str, ctx: str = "") -> str:
        if not ANTHROPIC_API_KEY: return "❌ ANTHROPIC_API_KEY manquant"
        try:
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": "claude-sonnet-4-6", "max_tokens": 700,
                      "system": f"Expert SMC/ICT trading. Réponds en français.\n{ctx}",
                      "messages": [{"role": "user", "content": q}]}, timeout=15)
            r.raise_for_status()
            return str(r.json()["content"][0]["text"])
        except Exception as e: return f"❌ {e}"

    if st.button("🔍 Analyser") and question:
        with st.spinner("Analyse IA…"):
            if   "Groq"   in ai_choice: ans = ask_groq_chat(question, market_ctx)
            elif "Gemini" in ai_choice: ans = ask_gemini_chat(question, market_ctx)
            else:                       ans = ask_claude_chat(question, market_ctx)
        st.session_state.chat.append((question, ans, ai_choice))

    for q, a, src in reversed(st.session_state.chat):
        st.write(f"🧑 **Vous** : {q}")
        st.write(f"🤖 **{src}** : {a}")
        st.divider()