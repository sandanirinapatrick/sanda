# pyright: basic
# type: ignore
"""
╔══════════════════════════════════════════════════════════════╗
║   6 BOTS IA TRADING — SMC / ICT / VWAP / VOLUME             ║
║   Données : MetaTrader 5 (temps réel)                        ║
║   Symbole : XAUUSD uniquement                                ║
║   Bot 1 : Swing Très Puissant                                ║
║   Bot 2 : Day Trading Puissant (Arbre)                       ║
║   Bot 3 : Day Trading Très Puissant                          ║
║   Bot 4 : Scalping Puissant                                  ║
║   Bot 5 : Scalping Très Puissant                             ║
║   Bot 6 : Swing Puissant                                     ║
╚══════════════════════════════════════════════════════════════╝

requirements.txt :
    streamlit
    streamlit-autorefresh
    pandas
    numpy
    plotly
    requests
    MetaTrader5

secrets.toml :
    MT5_LOGIN      = "12345678"          # Numéro de compte MT5
    MT5_PASSWORD   = "votre_mot_de_passe"
    MT5_SERVER     = "MetaQuotes-Demo"   # Nom du serveur broker
    MT5_PATH       = ""                  # Optionnel : chemin vers terminal.exe
    GROQ_API_KEY   = "..."
    ANTHROPIC_API_KEY = "..."
    GEMINI_API_KEY = "..."
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import threading
from datetime import datetime, timezone, timedelta

from streamlit_autorefresh import st_autorefresh

# ──────────────────────────────────────────────────────
# MetaTrader 5 — imports
# ──────────────────────────────────────────────────────
MT5_AVAILABLE = False
_import_error = ""
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError as e:
    _import_error = str(e)

# Mapping timeframes MT5
MT5_TF = {}
if MT5_AVAILABLE:
    MT5_TF = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }

# Bars à télécharger par timeframe
TF_BARS = {
    "W1":  200,
    "D1":  365,
    "H4":  500,
    "H1":  720,
    "M15": 480,
    "M5":  576,
    "M1":  480,
}

# ──────────────────────────────────────────────────────
# CONFIG STREAMLIT
# ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="6 Bots Trading IA — XAUUSD",
    layout="wide",
    page_icon="🥇",
)
st_autorefresh(interval=30_000, key="auto_refresh")

# ──────────────────────────────────────────────────────
# SECRETS
# ──────────────────────────────────────────────────────
try:
    GROQ_API_KEY      = st.secrets.get("GROQ_API_KEY", "")
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY    = st.secrets.get("GEMINI_API_KEY", "")
    MT5_LOGIN         = int(st.secrets.get("MT5_LOGIN", "0"))
    MT5_PASSWORD      = st.secrets.get("MT5_PASSWORD", "")
    MT5_SERVER        = st.secrets.get("MT5_SERVER", "MetaQuotes-Demo")
    MT5_PATH          = st.secrets.get("MT5_PATH", "")
except Exception:
    GROQ_API_KEY = ANTHROPIC_API_KEY = GEMINI_API_KEY = ""
    MT5_LOGIN = 0
    MT5_PASSWORD = MT5_SERVER = MT5_PATH = ""

# ──────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────
_defaults = {
    "b1_signals": {}, "b2_signals": {}, "b3_signals": {},
    "b4_signals": {}, "b5_signals": {}, "b6_signals": {},
    "chat": [],
    "mt5_ready": False,
    "mt5_live_bid": None,
    "mt5_live_ask": None,
    "mt5_bar_cache": {},
    "mt5_cache_ts": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════
# MetaTrader 5 Manager
# ══════════════════════════════════════════════════════
class MT5Manager:
    """
    Gère la connexion MetaTrader 5.
    MT5 fonctionne nativement en synchrone sur Windows.
    Sur Linux/Mac, la connexion n'est pas supportée nativement —
    dans ce cas, un fallback Yahoo Finance est utilisé pour le prix live.
    """

    _instance: "MT5Manager | None" = None

    @classmethod
    def get(cls) -> "MT5Manager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._connected   = False
        self._initialized = False
        self._lock        = threading.Lock()

    def connect(self) -> bool:
        """Initialise et connecte MT5."""
        if not MT5_AVAILABLE:
            return False
        with self._lock:
            if self._connected:
                return True
            try:
                kwargs = {}
                if MT5_PATH:
                    kwargs["path"] = MT5_PATH
                if not mt5.initialize(**kwargs):
                    return False
                self._initialized = True

                # Connexion au compte si identifiants fournis
                if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
                    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                        # Certains brokers ne nécessitent pas de login explicite
                        pass

                # Vérifier que XAUUSD est disponible
                info = mt5.symbol_info("XAUUSD")
                if info is None:
                    # Essayer d'activer le symbole
                    mt5.symbol_select("XAUUSD", True)
                    info = mt5.symbol_info("XAUUSD")

                self._connected = info is not None
                try:
                    st.session_state["mt5_ready"] = self._connected
                except Exception:
                    pass
                return self._connected
            except Exception:
                return False

    def disconnect(self):
        with self._lock:
            if self._initialized and MT5_AVAILABLE:
                mt5.shutdown()
            self._connected   = False
            self._initialized = False

    def fetch_bars(self, tf: str, bars: int = 300) -> "pd.DataFrame | None":
        """Récupère les bougies OHLCV depuis MT5."""
        if not MT5_AVAILABLE:
            return None
        if not self._connected:
            if not self.connect():
                return None

        tf_mt5 = MT5_TF.get(tf)
        if tf_mt5 is None:
            return None

        try:
            rates = mt5.copy_rates_from_pos("XAUUSD", tf_mt5, 0, min(bars, 99999))
            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["time"] = df["time"].dt.tz_localize(None)  # naive UTC
            df = df.rename(columns={
                "open":       "open",
                "high":       "high",
                "low":        "low",
                "close":      "close",
                "tick_volume": "tick_volume",
            })
            df = df[["time", "open", "high", "low", "close", "tick_volume"]]
            df = df.sort_values("time").reset_index(drop=True)
            return df.tail(bars).reset_index(drop=True)
        except Exception:
            return None

    def live_price(self) -> "float | None":
        """Prix bid/ask en temps réel via MT5."""
        if not MT5_AVAILABLE or not self._connected:
            return None
        try:
            tick = mt5.symbol_info_tick("XAUUSD")
            if tick is None:
                return None
            bid = tick.bid
            ask = tick.ask
            try:
                st.session_state["mt5_live_bid"] = bid
                st.session_state["mt5_live_ask"] = ask
            except Exception:
                pass
            return round((bid + ask) / 2, 2) if bid and ask else (bid or ask)
        except Exception:
            return None

    @property
    def ready(self) -> bool:
        return self._connected


# ══════════════════════════════════════════════════════
# INITIALISATION DU MANAGER (une seule fois par session)
# ══════════════════════════════════════════════════════
_mt5 = MT5Manager.get()
if MT5_AVAILABLE and MT5_LOGIN:
    _mt5.connect()

# ══════════════════════════════════════════════════════
# FETCH DATA (avec cache Streamlit 20s)
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=20, show_spinner=False)
def fetch(tf: str, bars: int = 300) -> "pd.DataFrame | None":
    return _mt5.fetch_bars(tf, bars)


def live_price() -> "float | None":
    # Essayer MT5 en premier
    p = _mt5.live_price()
    if p:
        return p
    # Fallback Yahoo Finance si MT5 non dispo (Linux/Mac)
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF?interval=1m&range=1d",
            timeout=5,
        )
        data = r.json()
        return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────
# INDICATEURS COMMUNS
# ──────────────────────────────────────────────────────
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
    # VWAP rolling 20
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
        lo, hi = round(float(c["low"]), 2), round(float(c["high"]), 2)
        broken = False
        for _, row in recent.iloc[i+3:].iterrows():
            cf = float(row["close"])
            if kind == "BULL" and cf < lo: broken = True; break
            if kind == "BEAR" and cf > hi: broken = True; break
        if not broken:
            res.append({"kind": kind, "low": lo, "high": hi,
                        "mid": round((lo+hi)/2, 2), "time": recent["time"].iloc[i], "idx": i})
    return res[-4:]


def detect_fvg(df: pd.DataFrame) -> list:
    fvgs: list = []
    recent = df.tail(80).reset_index(drop=True)
    for i in range(1, len(recent)-1):
        p = recent.iloc[i-1]; n = recent.iloc[i+1]; t = recent["time"].iloc[i]
        if float(n["low"]) > float(p["high"]):
            fvgs.append({"type": "BULL", "top": round(float(n["low"]), 2),
                         "bot": round(float(p["high"]), 2), "time": t})
        elif float(n["high"]) < float(p["low"]):
            fvgs.append({"type": "BEAR", "top": round(float(p["low"]), 2),
                         "bot": round(float(n["high"]), 2), "time": t})
    return fvgs[-6:]


def detect_bos(df: pd.DataFrame) -> str:
    if len(df) < 30: return "NONE"
    w = df.tail(30)
    highs = w["high"].values; lows = w["low"].values
    if highs[-1] > highs[-15:].max() * 0.999: return "BULL"
    if lows[-1]  < lows[-15:].min()  * 1.001: return "BEAR"
    return "NONE"


def session_ok(kind: str = "both") -> bool:
    h = datetime.now(timezone.utc).hour
    lon = 8  <= h <= 11
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


def make_signal(order: str, entry: float, atr: float,
                bot: str, reason: str,
                rr1: float = 1.5, rr2: float = 2.5, rr3: float = 3.5) -> dict:
    d = 2  # XAUUSD
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
    return {"order": order, "entry": entry, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "rr": rr, "atr": round(atr, d), "bot": bot,
            "reason": reason, "sym": "XAUUSD", "ts": time.time()}


# ──────────────────────────────────────────────────────
# HELPERS ARBRE
# ──────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════
def make_chart(df: pd.DataFrame, sig: dict | None, obs: list, fvgs: list,
               title: str, n: int = 60) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#131722", plot_bgcolor="#131722",
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
        name="XAUUSD",
        increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
        decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
    ), row=1, col=1)

    for col_n, col_c, lbl in [("ema20", "#f39c12", "EMA20"),
                               ("ema50", "#3498db", "EMA50"),
                               ("ema200", "#e74c3c", "EMA200")]:
        if col_n in pl.columns:
            fig.add_trace(go.Scatter(x=pl["time"], y=pl[col_n],
                line=dict(color=col_c, width=1.2), name=lbl), row=1, col=1)

    if "vwap" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["vwap"],
            line=dict(color="#e056fd", width=1.4, dash="dot"), name="VWAP"), row=1, col=1)

    for ob in obs:
        x0 = ob["time"] if ob["time"] >= t0 else t0
        fc = "rgba(0,200,120,0.18)"  if ob["kind"] == "BULL" else "rgba(255,80,80,0.18)"
        lc = "rgba(0,220,130,0.9)"   if ob["kind"] == "BULL" else "rgba(255,90,90,0.9)"
        lbl = "OB ↑"                 if ob["kind"] == "BULL" else "OB ↓"
        fig.add_shape(type="rect", x0=x0, x1=t1, y0=ob["low"], y1=ob["high"],
                      fillcolor=fc, line=dict(color=lc, width=1.5), row=1, col=1)
        fig.add_annotation(x=t1, y=ob["mid"], text=lbl, showarrow=False,
                           font=dict(size=9, color=lc), xanchor="left", row=1, col=1)

    for fvg in fvgs:
        if fvg["time"] < t0: continue
        fc = "rgba(0,180,255,0.12)"  if fvg["type"] == "BULL" else "rgba(255,140,0,0.12)"
        lc = "rgba(0,180,255,0.8)"   if fvg["type"] == "BULL" else "rgba(255,140,0,0.8)"
        fig.add_shape(type="rect", x0=fvg["time"], x1=t1, y0=fvg["bot"], y1=fvg["top"],
                      fillcolor=fc, line=dict(color=lc, width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=t1, y=(fvg["bot"]+fvg["top"])/2,
                           text="FVG↑" if fvg["type"] == "BULL" else "FVG↓",
                           showarrow=False, font=dict(size=8, color=lc), xanchor="left", row=1, col=1)

    if sig:
        ac = "#00e676" if "BUY" in sig["order"] else "#ff5252"
        ep = sig["entry"]
        fig.add_hline(y=ep, line_color=ac, line_dash="solid", line_width=2,
                      annotation_text=f"  ▶ {sig['order']} @ {ep}",
                      annotation_position="right", annotation_font=dict(color=ac, size=10))
        fig.add_hline(y=sig["sl"], line_color="rgba(230,50,50,0.85)", line_dash="dash",
                      annotation_text="  🛑 SL", annotation_position="right",
                      annotation_font=dict(color="rgba(230,50,50,0.9)", size=9))
        for tp, tc, lbl in [(sig["tp1"], "rgba(0,210,100,0.9)",  "TP1"),
                             (sig["tp2"], "rgba(0,150,210,0.9)",  "TP2"),
                             (sig["tp3"], "rgba(160,60,230,0.9)", "TP3")]:
            fig.add_hline(y=tp, line_color=tc, line_dash="dot",
                          annotation_text=f"  {lbl}", annotation_position="right",
                          annotation_font=dict(color=tc, size=9))

    if "rsi" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["rsi"],
            line=dict(color="#9b59b6", width=1.6), name="RSI14"), row=2, col=1)
    if "stoch_k" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["stoch_k"],
            line=dict(color="#f39c12", width=1, dash="dot"), name="K"), row=2, col=1)
    if "stoch_d" in pl.columns:
        fig.add_trace(go.Scatter(x=pl["time"], y=pl["stoch_d"],
            line=dict(color="#3498db", width=1, dash="dot"), name="D"), row=2, col=1)
    for lvl, lc in [(70, "rgba(255,80,80,0.6)"), (30, "rgba(80,200,80,0.6)"),
                    (50, "rgba(150,150,150,0.4)")]:
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


# ──────────────────────────────────────────────────────
# AFFICHAGE SIGNAL
# ──────────────────────────────────────────────────────
def signal_card(sig: dict):
    if not sig: return
    col = "🟢" if "BUY" in sig["order"] else "🔴"
    st.success(f"""
{col} **{sig['order']}** — XAUUSD
📍 Entrée : `{sig['entry']:.2f}`   |   🛑 SL : `{sig['sl']:.2f}`   |   R:R **{sig['rr']}**
🥇 TP1 : `{sig['tp1']:.2f}`   🥈 TP2 : `{sig['tp2']:.2f}`   🥉 TP3 : `{sig['tp3']:.2f}`
🔍 **Raison** : {sig['reason']}
⏱ ATR : `{sig['atr']:.2f}`
""")
    st.markdown("🔗 [TradingView XAUUSD](https://www.tradingview.com/chart/?symbol=XAUUSD)  "
                "🔗 [Investing.com Gold](https://www.investing.com/commodities/gold)")


def no_signal(reason: str = ""):
    st.info(f"⏸️ Pas de signal — {reason}" if reason else "⏸️ Conditions non remplies — attendre.")


# ══════════════════════════════════════════════════════
# BOT 1 — SWING TRÈS PUISSANT
# ══════════════════════════════════════════════════════
def bot1_run() -> tuple:
    steps: list = []
    dfw1 = fetch("W1"); dfd1 = fetch("D1")
    if dfw1 is None or dfd1 is None:
        return None, ["❌ Données W1/D1 indisponibles — vérifier connexion MT5."]
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

    dfh4 = fetch("H4")
    if dfh4 is None: return None, steps + ["2. ❌ H4 indisponible"]
    dfh4 = indicators(dfh4)
    obs_h4 = detect_ob(dfh4)
    matching_obs = [ob for ob in obs_h4 if ob["kind"] == ("BULL" if direction == "HAUSSIER" else "BEAR")]
    if not matching_obs:
        steps.append("2. ❌ Aucun Order Block H4 dans la direction — attendre.")
        return None, steps
    ob_ref = matching_obs[-1]
    steps.append(f"2. ✅ OB H4 : {ob_ref['low']:.2f} → {ob_ref['high']:.2f}")

    fvgs_h4 = detect_fvg(dfh4)
    fvg_ok  = any((f["type"] == "BULL" and direction == "HAUSSIER") or
                  (f["type"] == "BEAR" and direction == "BAISSIER") for f in fvgs_h4)
    steps.append("3. ✅ FVG H4 détecté." if fvg_ok else "3. ⚠️ Pas de FVG clair — surveiller.")

    near = price_near_ob(dfh4, ob_ref)
    if not near:
        steps.append("4. ⏳ Prix pas encore dans la zone OB — attendre.")
        return None, steps
    steps.append("4. ✅ Prix dans / proche de la zone OB.")

    rej = (rejection_candle(dfh4, "BUY") or bullish_confirm(dfh4)) if direction == "HAUSSIER" \
          else (rejection_candle(dfh4, "SELL") or bearish_confirm(dfh4))
    if not rej:
        steps.append("5. ❌ Pas de bougie de rejet/confirmation — invalider.")
        return None, steps
    steps.append("5. ✅ Bougie de rejet/confirmation présente.")

    vol_ok = volume_strong(dfh4, 1.2)
    steps.append("6. ✅ Volume fort — mouvement institutionnel." if vol_ok
                 else "6. ⚠️ Volume moyen — SL serré.")

    entry = float(dfh4.iloc[-1]["close"]); atr = float(dfh4.iloc[-1]["atr"])
    order  = "BUY LIMIT" if direction == "HAUSSIER" else "SELL LIMIT"
    reason = f"Swing {direction} | W1/D1/H4 | OB H4 {ob_ref['low']:.2f}-{ob_ref['high']:.2f}"
    if fvg_ok: reason += " | FVG ✅"
    if vol_ok: reason += " | Volume ✅"
    steps.append("7. ✅ 1-2% capital | SL sous OB | TP highs/lows précédents.")
    steps.append(f"8. 🎯 **ENTRÉE H4** — {order} @ {entry:.2f}")
    sig = make_signal(order, entry, atr, "Bot1-SwingTrèsPuissant", reason, rr1=2.0, rr2=3.5, rr3=5.0)
    return sig, steps


# ══════════════════════════════════════════════════════
# BOT 2 — DAY TRADING PUISSANT (ARBRE 4 PHASES)
# ══════════════════════════════════════════════════════
def bot2_run() -> tuple:
    steps: list = []
    if not session_ok("both"):
        steps.append("Phase 0. ❌ Hors session London/NY (08-11h / 13-16h UTC) — ne pas trader.")
        return None, steps
    steps.append("Phase 0. ✅ Session London ou NY active.")

    dfh1 = fetch("H1")
    if dfh1 is None: return None, steps + ["Phase 1. ❌ H1 indisponible"]
    dfh1 = indicators(dfh1)
    biais_h1 = trend_direction(dfh1)
    if biais_h1 == "NEUTRE":
        steps.append("Phase 1. ❌ Biais H1 neutre — attendre.")
        return None, steps
    steps.append(f"Phase 1. ✅ Biais H1 : **{biais_h1}**")

    dfm15 = fetch("M15")
    if dfm15 is None: return None, steps + ["Phase 1b. ❌ M15 indisponible"]
    dfm15 = indicators(dfm15)
    direction  = biais_h1
    kind_want  = "BULL" if direction == "HAUSSIER" else "BEAR"
    obs_ok     = [ob for ob in detect_ob(dfm15)  if ob["kind"] == kind_want]
    fvgs_ok    = [f  for f  in detect_fvg(dfm15) if f["type"]  == kind_want]
    if not obs_ok and not fvgs_ok:
        steps.append("Phase 1b. ❌ Pas d'OB/FVG M15 — attendre.")
        return None, steps
    steps.append(f"Phase 1b. ✅ {'OB' if obs_ok else 'FVG'} M15 trouvé.")

    vwap_ok = vwap_filter(dfm15, "BUY" if direction == "HAUSSIER" else "SELL")
    vol_ok  = volume_strong(dfm15, 1.25)
    if not vwap_ok:
        steps.append("Phase 2. ❌ Mauvais côté du VWAP — signal annulé.")
        return None, steps
    steps.append(f"Phase 2. ✅ VWAP aligné | Volume {'fort ✅' if vol_ok else 'moyen ⚠️'}")

    dfm5 = fetch("M5")
    if dfm5 is None: return None, steps + ["Phase 3. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    obs_m5  = detect_ob(dfm5)
    near_m5 = any(price_near_ob(dfm5, ob) for ob in obs_m5 if ob["kind"] == kind_want)
    rej_m5  = rejection_candle(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    if not near_m5 or not rej_m5:
        steps.append("Phase 3. ❌ Pas de rejet M5 propre — attendre.")
        return None, steps
    steps.append("Phase 3. ✅ Rejet M5 propre sur OB/FVG — signal VERT.")

    entry = float(dfm5.iloc[-1]["close"]); atr = float(dfm5.iloc[-1]["atr"])
    order  = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Day {direction} | H1→M15→M5 | VWAP ✅ | Vol {'✅' if vol_ok else '⚠️'}"
    steps.append(f"Phase 4. ✅ 1-2% capital | SL sous OB/FVG.")
    steps.append(f"Phase 4. 🎯 **SIGNAL** {order} @ {entry:.2f}")
    sig = make_signal(order, entry, atr, "Bot2-DayTradingPuissant", reason, rr1=1.5, rr2=2.5, rr3=3.5)
    return sig, steps


# ══════════════════════════════════════════════════════
# BOT 3 — DAY TRADING TRÈS PUISSANT
# ══════════════════════════════════════════════════════
def bot3_run() -> tuple:
    steps: list = []
    dfd1 = fetch("D1"); dfh1 = fetch("H1")
    if dfd1 is None or dfh1 is None: return None, ["❌ D1/H1 indisponibles"]
    dfd1 = indicators(dfd1); dfh1 = indicators(dfh1)
    b_d1 = trend_direction(dfd1); b_h1 = trend_direction(dfh1)
    if b_d1 == "NEUTRE":
        steps.append("1. ❌ Biais D1 neutre — ignorer.")
        return None, steps
    steps.append(f"1. ✅ Biais D1={b_d1}  H1={b_h1}")
    direction = b_d1

    if not session_ok():
        steps.append("2. ❌ Hors session London/NY — ne pas trader.")
        return None, steps
    steps.append("2. ✅ Session favorable.")

    dfm15 = fetch("M15")
    if dfm15 is None: return None, steps + ["3. ❌ M15 indisponible"]
    dfm15 = indicators(dfm15)
    kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    obs_m15  = [ob for ob in detect_ob(dfm15)  if ob["kind"] == kind_want]
    fvgs_m15 = [f  for f  in detect_fvg(dfm15) if f["type"]  == kind_want]
    if not obs_m15 and not fvgs_m15:
        steps.append("3. ❌ Pas d'OB/FVG M15.")
        return None, steps
    steps.append(f"3. ✅ {'OB' if obs_m15 else 'FVG'} M15 aligné avec D1.")

    ob_ref = obs_m15[-1] if obs_m15 else None
    near   = ob_ref and price_near_ob(dfm15, ob_ref)
    if not near:
        steps.append("4. ⏳ Prix pas encore sur la zone OB/FVG M15.")
        return None, steps
    steps.append(f"4. ✅ Prix reteste OB/FVG M15 ({ob_ref['low']:.2f}→{ob_ref['high']:.2f}).")

    vwap_ok = vwap_filter(dfm15, "BUY" if direction == "HAUSSIER" else "SELL")
    vol_ok  = volume_strong(dfm15, 1.2)
    conf    = sum([vwap_ok, vol_ok, bool(obs_m15), bool(fvgs_m15)])
    steps.append(f"5. Confluence : {conf}/4 — VWAP={'✅' if vwap_ok else '❌'} Vol={'✅' if vol_ok else '⚠️'}")
    if conf < 2:
        steps.append("5. ❌ Confluence insuffisante.")
        return None, steps

    dfm5 = fetch("M5")
    if dfm5 is None: return None, steps + ["6. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    rej  = rejection_candle(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    if not rej:
        steps.append("6. ❌ Pas de bougie de rejet M5.")
        return None, steps
    steps.append("6. ✅ Bougie de rejet M5 confirmée.")

    bos    = detect_bos(dfm5)
    bos_ok = (bos == "BULL" and direction == "HAUSSIER") or (bos == "BEAR" and direction == "BAISSIER")
    steps.append(f"7. BOS M5 : {bos} {'✅' if bos_ok else '⚠️'}")

    entry = float(dfm5.iloc[-1]["close"]); atr = float(dfm5.iloc[-1]["atr"])
    order  = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Day Très Puissant {direction} | D1/H1/M15/M5 | Conf {conf}/4"
    if vwap_ok: reason += " | VWAP ✅"
    if bos_ok:  reason += " | BOS ✅"
    steps.append("8. ✅ 1-2% capital | SL sous OB | TP 1:2-1:3")
    steps.append(f"9. 🎯 **SIGNAL** {order} @ {entry:.2f}")
    sig = make_signal(order, entry, atr, "Bot3-DayTradingTrèsPuissant", reason, rr1=1.5, rr2=2.5, rr3=3.5)
    return sig, steps


# ══════════════════════════════════════════════════════
# BOT 4 — SCALPING PUISSANT
# ══════════════════════════════════════════════════════
def bot4_run() -> tuple:
    steps: list = []
    if not session_ok("both"):
        steps.append("0. ❌ Hors Kill Zone (08-11h / 13-16h UTC) — ne pas scalper.")
        return None, steps
    steps.append("0. ✅ Kill Zone active.")

    dfh1 = fetch("H1")
    if dfh1 is None: return None, ["❌ H1 indisponible"]
    dfh1 = indicators(dfh1)
    b_h1 = trend_direction(dfh1)
    if b_h1 == "NEUTRE":
        steps.append("1. ❌ H1 biais neutre — passer.")
        return None, steps
    direction = b_h1; kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    steps.append(f"1. ✅ H1 biais={direction}")

    dfm5 = fetch("M5")
    if dfm5 is None: return None, steps + ["1.2. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    obs_m5  = [ob for ob in detect_ob(dfm5)  if ob["kind"] == kind_want]
    fvgs_m5 = [f  for f  in detect_fvg(dfm5) if f["type"] == kind_want]
    if not obs_m5 and not fvgs_m5:
        steps.append("1.3. ❌ Pas d'OB/FVG M5 — attendre.")
        return None, steps
    steps.append(f"1.3. ✅ M5 : {len(obs_m5)} OBs / {len(fvgs_m5)} FVGs.")

    ob_m5   = obs_m5[-1] if obs_m5 else None
    near_m5 = ob_m5 and price_near_ob(dfm5, ob_m5)
    if not near_m5:
        steps.append("1.4. ⏳ Prix pas sur OB/FVG M5 — attendre.")
        return None, steps
    steps.append(f"1.4. ✅ Prix reteste OB M5 : {ob_m5['low']:.2f}→{ob_m5['high']:.2f}")

    vwap_ok = vwap_filter(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    vol_ok  = volume_strong(dfm5, 1.3)
    if not vwap_ok:
        steps.append("2. ❌ Mauvais côté VWAP — signal annulé.")
        return None, steps
    steps.append(f"2. ✅ VWAP + Volume {'fort ✅' if vol_ok else 'moyen ⚠️'}")

    dfm1 = fetch("M1")
    if dfm1 is None: return None, steps + ["3. ❌ M1 indisponible"]
    dfm1 = indicators(dfm1)
    rej_m1  = rejection_candle(dfm1, "BUY" if direction == "HAUSSIER" else "SELL")
    bull_m1 = bullish_confirm(dfm1) if direction == "HAUSSIER" else bearish_confirm(dfm1)
    ob_m1   = detect_ob(dfm1)
    near_m1 = any(price_near_ob(dfm1, ob) for ob in ob_m1 if ob["kind"] == kind_want)
    if not (rej_m1 or bull_m1) or not near_m1:
        steps.append("3. ❌ Bougie M1 pas propre — annuler.")
        return None, steps
    steps.append("3. ✅ Bougie M1 propre sur OB/FVG.")

    entry = float(dfm1.iloc[-1]["close"]); atr = float(dfm1.iloc[-1]["atr"])
    order  = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Scalping {direction} | H1→M5→M1 | VWAP ✅ | Vol {'✅' if vol_ok else '⚠️'} | KZ ✅"
    steps.append("4. ✅ 0.5% capital | SL sous rejet | TP 1:2")
    steps.append(f"4. 🎯 **SCALP** {order} @ {entry:.2f}")
    sig = make_signal(order, entry, atr, "Bot4-ScalpingPuissant", reason, rr1=1.0, rr2=1.5, rr3=2.0)
    return sig, steps


# ══════════════════════════════════════════════════════
# BOT 5 — SCALPING TRÈS PUISSANT
# ══════════════════════════════════════════════════════
def bot5_run() -> tuple:
    steps: list = []
    dfh1 = fetch("H1")
    if dfh1 is None: return None, ["❌ H1 indisponible"]
    dfh1 = indicators(dfh1)
    biais = trend_direction(dfh1)
    if biais == "NEUTRE":
        steps.append("1. ❌ Biais H1 pas clair — attendre.")
        return None, steps
    direction = biais; kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    steps.append(f"1. ✅ Biais H1 : **{direction}**")

    if not session_ok():
        steps.append("2. ❌ Hors Kill Zone — ne pas scalper.")
        return None, steps
    steps.append("2. ✅ Kill Zone active (London / NY).")

    dfm5 = fetch("M5")
    if dfm5 is None: return None, steps + ["3. ❌ M5 indisponible"]
    dfm5 = indicators(dfm5)
    obs_m5  = [ob for ob in detect_ob(dfm5)  if ob["kind"] == kind_want]
    fvgs_m5 = [f  for f  in detect_fvg(dfm5) if f["type"]  == kind_want]
    if not obs_m5 and not fvgs_m5:
        steps.append("3. ❌ Pas d'OB/FVG M5 — attendre.")
        return None, steps
    steps.append(f"3. ✅ M5 : {len(obs_m5)} OBs + {len(fvgs_m5)} FVGs.")

    ob_ref = obs_m5[-1] if obs_m5 else None
    near   = ob_ref and price_near_ob(dfm5, ob_ref)
    if not near:
        steps.append("4. ⏳ Prix pas sur l'OB/FVG M5 — attendre.")
        return None, steps
    steps.append(f"4. ✅ Prix reteste OB M5 : {ob_ref['low']:.2f}→{ob_ref['high']:.2f}")

    vwap_ok = vwap_filter(dfm5, "BUY" if direction == "HAUSSIER" else "SELL")
    steps.append(f"5. VWAP : {'✅' if vwap_ok else '❌ défavorable — attendre.'}")
    if not vwap_ok:
        return None, steps

    vol_ok = volume_strong(dfm5, 1.2)
    steps.append(f"6. Volume : {'✅ fort' if vol_ok else '⚠️ faible — TP 1:1'}")

    dfm1 = fetch("M1")
    if dfm1 is None: return None, steps + ["7. ❌ M1 indisponible"]
    dfm1 = indicators(dfm1)
    rej_m1  = rejection_candle(dfm1, "BUY" if direction == "HAUSSIER" else "SELL")
    conf_m1 = bullish_confirm(dfm1) if direction == "HAUSSIER" else bearish_confirm(dfm1)
    if not (rej_m1 or conf_m1):
        steps.append("7. ❌ Pas de rejet/clôture propre M1.")
        return None, steps
    steps.append("7. ✅ M1 : rebond/clôture dans la bonne direction.")

    entry = float(dfm1.iloc[-1]["close"]); atr = float(dfm1.iloc[-1]["atr"])
    order  = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = f"Scalp Très Puissant {direction} | H1/M5/M1 | KZ ✅ | VWAP ✅ | Vol {'✅' if vol_ok else '⚠️'}"
    steps.append("8. ✅ 1-2% capital | SL sous OB/FVG | TP 1:1-1:3")
    steps.append(f"9. 🎯 **SCALP M1** {order} @ {entry:.2f}")
    rr1 = 1.0 if not vol_ok else 1.5
    sig = make_signal(order, entry, atr, "Bot5-ScalpingTrèsPuissant", reason, rr1=rr1, rr2=2.0, rr3=3.0)
    return sig, steps


# ══════════════════════════════════════════════════════
# BOT 6 — SWING PUISSANT
# ══════════════════════════════════════════════════════
def bot6_run() -> tuple:
    steps: list = []
    dfw1 = fetch("W1")
    if dfw1 is None: return None, ["❌ W1 indisponible"]
    dfw1 = indicators(dfw1)
    b_w1 = trend_direction(dfw1)
    if b_w1 == "NEUTRE":
        steps.append("W1. ❌ Biais macro W1 non clair — attendre.")
        return None, steps
    direction = b_w1; kind_want = "BULL" if direction == "HAUSSIER" else "BEAR"
    w1_close = dfw1["close"].values
    hh_hl = w1_close[-1] > w1_close[-3] and w1_close[-2] > w1_close[-4]
    lh_ll = w1_close[-1] < w1_close[-3] and w1_close[-2] < w1_close[-4]
    steps.append(f"W1. ✅ Biais macro : **{direction}** | HH/HL={'✅' if hh_hl else '—'} LH/LL={'✅' if lh_ll else '—'}")

    dfd1 = fetch("D1")
    if dfd1 is None: return None, steps + ["D1. ❌ D1 indisponible"]
    dfd1 = indicators(dfd1)
    obs_d1  = [ob for ob in detect_ob(dfd1)  if ob["kind"] == kind_want]
    fvgs_d1 = [f  for f  in detect_fvg(dfd1) if f["type"]  == kind_want]
    bos_d1  = detect_bos(dfd1)
    bos_ok  = (bos_d1 == "BULL" and direction == "HAUSSIER") or (bos_d1 == "BEAR" and direction == "BAISSIER")

    if not obs_d1 and not fvgs_d1:
        steps.append("D1. ❌ Pas d'OB/FVG D1 — pas d'entrée.")
        return None, steps
    steps.append(f"D1. {'✅' if bos_ok else '⚠️'} BOS D1 : {bos_d1}")

    ob_d1 = obs_d1[-1] if obs_d1 else None
    if ob_d1:
        steps.append(f"D1. ✅ OB D1 : {ob_d1['low']:.2f} → {ob_d1['high']:.2f}")

    near_d1 = ob_d1 and price_near_ob(dfd1, ob_d1, tol_atr=1.0)
    if not near_d1:
        steps.append("D1. ⏳ Prix loin de la zone OB/FVG D1 — attendre.")
        return None, steps
    steps.append("D1. ✅ Prix approche la zone OB/FVG D1.")

    dfh4 = fetch("H4")
    if dfh4 is None: return None, steps + ["H4. ❌ H4 indisponible"]
    dfh4 = indicators(dfh4)
    rej_h4 = rejection_candle(dfh4, "BUY" if direction == "HAUSSIER" else "SELL")
    if not rej_h4:
        steps.append("H4. ❌ Pas de bougie de rejet H4 — attendre.")
        return None, steps
    steps.append("H4. ✅ Bougie de rejet H4 confirmée.")

    entry = float(dfh4.iloc[-1]["close"]); atr = float(dfh4.iloc[-1]["atr"])
    if direction == "HAUSSIER":
        sl_ref = ob_d1["low"] if ob_d1 else entry - 2 * atr
    else:
        sl_ref = ob_d1["high"] if ob_d1 else entry + 2 * atr
    dist_sl = abs(entry - sl_ref)
    if dist_sl > 0.005 * entry:
        steps.append("H4. ⚠️ SL éloigné > 0.5% — réduire le lot.")

    order  = "BUY" if direction == "HAUSSIER" else "SELL"
    reason = (f"Swing Puissant {direction} | W1/D1/H4 | "
              f"OB D1 {ob_d1['low'] if ob_d1 else '?':.2f}-{ob_d1['high'] if ob_d1 else '?':.2f} "
              f"| BOS={'✅' if bos_ok else '⚠️'}")
    steps.append("H4. ✅ RR ≥ 1:3 | 1-2% capital | 1 trade swing ouvert.")
    steps.append("H4. ✅ Couvrir 50% à 1:2, trailing stop sur le reste.")
    steps.append(f"H4. 🎯 **SIGNAL SWING** {order} @ {entry:.2f}")
    sig = make_signal(order, entry, atr, "Bot6-SwingPuissant", reason, rr1=2.0, rr2=3.5, rr3=5.0)
    if direction == "HAUSSIER":
        sig["sl"] = round(sl_ref - 0.2 * atr, 2)
    else:
        sig["sl"] = round(sl_ref + 0.2 * atr, 2)
    return sig, steps


# ══════════════════════════════════════════════════════
# UI PRINCIPALE
# ══════════════════════════════════════════════════════
st.title("🥇 6 BOTS IA TRADING — XAUUSD — MetaTrader 5")
st.caption("Données temps réel MetaTrader 5 | Refresh automatique 30s")

# ── Bannière statut connexion ──────────────────────────
if not MT5_AVAILABLE:
    st.error(
        f"❌ Bibliothèque MetaTrader5 non installée. "
        f"Ajoutez `MetaTrader5` à requirements.txt. "
        f"⚠️ MetaTrader5 est uniquement supporté sur **Windows**. "
        f"({_import_error})"
    )
elif not MT5_LOGIN:
    st.warning(
        "⚠️ Identifiants MT5 manquants dans `.streamlit/secrets.toml`. "
        "Ajouter : MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (et optionnellement MT5_PATH)."
    )
else:
    ready = _mt5.ready
    if ready:
        acc_info = None
        try:
            acc_info = mt5.account_info()
        except Exception:
            pass
        if acc_info:
            st.success(
                f"✅ MT5 connecté | Compte : {acc_info.login} | "
                f"Broker : {acc_info.company} | "
                f"Serveur : {acc_info.server} | "
                f"Balance : {acc_info.balance:.2f} {acc_info.currency}"
            )
        else:
            st.success("✅ MT5 connecté — XAUUSD disponible.")
    else:
        st.warning(
            "⏳ Connexion MT5 en cours… "
            "Vérifiez que le terminal MetaTrader 5 est ouvert et connecté, "
            "puis relancez l'analyse."
        )

BOT_INFO = [
    ("🔵 Bot 1 — Swing Très Puissant",     "W1/D1/H4 | OB H4 | FVG | Volume/ATR"),
    ("🟠 Bot 2 — Day Trading Puissant",    "H1→M15→M5 | SMC + VWAP + Volume | London/NY"),
    ("🔴 Bot 3 — Day Trading Très Puissant","D1/H1/M15/M5 | Confluence 4 critères"),
    ("🟡 Bot 4 — Scalping Puissant",       "H1→M5→M1 | Kill Zone | VWAP | TP 1:2"),
    ("🟢 Bot 5 — Scalping Très Puissant",  "H1/M5/M1 | SMC/ICT | Kill Zone | VWAP"),
    ("🟣 Bot 6 — Swing Puissant",          "W1→D1→H4 | OB+FVG+BOS | RR ≥ 1:3"),
]

RUNNERS    = [bot1_run, bot2_run, bot3_run, bot4_run, bot5_run, bot6_run]
CACHE_KEYS = ["b1_signals", "b2_signals", "b3_signals", "b4_signals", "b5_signals", "b6_signals"]
TF_CHART   = ["H4", "M15", "M15", "M5", "M5", "H4"]

tabs = st.tabs([b[0] for b in BOT_INFO] + ["📊 Dashboard", "🤖 Chat IA"])

for i, (tab, (bot_name, bot_desc), runner, ck, tf_c) in enumerate(
    zip(tabs[:6], BOT_INFO, RUNNERS, CACHE_KEYS, TF_CHART)
):
    with tab:
        st.subheader(bot_name)
        st.caption(bot_desc)

        col_run, col_price = st.columns([1, 3])
        with col_run:
            run_btn = st.button("▶ Analyser", key=f"run_b{i+1}")
        with col_price:
            p = live_price()
            if p:
                st.metric("XAUUSD live", f"{p:.2f} $")

        if run_btn:
            if not MT5_AVAILABLE:
                st.error("MetaTrader5 non installé (Windows requis).")
            elif not _mt5.ready:
                st.warning("Connexion MT5 pas encore prête — vérifiez que le terminal MT5 est ouvert.")
            else:
                with st.spinner("Analyse multi-timeframes en cours…"):
                    sig, steps = runner()
                st.session_state[ck]["XAUUSD"] = {"sig": sig, "steps": steps, "ts": time.time()}

        cached = st.session_state[ck].get("XAUUSD")
        if cached:
            age = int(time.time() - cached["ts"])
            st.caption(f"⏱ Dernière analyse : il y a {age}s")
            st.divider()
            st.subheader("🌲 Arbre décisionnel")
            for step in cached["steps"]:
                if   "✅" in step: st.success(step)
                elif "❌" in step: st.error(step)
                elif "⏳" in step or "⚠️" in step: st.warning(step)
                elif "🎯" in step: st.info(step)
                else: st.write(step)

            st.divider()
            sig = cached["sig"]
            if sig:
                st.subheader("📡 Signal généré")
                signal_card(sig)
                st.subheader("📈 Graphique")
                df_chart = fetch(tf_c)
                if df_chart is not None:
                    df_chart = indicators(df_chart)
                    st.plotly_chart(
                        make_chart(df_chart, sig,
                                   detect_ob(df_chart), detect_fvg(df_chart),
                                   f"XAUUSD — {tf_c} — {bot_name}", n=60),
                        use_container_width=True,
                    )
            else:
                no_signal("Conditions non réunies — relancer après un cycle de marché.")
        else:
            st.info("Cliquer sur **▶ Analyser** pour lancer le bot.")

        st.markdown("🔗 [TradingView XAUUSD](https://www.tradingview.com/chart/?symbol=XAUUSD)  "
                    "🔗 [Investing.com Gold](https://www.investing.com/commodities/gold)")


# ── Dashboard ─────────────────────────────────────────
with tabs[6]:
    st.subheader("📊 Dashboard — Tous les Bots XAUUSD")

    # Infos compte MT5
    if MT5_AVAILABLE and _mt5.ready:
        try:
            acc = mt5.account_info()
            if acc:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Balance", f"{acc.balance:.2f} {acc.currency}")
                c2.metric("Equity",  f"{acc.equity:.2f} {acc.currency}")
                c3.metric("Marge libre", f"{acc.margin_free:.2f} {acc.currency}")
                c4.metric("Levier", f"1:{acc.leverage}")
                st.divider()
        except Exception:
            pass

    rows = []
    for i, (bot_name, _) in enumerate(BOT_INFO):
        c = st.session_state[CACHE_KEYS[i]].get("XAUUSD")
        if c and c.get("sig"):
            s = c["sig"]
            rows.append({"Bot": bot_name, "Signal": s["order"],
                         "Entrée": s["entry"], "SL": s["sl"],
                         "TP1": s["tp1"], "RR": s["rr"],
                         "Raison": s["reason"][:65]})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        buy  = sum(1 for r in rows if "BUY"  in r["Signal"])
        sell = sum(1 for r in rows if "SELL" in r["Signal"])
        tot  = buy + sell
        if tot:
            st.metric("🟢 Consensus BUY",  f"{buy}/{tot}  ({round(buy/tot*100)}%)")
            st.metric("🔴 Consensus SELL", f"{sell}/{tot} ({round(sell/tot*100)}%)")
            pct_b = buy / tot * 100
            pct_s = sell / tot * 100
            if pct_b >= 70:   st.success("🟢 Majorité BUY — marché haussier dominant.")
            elif pct_s >= 70: st.error("🔴 Majorité SELL — marché baissier dominant.")
            else:             st.warning("⚠️ Signaux mixtes — prudence.")
    else:
        st.info("Aucun signal — lancez les bots dans leurs onglets.")

    st.divider()
    p = live_price()
    st.metric("💰 XAUUSD Live", f"{p:.2f} $" if p else "—")


# ── Chat IA ───────────────────────────────────────────
with tabs[7]:
    st.subheader("🤖 Chat IA — Assistant Trading XAUUSD")

    ctx_parts = []
    for i, ck in enumerate(CACHE_KEYS):
        c = st.session_state[ck].get("XAUUSD")
        if c and c.get("sig"):
            s = c["sig"]
            ctx_parts.append(f"Bot{i+1} XAUUSD: {s['order']} @ {s['entry']:.2f}, "
                              f"SL={s['sl']:.2f}, TP1={s['tp1']:.2f}, RR={s['rr']}")
    market_ctx = ("Signaux actifs XAUUSD :\n" + "\n".join(ctx_parts)) if ctx_parts else ""

    ai_choice = st.selectbox("IA", ["Groq (Llama 3.3-70b)", "Gemini (2.0 Flash)", "Claude (Sonnet)"])
    question  = st.text_input("💬 Question trading (XAUUSD, stratégie, signal, risque…)")

    def ask_groq(q, ctx=""):
        if not GROQ_API_KEY: return "❌ GROQ_API_KEY manquant"
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "system",
                                    "content": f"Expert SMC/ICT XAUUSD. Réponds en français.\n{ctx}"},
                                   {"role": "user", "content": q}],
                      "max_tokens": 700, "temperature": 0.4}, timeout=15)
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        except Exception as e: return f"❌ {e}"

    def ask_gemini(q, ctx=""):
        if not GEMINI_API_KEY: return "❌ GEMINI_API_KEY manquant"
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": f"Expert SMC/ICT XAUUSD. Français.\n{ctx}\n\n{q}"}]}]},
                timeout=15)
            r.raise_for_status()
            return str(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e: return f"❌ {e}"

    def ask_claude(q, ctx=""):
        if not ANTHROPIC_API_KEY: return "❌ ANTHROPIC_API_KEY manquant"
        try:
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json={"model": "claude-sonnet-4-6", "max_tokens": 700,
                      "system": f"Expert SMC/ICT XAUUSD. Réponds en français.\n{ctx}",
                      "messages": [{"role": "user", "content": q}]},
                timeout=15)
            r.raise_for_status()
            return str(r.json()["content"][0]["text"])
        except Exception as e: return f"❌ {e}"

    if st.button("🔍 Analyser") and question:
        with st.spinner("Analyse IA…"):
            if   "Groq"   in ai_choice: ans = ask_groq(question, market_ctx)
            elif "Gemini" in ai_choice: ans = ask_gemini(question, market_ctx)
            else:                       ans = ask_claude(question, market_ctx)
        st.session_state.chat.append((question, ans, ai_choice))

    for q, a, src in reversed(st.session_state.chat):
        st.write(f"🧑 **Vous** : {q}")
        st.write(f"🤖 **{src}** : {a}")
        st.divider()
