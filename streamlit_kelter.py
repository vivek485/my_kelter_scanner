import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import time
import ta
import asyncio
import aiohttp
import nest_asyncio

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Keltner Screener")

# Apply nest_asyncio at startup so asyncio.run() works inside Streamlit
nest_asyncio.apply()

# ─────────────────────────────────────────────
# SYMBOL LIST  – put your own symbols here
# or import from a separate file: from symbols import symbols
# ─────────────────────────────────────────────
try:
    from symbols import symbols          # your own symbols.py
except ImportError:
    # fallback demo list so the app at least starts
    symbols = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "WIPRO", "AXISBANK", "LT", "KOTAKBANK"
    ]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def to_unix_ms(dt: datetime) -> str:
    """Convert datetime → Groww-style millisecond epoch string."""
    ms = int(dt.timestamp() * 1000)
    return str(ms)


def build_chart(final_df: pd.DataFrame, stock: str) -> go.Figure:
    """Return a Plotly candlestick figure with indicators."""
    fig = go.Figure(data=[
        go.Candlestick(
            x=final_df.index,
            open=final_df["Open"],
            close=final_df["Close"],   # ← was wrongly set to Open before
            high=final_df["High"],
            low=final_df["Low"],
            name=stock,
        ),
        go.Scatter(x=final_df.index, y=final_df["ma200"],
                   line=dict(color="red", width=1), name="MA200"),
        go.Scatter(x=final_df.index, y=final_df["highband"],
                   line=dict(color="blue", width=1), name="Upper Band"),
        go.Scatter(x=final_df.index, y=final_df["middleband"],
                   line=dict(color="royalblue", width=1, dash="dot"), name="Mid Band"),
        go.Scatter(x=final_df.index, y=final_df["lowerband"],
                   line=dict(color="blue", width=1), name="Lower Band"),
    ])
    # buy / sell markers
    fig.add_trace(go.Scatter(
        x=final_df.index, y=final_df["signal1p"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="lime"),
        name="Buy Signal",
    ))
    fig.add_trace(go.Scatter(
        x=final_df.index, y=final_df["signal2p"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=14, color="red"),
        name="Sell Signal",
    ))
    fig.update_layout(
        autosize=False,
        width=1700,
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=stock,
    )
    fig.layout.xaxis.type = "category"
    return fig


# ─────────────────────────────────────────────
# ASYNC DATA FETCH + INDICATOR COMPUTATION
# ─────────────────────────────────────────────
async def fetch_and_process(session: aiohttp.ClientSession,
                            stock: str,
                            fromdate: str,
                            todate: str,
                            interval: int) -> dict | None:
    """
    Fetch OHLCV data from Groww, compute all indicators and signals.
    Returns a result dict or None on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:96.0) "
            "Gecko/20100101 Firefox/96.0"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
    }
    url = (
        f"https://groww.in/v1/api/charting_service/v2/chart/"
        f"exchange/NSE/segment/CASH/{stock}"
        f"?endTimeInMillis={todate}"
        f"&intervalInMinutes={interval}"
        f"&startTimeInMillis={fromdate}"
    )

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
            resp = await response.json(content_type=None)

        candles = resp.get("candles")
        if not candles or len(candles) < 50:      # not enough data
            return None

        # ── Build DataFrame ──────────────────────────────────────────
        df = pd.DataFrame(candles, columns=["time", "Open", "High", "Low", "Close", "Volume"])
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["time"], inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)

        # ── Indicators ───────────────────────────────────────────────
        df["prevclose"]  = df["Close"].shift(1)
        df["prevlow"]    = df["Low"].shift(1)
        df["prevhigh"]   = df["High"].shift(1)

        # MA-200
        df["ma200"] = ta.trend.ema_indicator(df["Close"], window=200)

        # Stochastic
        df["st"]   = ta.momentum.stoch(high=df["High"], low=df["Low"],
                                       close=df["Close"], window=14, smooth_window=3)
        df["prst"] = df["st"].shift(1)

        # EMA of (H+L)/2
        hl              = (df["High"] + df["Low"]) / 2
        df["hl"]        = ta.trend.ema_indicator(hl, window=4)
        df["prhl"]      = df["hl"].shift(1)

        # Keltner Channel
        kc               = ta.volatility.KeltnerChannel(
                                close=df["Close"], high=df["High"], low=df["Low"],
                                original_version=False)
        df["highband"]   = kc.keltner_channel_hband()
        df["middleband"] = kc.keltner_channel_mband()
        df["lowerband"]  = kc.keltner_channel_lband()

        # MACD histogram
        df["macd_ind"]  = ta.trend.macd_diff(close=df["Close"],
                                              window_slow=26, window_fast=12,
                                              window_sign=9, fillna=False)
        df["prevmacd"]  = df["macd_ind"].shift(1)

        df["symbol"] = stock

        # ── Signal components ────────────────────────────────────────
        df["sig_sell"]      = np.where(df["Close"] < df["hl"], 1, 0)
        df["sig_buy"]       = np.where(df["Close"] > df["hl"], 2, 0)

        df["sig_sellst"]    = np.where(
            (df["st"] < 90) & (df["prst"] > df["st"]) & (df["st"] > 50), 1, 0)
        df["sig_buyst"]     = np.where(
            (df["st"] > 10) & (df["prst"] < df["st"]) & (df["st"] < 50), 2, 0)

        df["sig_sellmacd"]  = np.where(df["macd_ind"] < df["prevmacd"], 1, 0)
        df["sig_buymacd"]   = np.where(df["macd_ind"] > df["prevmacd"], 2, 0)

        df["sellsig"]       = df["sig_sell"] + df["sig_sellst"] + df["sig_sellmacd"]
        df["buysig"]        = df["sig_buy"]  + df["sig_buyst"]  + df["sig_buymacd"]

        df["prbuysig"]      = df["buysig"].shift(1)
        df["prsellsig"]     = df["sellsig"].shift(1)

        # Final signals (only on transition)
        df["signal1"] = np.where((df["buysig"]  == 6) & (df["prbuysig"]  != 6), 1, 0)
        df["signal2"] = np.where((df["sellsig"] == 3) & (df["prsellsig"] != 3), 2, 0)

        # Price markers for chart
        df["signal1p"] = np.where(df["signal1"] == 1, df["Low"],  np.nan)
        df["signal2p"] = np.where(df["signal2"] == 2, df["High"], np.nan)

        # ── Last recent buy/sell dates ────────────────────────────────
        buy_rows  = df[df["signal1"] == 1]
        sell_rows = df[df["signal2"] == 2]

        last_buy_date  = buy_rows.index[-1]  if not buy_rows.empty  else None
        last_sell_date = sell_rows.index[-1] if not sell_rows.empty else None

        # ── Did a signal fire in the last 2 candles? ─────────────────
        tail          = df.tail(2)
        has_buy_now   = tail["signal1"].sum() > 0
        has_sell_now  = tail["signal2"].sum() > 0

        return {
            "symbol":         stock,
            "df":             df,
            "has_buy_now":    has_buy_now,
            "has_sell_now":   has_sell_now,
            "last_buy_date":  last_buy_date,
            "last_sell_date": last_sell_date,
        }

    except Exception as e:
        # Uncomment for debugging:
        # st.warning(f"{stock}: {e}")
        return None


# ─────────────────────────────────────────────
# ASYNC MAIN – gather all stocks
# ─────────────────────────────────────────────
async def run_screener(fromdate: str, todate: str, interval: int) -> list[dict]:
    connector = aiohttp.TCPConnector(limit=20)   # limit concurrent connections
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.ensure_future(fetch_and_process(session, s, fromdate, todate, interval))
            for s in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    # filter out None / failed
    return [r for r in results if r is not None]


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("📈 Keltner Channel Screener — NSE")

st.sidebar.header("⚙️ Settings")

dt_days    = st.sidebar.number_input("Recent days (for buy/sell table)", min_value=1, max_value=50,   value=2)
tframe     = st.sidebar.number_input("Timeframe (minutes)",              min_value=1, max_value=43800, value=15,
                                     help="1, 3, 5, 15, 60, 240, 1440, 10080, 43800")
dy_back    = st.sidebar.number_input("Days of history to fetch",         min_value=1, max_value=5000,  value=100,
                                     help="Use ~500 for 1-day candles")

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ Run Screener", type="primary")

if run_button:
    ed      = datetime.now()
    stdate  = ed - timedelta(days=dy_back)

    # Groww API needs ms epoch
    fromdate = to_unix_ms(stdate)
    todate   = to_unix_ms(ed)

    with st.spinner(f"Fetching {len(symbols)} symbols …"):
        results = asyncio.run(run_screener(fromdate, todate, int(tframe)))

    st.success(f"Done — processed {len(results)} symbols successfully.")

    # ── Separate buy / sell signals ──────────────────────────────────
    buy_records  = []
    sell_records = []
    buy_charts   = []
    sell_charts  = []

    bcdate = ed - timedelta(days=dt_days)

    for r in results:
        sym           = r["symbol"]
        last_buy_dt   = r["last_buy_date"]
        last_sell_dt  = r["last_sell_date"]

        if last_buy_dt:
            buy_records.append({"symbol": sym, "last_buy_date": last_buy_dt})
        if last_sell_dt:
            sell_records.append({"symbol": sym, "last_sell_date": last_sell_dt})

        if r["has_buy_now"]:
            buy_charts.append(r)
        if r["has_sell_now"]:
            sell_charts.append(r)

    # ── Summary tables ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Recent Buy Signals")
        if buy_records:
            d = pd.DataFrame(buy_records)
            d["last_buy_date"] = pd.to_datetime(d["last_buy_date"])
            st.dataframe(d[d["last_buy_date"] > bcdate].reset_index(drop=True),
                         use_container_width=True)
        else:
            st.info("No buy signals found.")

    with col2:
        st.subheader("🔴 Recent Sell Signals")
        if sell_records:
            d = pd.DataFrame(sell_records)
            d["last_sell_date"] = pd.to_datetime(d["last_sell_date"])
            st.dataframe(d[d["last_sell_date"] > bcdate].reset_index(drop=True),
                         use_container_width=True)
        else:
            st.info("No sell signals found.")

    # ── Charts: live buy signals ─────────────────────────────────────
    if buy_charts:
        st.markdown("---")
        st.subheader(f"📊 Charts — Active BUY ({len(buy_charts)} stocks)")
        for r in buy_charts:
            st.markdown(f"### 🟢 {r['symbol']} — BUY signal")
            st.plotly_chart(build_chart(r["df"], r["symbol"]), use_container_width=True)

    # ── Charts: live sell signals ────────────────────────────────────
    if sell_charts:
        st.markdown("---")
        st.subheader(f"📊 Charts — Active SELL ({len(sell_charts)} stocks)")
        for r in sell_charts:
            st.markdown(f"### 🔴 {r['symbol']} — SELL signal")
            st.plotly_chart(build_chart(r["df"], r["symbol"]), use_container_width=True)

    if not buy_charts and not sell_charts:
        st.info("No fresh buy or sell signals on the latest candles for any symbol.")

    # ── Export parquet ───────────────────────────────────────────────
    if buy_records:
        pd.DataFrame(buy_records).to_parquet("buy.parquet", index=False)
    if sell_records:
        pd.DataFrame(sell_records).to_parquet("sell.parquet", index=False)
