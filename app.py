from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# Fetch current S&P 500 tickers
def get_sp500_tickers():
    url = "[https://en.wikipedia.org/wiki/List_of_S%26P_500_companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

# Add Ichimoku components
def add_ichimoku(df):
    high9 = df['High'].rolling(9).max()
    low9 = df['Low'].rolling(9).min()
    df['Tenkan'] = (high9 + low9) / 2

    high26 = df['High'].rolling(26).max()
    low26 = df['Low'].rolling(26).min()
    df['Kijun'] = (high26 + low26) / 2

    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    df['Senkou_B'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    return df

# Add MACD
def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

# Check if close is above the cloud
def is_above_cloud(df):
    if len(df) < 100:
        return False
    df = add_ichimoku(df.copy())
    if pd.isna(df['Senkou_A'].iloc[-1]) or pd.isna(df['Senkou_B'].iloc[-1]):
        return False
    cloud_top = max(df['Senkou_A'].iloc[-1], df['Senkou_B'].iloc[-1])
    return df['Close'].iloc[-1] > cloud_top

# Main bullish check with multi-timeframe cloud + daily MACD cross
def is_bullish(ticker):
    try:
        # Daily first (most will fail here)
        daily_data = yf.download(ticker, period='3y', interval='1d', progress=False)
        if daily_data.empty or len(daily_data) < 100:
            return False

        above_daily = is_above_cloud(daily_data.copy())
        daily_data = add_macd(daily_data)
        if pd.isna(daily_data['MACD'].iloc[-1]):
            return False

        macd_cross = (daily_data['MACD'].iloc[-2] <= daily_data['MACD_Signal'].iloc[-2]) and \
                     (daily_data['MACD'].iloc[-1] > daily_data['MACD_Signal'].iloc[-1])

        if not (above_daily and macd_cross):
            return False

        # Only if daily passes → check lower timeframes
        hourly_data = yf.download(ticker, period='730d', interval='60m', progress=False)
        min_data = yf.download(ticker, period='7d', interval='1m', progress=False)

        if hourly_data.empty or min_data.empty:
            return False

        above_hourly = is_above_cloud(hourly_data)
        above_min = is_above_cloud(min_data)

        return above_hourly and above_min

    except Exception:
        return False

@app.route('/screen')
def screen():
    tickers = get_sp500_tickers()
    bullish_stocks = []

    # Parallel execution for speed
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_ticker = {executor.submit(is_bullish, ticker): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            if future.result():
                bullish_stocks.append(future_to_ticker[future])

    bullish_stocks.sort()

    count = len(bullish_stocks)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds') + 'Z'
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')

    report_title = f"Bullish Signals — {count} found ({date_str})"
    report_body = (
        f"Found {count} stocks meeting all criteria (close above Ichimoku cloud on 1m/1h/daily + daily MACD bullish cross):\n\n"
        + (", ".join(bullish_stocks) if bullish_stocks else "None today.")
    )
    bullish_stocks_str = ", ".join(bullish_stocks) if bullish_stocks else ""

    return jsonify({
        "timestamp": timestamp,
        "bullish_stocks": bullish_stocks,
        "bullish_stocks_str": bullish_stocks_str,
        "count": count,
        "total_scanned": len(tickers),
        "report_title": report_title,
        "report_body": report_body
    })

@app.route('/')
def home():
    return "Stock screener API is running. Call /screen for results."
