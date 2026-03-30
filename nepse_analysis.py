"""
============================================================
NEPSE STOCK ANALYSIS, TREND DETECTION, AND PREDICTION SYSTEM
WITH INVESTOR SENTIMENT FOR NEWLY LISTED COMPANIES
============================================================
Author  : Data Science Project
Exchange: Nepal Stock Exchange (NEPSE)
Language: Python (pandas, matplotlib, scikit-learn)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_FILE   = "nepse_data.csv"
OUTPUT_DIR  = "outputs"
COMPANIES   = ["NABIL", "NICA", "NLIC", "MNBBL", "CHCL",
               "SCB",   "EBL",  "SBI",  "UPPER", "NHPC"]
IPO_COMPANY = "MNBBL"          # newly listed company
IPO_DATE    = "2023-01-16"     # IPO listing date

# Sector mapping (NEPSE sector classifications)
SECTOR_MAP = {
    "NABIL":  "Commercial Banks",
    "NICA":   "Commercial Banks",
    "SCB":    "Commercial Banks",
    "EBL":    "Commercial Banks",
    "SBI":    "Commercial Banks",
    "NLIC":   "Life Insurance",
    "MNBBL":  "Microfinance",
    "CHCL":   "Hydropower",
    "UPPER":  "Hydropower",
    "NHPC":   "Hydropower",
}

# Chart style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
})

COLORS = {
    "NABIL":  "#58a6ff",
    "NICA":   "#3fb950",
    "NLIC":   "#d2a8ff",
    "MNBBL":  "#ffa657",
    "CHCL":   "#ff7b72",
    "SCB":    "#79c0ff",
    "EBL":    "#56d364",
    "SBI":    "#bc8cff",
    "UPPER":  "#f78166",
    "NHPC":   "#ffb77c",
    "bull":   "#3fb950",
    "bear":   "#f85149",
    "neutral":"#8b949e",
    "ma7":    "#ffa657",
    "ma20":   "#d2a8ff",
    "ma50":   "#ff7b72",
}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  NEPSE STOCK ANALYSIS SYSTEM — Loading Data")
print("═"*60)

df = pd.read_csv(DATA_FILE)

# Convert date and sort
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['Company', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Handle missing values
numeric_cols = ['Closing_Price', 'Volume', 'Open_Price', 'High_Price', 'Low_Price']
df[numeric_cols] = df[numeric_cols].ffill().bfill()
df.dropna(subset=['Date', 'Company', 'Closing_Price'], inplace=True)

# Fix #2: Remove zero-price rows (caused by failed API fetches)
zero_rows = (df['Closing_Price'] == 0).sum()
if zero_rows > 0:
    print(f"⚠️  WARNING: Dropping {zero_rows} rows with zero closing price (bad API data)")
df = df[df['Closing_Price'] > 0].reset_index(drop=True)

print(f"✔  Loaded {len(df)} rows | {df['Company'].nunique()} companies")
print(f"   Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"   Companies: {', '.join(df['Company'].unique())}\n")

# Fix #8: Data freshness check
days_old = (pd.Timestamp.today() - df['Date'].max()).days
if days_old > 7:
    print(f"⚠️  WARNING: Data is {days_old} days old! Run fetch_nepse_data.py to update.\n")

# ══════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def compute_features(grp):
    grp = grp.copy().sort_values('Date')
    grp['Daily_Return']  = grp['Closing_Price'].pct_change() * 100
    grp['MA7']           = grp['Closing_Price'].rolling(7,  min_periods=1).mean()
    grp['MA20']          = grp['Closing_Price'].rolling(20, min_periods=1).mean()
    grp['MA50']          = grp['Closing_Price'].rolling(50, min_periods=1).mean()
    grp['Volatility']    = grp['Daily_Return'].rolling(7, min_periods=1).std()
    grp['Vol_MA7']       = grp['Volume'].rolling(7, min_periods=1).mean()
    grp['Vol_Spike']     = grp['Volume'] / grp['Vol_MA7'].replace(0, 1)
    grp['Price_Change']  = grp['Closing_Price'].diff()
    grp['Momentum']      = grp['Closing_Price'] - grp['Closing_Price'].shift(5)

    # ── RSI (14-period) ───────────────────────────────────
    delta  = grp['Closing_Price'].diff()
    gain   = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss   = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs     = gain / loss.replace(0, 1e-9)
    grp['RSI'] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ─────────────────────────────────
    ema12          = grp['Closing_Price'].ewm(span=12, adjust=False).mean()
    ema26          = grp['Closing_Price'].ewm(span=26, adjust=False).mean()
    grp['MACD']        = ema12 - ema26
    grp['MACD_Signal'] = grp['MACD'].ewm(span=9, adjust=False).mean()
    grp['MACD_Hist']   = grp['MACD'] - grp['MACD_Signal']

    # ── Bollinger Bands (20-period, 2σ) ──────────────────
    bb_mid            = grp['Closing_Price'].rolling(20, min_periods=1).mean()
    bb_std            = grp['Closing_Price'].rolling(20, min_periods=1).std().fillna(0)
    grp['BB_Mid']     = bb_mid
    grp['BB_Upper']   = bb_mid + 2 * bb_std
    grp['BB_Lower']   = bb_mid - 2 * bb_std
    grp['BB_Width']   = (grp['BB_Upper'] - grp['BB_Lower']) / bb_mid.replace(0, 1)
    # %B: where price sits within the bands (0=lower band, 1=upper band)
    grp['BB_PctB']    = (grp['Closing_Price'] - grp['BB_Lower']) / \
                        (grp['BB_Upper'] - grp['BB_Lower']).replace(0, 1)

    return grp

dfs = []
for comp in df['Company'].unique():
    sub = df[df['Company'] == comp].copy()
    sub = compute_features(sub)
    sub['Company'] = comp
    dfs.append(sub)
df = pd.concat(dfs, ignore_index=True)

# ══════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
print("═"*60)
print("  EXPLORATORY DATA ANALYSIS")
print("═"*60)

eda_rows = []
for comp in COMPANIES:
    sub = df[df['Company'] == comp]
    eda_rows.append({
        "Company":     comp,
        "Avg Price":   round(sub['Closing_Price'].mean(), 2),
        "High":        sub['High_Price'].max(),
        "Low":         sub['Low_Price'].min(),
        "Avg Volume":  int(sub['Volume'].mean()),
        "Avg Return%": round(sub['Daily_Return'].mean(), 3),
        "Volatility":  round(sub['Volatility'].mean(), 3),
        "Sector":      SECTOR_MAP.get(comp, "Other"),
    })

eda_df = pd.DataFrame(eda_rows)
print(eda_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 4. TREND DETECTION
# ══════════════════════════════════════════════════════════════
def detect_trend(sub):
    """Detect trend using MA crossover, momentum, RSI, and MACD."""
    latest = sub.iloc[-1]
    recent = sub.tail(10)

    # MA signal
    if latest['MA7'] > latest['MA20']:
        ma_sig = "Uptrend"
    elif latest['MA7'] < latest['MA20']:
        ma_sig = "Downtrend"
    else:
        ma_sig = "Sideways"

    # Momentum
    avg_mom = recent['Momentum'].mean()
    if avg_mom > 0:
        mom_sig = "Bullish"
    elif avg_mom < 0:
        mom_sig = "Bearish"
    else:
        mom_sig = "Neutral"

    # RSI signal
    rsi = latest['RSI']
    if rsi >= 70:
        rsi_sig = "Overbought"
    elif rsi <= 30:
        rsi_sig = "Oversold"
    else:
        rsi_sig = "Neutral"

    # MACD signal
    if latest['MACD'] > latest['MACD_Signal']:
        macd_sig = "Bullish"
    elif latest['MACD'] < latest['MACD_Signal']:
        macd_sig = "Bearish"
    else:
        macd_sig = "Neutral"

    # Support / Resistance (rolling 20-day)
    support    = round(sub['Low_Price'].tail(20).min(), 2)
    resistance = round(sub['High_Price'].tail(20).max(), 2)

    return ma_sig, mom_sig, support, resistance, rsi_sig, macd_sig, round(rsi, 1)

print("\n" + "═"*60)
print("  TREND DETECTION")
print("═"*60)
trend_data = {}
for comp in COMPANIES:
    sub = df[df['Company'] == comp]
    if sub.empty:
        continue
    ma_sig, mom_sig, sup, res, rsi_sig, macd_sig, rsi_val = detect_trend(sub)
    trend_data[comp] = (ma_sig, mom_sig, sup, res, rsi_sig, macd_sig, rsi_val)
    print(f"  {comp:6s} | {ma_sig:10s} | Mom: {mom_sig:8s} | RSI: {rsi_val:5.1f} ({rsi_sig:10s}) | MACD: {macd_sig:8s} | S:{sup:.0f} R:{res:.0f}")

# ══════════════════════════════════════════════════════════════
# 5. INVESTOR SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
def classify_sentiment(sub):
    """Rule-based sentiment classifier."""
    latest     = sub.iloc[-1]
    prev       = sub.iloc[-2] if len(sub) > 1 else latest
    recent5    = sub.tail(5)

    price_up   = latest['Closing_Price'] > prev['Closing_Price']
    vol_spike  = latest['Vol_Spike'] > 1.5
    price_drop = latest['Closing_Price'] < prev['Closing_Price']
    cont_rise  = (recent5['Daily_Return'] > 0).sum() >= 4
    cont_drop  = (recent5['Daily_Return'] < 0).sum() >= 4

    if vol_spike and price_up:
        return "Bullish", "High demand (volume spike + price rise)"
    elif price_drop and vol_spike:
        return "Bearish", "Selling pressure (volume spike + price fall)"
    elif cont_rise:
        return "Bullish", "Continuous upward trend — buying interest"
    elif cont_drop:
        return "Bearish", "Continuous downward trend — panic selling"
    # Fix #6: Softer fallback — use average return over last 5 days
    elif recent5['Daily_Return'].mean() > 0.5:
        return "Bullish", "Mild positive bias — slight buying interest"
    elif recent5['Daily_Return'].mean() < -0.5:
        return "Bearish", "Mild negative bias — slight selling pressure"
    else:
        return "Neutral", "Stable / consolidating market"

print("\n" + "═"*60)
print("  INVESTOR SENTIMENT ANALYSIS")
print("═"*60)
sentiment_data = {}
for comp in COMPANIES:
    sub = df[df['Company'] == comp]
    if sub.empty:
        continue
    sent, reason = classify_sentiment(sub)
    sentiment_data[comp] = sent
    icon = "🟢" if sent == "Bullish" else ("🔴" if sent == "Bearish" else "🟡")
    print(f"  {comp:6s} | {icon} {sent:8s} | {reason}")

# ══════════════════════════════════════════════════════════════
# 6. IPO / NEWLY LISTED COMPANY ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print(f"  IPO ANALYSIS — {IPO_COMPANY} (Listed: {IPO_DATE})")
print("═"*60)

ipo_sub = df[(df['Company'] == IPO_COMPANY) & (df['Date'] >= IPO_DATE)].copy()
ipo_sub['Day'] = range(1, len(ipo_sub) + 1)

peak_price = ipo_sub['Closing_Price'].max()
peak_day   = ipo_sub.loc[ipo_sub['Closing_Price'].idxmax(), 'Day']
ipo_open   = ipo_sub['Closing_Price'].iloc[0]
latest_p   = ipo_sub['Closing_Price'].iloc[-1]
from_ipo   = round((latest_p - ipo_open) / ipo_open * 100, 2)
from_peak  = round((latest_p - peak_price) / peak_price * 100, 2)

# Identify pattern
upper_circuit = (ipo_sub['Daily_Return'].head(7) >= 10).sum()  # Fix #5: NEPSE limit is 10%, not 9%
sudden_fall   = (ipo_sub['Daily_Return'] < -10).sum()

print(f"  IPO Price (Day 1 Close) : NPR {ipo_open}")
print(f"  Peak Price              : NPR {peak_price}  (Day {peak_day})")
print(f"  Current Price           : NPR {latest_p}")
print(f"  Return from IPO         : {from_ipo:+.2f}%")
print(f"  From Peak               : {from_peak:+.2f}%")
print(f"  Upper Circuit Days      : {upper_circuit}")
print(f"  Sudden Fall Events      : {sudden_fall}")

if upper_circuit >= 3:
    print(f"\n  📈 Pattern: UPPER CIRCUIT TREND — Strong bullish momentum post-IPO")
if sudden_fall >= 1:
    print(f"  📉 Pattern: PROFIT BOOKING DETECTED — Selling pressure after peak")

# IPO investor action prediction
if from_ipo > 50 and from_peak < -15:
    ipo_action = "SELL — Over-valued; profit booking recommended"
elif from_ipo > 0 and from_peak > -10:
    ipo_action = "HOLD — Still performing well from IPO price"
else:
    ipo_action = "BUY — Price corrected; possible entry opportunity"

print(f"\n  💡 Predicted Investor Action: {ipo_action}")

# ══════════════════════════════════════════════════════════════
# 7. PREDICTION MODULE (Linear Regression)
# ══════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression

def predict_prices(sub, days_ahead=7):
    # Fix #3: Train only on last 60 days so slope reflects recent trend, not full history
    sub = sub.tail(60).reset_index(drop=True)
    sub['Day_Num'] = np.arange(len(sub))
    X = sub[['Day_Num']].values
    y = sub['Closing_Price'].values

    model = LinearRegression()
    model.fit(X, y)
    sub['Predicted'] = model.predict(X)

    future_days = np.arange(len(sub), len(sub) + days_ahead).reshape(-1, 1)
    future_pred = model.predict(future_days)
    future_dates = pd.date_range(start=sub['Date'].max() + pd.Timedelta(days=1),
                                  periods=days_ahead, freq='B')

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': np.round(future_pred, 2)})
    return sub, future_df, model.coef_[0]

print("\n" + "═"*60)
print("  PRICE PREDICTION (Next 7 Trading Days)")
print("═"*60)

prediction_store = {}
for comp in COMPANIES:
    sub = df[df['Company'] == comp].copy()
    if sub.empty:
        continue
    sub_pred, future_df, slope = predict_prices(sub)
    prediction_store[comp] = (sub_pred, future_df, slope)  # Fix #4: store slope too

    direction = "↑ Rising" if slope > 0 else "↓ Falling"
    print(f"\n  {comp} ({direction}, slope={slope:+.2f})")
    print(future_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 8. DECISION SUPPORT SYSTEM
# ══════════════════════════════════════════════════════════════
def decide(trend, sentiment, slope, momentum="Neutral", rsi_sig="Neutral", macd_sig="Neutral"):
    """
    Score-based decision using 6 signals:
    MA trend, sentiment, regression slope, momentum, RSI, MACD.
    """
    bull_score = 0
    bear_score = 0

    if trend      == "Uptrend":    bull_score += 1
    else:                          bear_score += 1

    if sentiment  == "Bullish":    bull_score += 1
    elif sentiment == "Bearish":   bear_score += 1

    if slope > 0:                  bull_score += 1
    else:                          bear_score += 1

    if momentum   == "Bullish":    bull_score += 1
    elif momentum == "Bearish":    bear_score += 1

    if macd_sig   == "Bullish":    bull_score += 1
    elif macd_sig == "Bearish":    bear_score += 1

    # RSI: oversold = buy opportunity, overbought = caution
    if rsi_sig    == "Oversold":   bull_score += 1
    elif rsi_sig  == "Overbought": bear_score += 1

    net = bull_score - bear_score
    if net >= 3:    return "🟢 BUY"
    elif net <= -2: return "🔴 SELL"
    else:           return "🟡 HOLD"

print("\n" + "═"*60)
print("  DECISION SUPPORT SYSTEM — Buy / Hold / Sell")
print("═"*60)
print(f"  {'Company':8s} | {'Trend':10s} | {'Sentiment':8s} | {'Prediction':10s} | Decision")
print("  " + "─"*60)

decision_data = {}
for comp in COMPANIES:
    if comp not in trend_data:
        continue
    ma_sig, mom_sig, sup, res, rsi_sig, macd_sig, rsi_val = trend_data[comp]
    sent = sentiment_data[comp]
    # Fix #4: Reuse cached predictions instead of re-running predict_prices()
    sub_pred, future_df, slope = prediction_store[comp]
    dec = decide(ma_sig, sent, slope, momentum=mom_sig, rsi_sig=rsi_sig, macd_sig=macd_sig)
    decision_data[comp] = dec
    pred_dir = "↑" if slope > 0 else "↓"
    print(f"  {comp:6s} | {ma_sig:10s} | {sent:8s} | {pred_dir} {future_df['Predicted_Price'].iloc[-1]:<8.0f} | RSI:{rsi_val:5.1f} | MACD:{macd_sig:8s} | {dec}")

# ══════════════════════════════════════════════════════════════
# 9. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  GENERATING CHARTS...")
print("═"*60)

# ── Chart 1: Multi-company price overview ──────────────────
fig, axes = plt.subplots(len(COMPANIES), 1, figsize=(16, 4 * len(COMPANIES)), sharex=False)
fig.suptitle("NEPSE — Stock Price Overview with Moving Averages", fontsize=16, y=0.98, color="#e6edf3")

for i, comp in enumerate(COMPANIES):
    ax = axes[i]
    sub = df[df['Company'] == comp]
    col = COLORS[comp]

    ax.plot(sub['Date'], sub['Closing_Price'], color=col,       lw=1.8, label="Close")
    ax.plot(sub['Date'], sub['MA7'],           color=COLORS['ma7'],  lw=1.2, ls='--', label="MA7")
    ax.plot(sub['Date'], sub['MA20'],          color=COLORS['ma20'], lw=1.2, ls='-.',  label="MA20")
    ax.fill_between(sub['Date'], sub['MA7'], sub['MA20'],
                    where=sub['MA7'] >= sub['MA20'], alpha=0.15, color=COLORS['bull'])
    ax.fill_between(sub['Date'], sub['MA7'], sub['MA20'],
                    where=sub['MA7'] <  sub['MA20'], alpha=0.15, color=COLORS['bear'])

    sent = sentiment_data[comp]
    dec  = decision_data[comp]
    sc   = COLORS['bull'] if sent == 'Bullish' else (COLORS['bear'] if sent == 'Bearish' else COLORS['neutral'])
    ax.set_title(f"{comp}  |  Sentiment: {sent}  |  {dec}", color=col, fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("NPR", fontsize=9)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/01_price_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 1: Price Overview saved")

# ── Chart 2: Volume vs Price ───────────────────────────────
fig, axes = plt.subplots(len(COMPANIES), 2, figsize=(18, 4 * len(COMPANIES)))
fig.suptitle("NEPSE — Volume vs Price Analysis", fontsize=16, y=0.99)

for i, comp in enumerate(COMPANIES):
    sub = df[df['Company'] == comp]
    col = COLORS[comp]

    # Price
    ax1 = axes[i][0]
    ax1.plot(sub['Date'], sub['Closing_Price'], color=col, lw=1.8)
    ax1.set_title(f"{comp} — Closing Price", fontsize=11, color=col)
    ax1.set_ylabel("NPR"); ax1.grid(True, alpha=0.3)
    ax1.spines[['top','right']].set_visible(False)

    # Volume bars coloured by daily return
    ax2 = axes[i][1]
    colours = [COLORS['bull'] if r >= 0 else COLORS['bear'] for r in sub['Daily_Return'].fillna(0)]
    ax2.bar(sub['Date'], sub['Volume'], color=colours, width=0.8, alpha=0.8)
    ax2.plot(sub['Date'], sub['Vol_MA7'], color=COLORS['ma7'], lw=1.5, label="Vol MA7")
    ax2.set_title(f"{comp} — Volume (green=up, red=down)", fontsize=11, color=col)
    ax2.set_ylabel("Volume"); ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_volume_price.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 2: Volume vs Price saved")

# ── Chart 3: Daily Returns & Volatility ───────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("NEPSE — Daily Returns & Volatility Comparison", fontsize=16)

for comp in COMPANIES:
    sub = df[df['Company'] == comp]
    axes[0].plot(sub['Date'], sub['Daily_Return'], color=COLORS[comp], lw=1.2, alpha=0.8, label=comp)
    axes[1].plot(sub['Date'], sub['Volatility'],   color=COLORS[comp], lw=1.2, alpha=0.8, label=comp)

axes[0].axhline(0, color='#8b949e', lw=0.8, ls='--')
axes[0].set_title("Daily Return (%)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].spines[['top','right']].set_visible(False)

axes[1].set_title("7-Day Rolling Volatility (%)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_returns_volatility.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 3: Returns & Volatility saved")

# ── Chart 4: Trend Detection Highlights ───────────────────
fig, axes = plt.subplots(1, len(COMPANIES), figsize=(20, 6))
fig.suptitle("NEPSE — Trend Detection (MA7 vs MA20 Crossover)", fontsize=15)

for i, comp in enumerate(COMPANIES):
    ax  = axes[i]
    sub = df[df['Company'] == comp]
    col = COLORS[comp]

    ax.plot(sub['Date'], sub['Closing_Price'], color=col,        lw=1.5, label='Close')
    ax.plot(sub['Date'], sub['MA7'],           color=COLORS['ma7'],  lw=1.0, ls='--', label='MA7')
    ax.plot(sub['Date'], sub['MA20'],          color=COLORS['ma20'], lw=1.0, ls='-.', label='MA20')

    # Shade trend regions
    ax.fill_between(sub['Date'], sub['Closing_Price'].min() * 0.95,
                    sub['Closing_Price'].max() * 1.05,
                    where=sub['MA7'] >= sub['MA20'], alpha=0.1, color=COLORS['bull'], label='Uptrend')
    ax.fill_between(sub['Date'], sub['Closing_Price'].min() * 0.95,
                    sub['Closing_Price'].max() * 1.05,
                    where=sub['MA7'] <  sub['MA20'], alpha=0.1, color=COLORS['bear'], label='Downtrend')

    # Support / resistance
    sup = sub['Low_Price'].tail(20).min()
    res = sub['High_Price'].tail(20).max()
    ax.axhline(sup, color='#3fb950', lw=0.8, ls=':', alpha=0.7)
    ax.axhline(res, color='#f85149', lw=0.8, ls=':', alpha=0.7)
    ax.text(sub['Date'].iloc[-1], sup, f'S:{sup:.0f}', fontsize=7, color='#3fb950', ha='right')
    ax.text(sub['Date'].iloc[-1], res, f'R:{res:.0f}', fontsize=7, color='#f85149', ha='right')

    trend_label = trend_data[comp][0]
    tc = COLORS['bull'] if 'Up' in trend_label else (COLORS['bear'] if 'Down' in trend_label else COLORS['neutral'])
    ax.set_title(f"{comp}\n{trend_label}", color=col, fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_trend_detection.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 4: Trend Detection saved")

# ── Chart 5: IPO Analysis ─────────────────────────────────
ipo_sub = df[(df['Company'] == IPO_COMPANY) & (df['Date'] >= IPO_DATE)].copy()
ipo_sub['Day'] = range(1, len(ipo_sub) + 1)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle(f"IPO Analysis — {IPO_COMPANY} (Nepal Stock Exchange)", fontsize=16)

ax = axes[0]
ax.plot(ipo_sub['Day'], ipo_sub['Closing_Price'], color=COLORS['MNBBL'], lw=2, marker='o', ms=4, label='Close')
ax.axhline(ipo_open,   color='#8b949e', lw=1, ls='--', label=f'IPO Price: {ipo_open}')
ax.axhline(peak_price, color=COLORS['bull'],  lw=1, ls='--', label=f'Peak: {peak_price}')
ax.axvline(peak_day,   color=COLORS['bear'],  lw=1, ls=':',  label=f'Peak Day {peak_day}')
ax.fill_between(ipo_sub['Day'], ipo_open, ipo_sub['Closing_Price'],
                where=ipo_sub['Closing_Price'] >= ipo_open, alpha=0.2, color=COLORS['bull'])
ax.fill_between(ipo_sub['Day'], ipo_open, ipo_sub['Closing_Price'],
                where=ipo_sub['Closing_Price'] <  ipo_open, alpha=0.2, color=COLORS['bear'])

# Annotate peak
ax.annotate(f'Peak\nNPR {peak_price}',
            xy=(peak_day, peak_price),
            xytext=(peak_day + 2, peak_price + 20),
            arrowprops=dict(arrowstyle='->', color='white'),
            color='white', fontsize=9)

ax.set_title(f"{IPO_COMPANY} — Price Journey Post-IPO", color=COLORS['MNBBL'])
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlabel("Trading Day"); ax.set_ylabel("NPR")
ax.spines[['top','right']].set_visible(False)

# Volume
ax2 = axes[1]
vol_colors = [COLORS['bull'] if r >= 0 else COLORS['bear']
              for r in ipo_sub['Daily_Return'].fillna(0)]
ax2.bar(ipo_sub['Day'], ipo_sub['Volume'], color=vol_colors, alpha=0.8)
ax2.axvline(peak_day, color=COLORS['bear'], lw=1, ls=':')
ax2.set_title(f"{IPO_COMPANY} — Volume Post-IPO (Investor Activity)", color=COLORS['MNBBL'])
ax2.set_xlabel("Trading Day"); ax2.set_ylabel("Volume")
ax2.grid(True, alpha=0.3)
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_ipo_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 5: IPO Analysis saved")

# ── Chart 6: Prediction Charts ────────────────────────────
fig, axes = plt.subplots(len(COMPANIES), 1, figsize=(16, 4.5 * len(COMPANIES)))
fig.suptitle("NEPSE — Price Prediction (Linear Regression)", fontsize=16, y=0.99)

for i, comp in enumerate(COMPANIES):
    ax = axes[i]
    col = COLORS[comp]
    # Fix #4: Reuse cached predictions — no third call to predict_prices()
    sub_pred, future_df, slope = prediction_store[comp]

    ax.plot(sub_pred['Date'], sub_pred['Closing_Price'], color=col,         lw=1.8, label='Actual')
    ax.plot(sub_pred['Date'], sub_pred['Predicted'],     color='#8b949e',   lw=1.2, ls='--', label='Fitted')
    ax.plot(future_df['Date'], future_df['Predicted_Price'],
            color=COLORS['ma7'], lw=1.8, ls='--', marker='o', ms=5, label='Forecast')

    # Shade forecast zone
    ax.axvspan(sub_pred['Date'].max(), future_df['Date'].max(),
               alpha=0.1, color=COLORS['bull'] if slope > 0 else COLORS['bear'])

    last_actual = sub_pred['Closing_Price'].iloc[-1]
    last_pred   = future_df['Predicted_Price'].iloc[-1]
    change_pct  = (last_pred - last_actual) / last_actual * 100

    ax.set_title(f"{comp}  |  7-Day Forecast: NPR {last_pred:.0f}  ({change_pct:+.2f}%)  |  {decision_data[comp]}",
                 color=col, fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylabel("NPR")
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUTPUT_DIR}/06_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 6: Predictions saved")

# ── Chart 7: Sentiment Dashboard ─────────────────────────
fig = plt.figure(figsize=(16, 8))
fig.suptitle("NEPSE — Investor Sentiment & Decision Dashboard", fontsize=17)
gs = gridspec.GridSpec(2, len(COMPANIES), hspace=0.5, wspace=0.35)

for i, comp in enumerate(COMPANIES):
    sent = sentiment_data[comp]
    dec  = decision_data[comp]
    ma_sig = trend_data[comp][0]
    col  = COLORS[comp]
    sc   = COLORS['bull'] if sent == 'Bullish' else (COLORS['bear'] if sent == 'Bearish' else COLORS['neutral'])

    # Sentiment meter (top row)
    ax_top = fig.add_subplot(gs[0, i])
    wedge_val = 1 if sent == 'Bullish' else (0 if sent == 'Bearish' else 0.5)
    wedge_clrs = [COLORS['bear'], COLORS['neutral'], COLORS['bull']]
    ax_top.pie([1, 1, 1], colors=wedge_clrs, startangle=180, counterclock=False,
               wedgeprops=dict(width=0.4))
    gauge_angle = 180 - (wedge_val * 180)
    rad = np.deg2rad(gauge_angle)
    ax_top.annotate('', xy=(0.38 * np.cos(rad), 0.38 * np.sin(rad)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax_top.set_title(f"{comp}\n{sent}", color=col, fontsize=11)
    ax_top.axis('equal')

    # Decision card (bottom row)
    ax_bot = fig.add_subplot(gs[1, i])
    ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1)
    ax_bot.set_facecolor('#0d1117')
    ax_bot.axis('off')
    dec_color = COLORS['bull'] if 'BUY' in dec else (COLORS['bear'] if 'SELL' in dec else '#ffa657')
    ax_bot.add_patch(mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                     boxstyle="round,pad=0.05", facecolor=dec_color, alpha=0.2,
                     edgecolor=dec_color, linewidth=2))
    ax_bot.text(0.5, 0.72, dec,       ha='center', va='center', fontsize=13, color=dec_color, fontweight='bold')
    ax_bot.text(0.5, 0.50, ma_sig,    ha='center', va='center', fontsize=10, color='#8b949e')
    ax_bot.text(0.5, 0.30, f"NPR {df[df['Company']==comp]['Closing_Price'].iloc[-1]:.0f}",
                ha='center', va='center', fontsize=11, color=col)

plt.savefig(f"{OUTPUT_DIR}/07_sentiment_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 7: Sentiment Dashboard saved")

# ── Chart 8: Correlation Heatmap ─────────────────────────
pivot = df.pivot_table(index='Date', columns='Company', values='Closing_Price')
corr  = pivot.corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("NEPSE — Stock Price Correlation Matrix", fontsize=15)
import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr.index)));   ax.set_yticklabels(corr.index)
for r in range(len(corr.index)):
    for c in range(len(corr.columns)):
        ax.text(c, r, f"{corr.values[r, c]:.2f}", ha='center', va='center',
                fontsize=10, color='black', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 8: Correlation Heatmap saved")

# ── Chart 9: RSI / MACD / Bollinger Bands ────────────
active_comps = [c for c in COMPANIES if c in trend_data]
n = len(active_comps)
fig, axes = plt.subplots(n, 3, figsize=(22, 4.5 * n))
if n == 1:
    axes = [axes]
fig.suptitle("NEPSE — RSI  |  MACD  |  Bollinger Bands", fontsize=16, y=0.99)

for i, comp in enumerate(active_comps):
    sub = df[df['Company'] == comp].copy()
    col = COLORS.get(comp, "#8b949e")
    rsi_val = trend_data[comp][6]

    # ── RSI panel ──────────────────────────────────────
    ax_rsi = axes[i][0]
    ax_rsi.plot(sub['Date'], sub['RSI'], color=col, lw=1.5)
    ax_rsi.axhline(70, color=COLORS['bear'],    lw=1, ls='--', label='Overbought (70)')
    ax_rsi.axhline(30, color=COLORS['bull'],    lw=1, ls='--', label='Oversold (30)')
    ax_rsi.axhline(50, color=COLORS['neutral'], lw=0.8, ls=':')
    ax_rsi.fill_between(sub['Date'], 70, sub['RSI'],
                        where=sub['RSI'] >= 70, alpha=0.2, color=COLORS['bear'])
    ax_rsi.fill_between(sub['Date'], sub['RSI'], 30,
                        where=sub['RSI'] <= 30, alpha=0.2, color=COLORS['bull'])
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title(f"{comp} — RSI  (current: {rsi_val:.1f})", color=col, fontsize=11)
    ax_rsi.legend(fontsize=7, loc='upper left')
    ax_rsi.grid(True, alpha=0.3)
    ax_rsi.spines[['top', 'right']].set_visible(False)
    ax_rsi.tick_params(axis='x', rotation=30)

    # ── MACD panel ─────────────────────────────────────
    ax_macd = axes[i][1]
    ax_macd.plot(sub['Date'], sub['MACD'],        color=col,             lw=1.5, label='MACD')
    ax_macd.plot(sub['Date'], sub['MACD_Signal'], color=COLORS['ma20'],  lw=1.2, ls='--', label='Signal')
    hist_colors = [COLORS['bull'] if v >= 0 else COLORS['bear'] for v in sub['MACD_Hist']]
    ax_macd.bar(sub['Date'], sub['MACD_Hist'], color=hist_colors, alpha=0.6, width=0.8, label='Histogram')
    ax_macd.axhline(0, color=COLORS['neutral'], lw=0.8, ls=':')
    ax_macd.set_title(f"{comp} — MACD (12,26,9)", color=col, fontsize=11)
    ax_macd.legend(fontsize=7, loc='upper left')
    ax_macd.grid(True, alpha=0.3)
    ax_macd.spines[['top', 'right']].set_visible(False)
    ax_macd.tick_params(axis='x', rotation=30)

    # ── Bollinger Bands panel ──────────────────────────
    ax_bb = axes[i][2]
    ax_bb.plot(sub['Date'], sub['Closing_Price'], color=col,             lw=1.8, label='Close')
    ax_bb.plot(sub['Date'], sub['BB_Upper'],      color=COLORS['bear'],  lw=1,   ls='--', label='Upper Band')
    ax_bb.plot(sub['Date'], sub['BB_Mid'],        color=COLORS['neutral'], lw=1, ls='-.',  label='Mid (MA20)')
    ax_bb.plot(sub['Date'], sub['BB_Lower'],      color=COLORS['bull'],  lw=1,   ls='--', label='Lower Band')
    ax_bb.fill_between(sub['Date'], sub['BB_Lower'], sub['BB_Upper'], alpha=0.08, color=col)
    ax_bb.set_title(f"{comp} — Bollinger Bands (20, 2σ)", color=col, fontsize=11)
    ax_bb.legend(fontsize=7, loc='upper left')
    ax_bb.grid(True, alpha=0.3)
    ax_bb.set_ylabel("NPR")
    ax_bb.spines[['top', 'right']].set_visible(False)
    ax_bb.tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUTPUT_DIR}/09_rsi_macd_bollinger.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✔  Chart 9: RSI / MACD / Bollinger Bands saved")

# ══════════════════════════════════════════════════════════════
# 10. FINAL SUMMARY REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  FINAL SUMMARY REPORT")
print("═"*60)
print(f"""
  ┌──────────┬───────────┬──────────┬───────┬────────┬───────────────────┐
  │ Company  │ Trend     │ Sentiment│ RSI   │ MACD   │ Decision          │
  ├──────────┼───────────┼──────────┼───────┼────────┼───────────────────┤""")
for comp in active_comps:
    tr   = trend_data[comp][0]
    st   = sentiment_data.get(comp, "N/A")
    rsi  = trend_data[comp][6]
    mcd  = trend_data[comp][5]
    de   = decision_data.get(comp, "N/A")
    print(f"  │ {comp:<8s} │ {tr:<9s} │ {st:<8s} │ {rsi:5.1f} │ {mcd:<6s} │ {de:<17s} │")
print("  └──────────┴───────────┴──────────┴───────┴────────┴───────────────────┘")

print(f"""
  IPO ANALYSIS ({IPO_COMPANY}):
  • Listed at NPR {ipo_open} → Peaked at NPR {peak_price} on Day {peak_day}
  • Upper circuit streak: {upper_circuit} days
  • Investor recommendation: {ipo_action}

  OUTPUT FILES:
  • outputs/01_price_overview.png
  • outputs/02_volume_price.png
  • outputs/03_returns_volatility.png
  • outputs/04_trend_detection.png
  • outputs/05_ipo_analysis.png
  • outputs/06_predictions.png
  • outputs/07_sentiment_dashboard.png
  • outputs/08_correlation_heatmap.png
  • outputs/09_rsi_macd_bollinger.png   ← NEW

  NOTE: Run fetch_nepse_data.py daily after market close (3 PM NPT)
  to keep nepse_data.csv current. Supports 10 stocks across 3 sectors.
""")
print("═"*60)
print("  ✅  Analysis Complete!")
print("═"*60)

