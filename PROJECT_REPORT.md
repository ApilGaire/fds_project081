# NEPSE Stock Analysis, Trend Detection, and Prediction System
## with Investor Sentiment for Newly Listed Companies

**Nepal Stock Exchange (NEPSE) | Data Science Project**
**Language:** Python 3 | pandas · matplotlib · scikit-learn
**Date:** February 2026

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Data Sources & Structure](#2-data-sources--structure)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Visualization Charts](#5-visualization-charts)
6. [Trend Detection](#6-trend-detection)
7. [Prediction Module](#7-prediction-module)
8. [Investor Sentiment Analysis](#8-investor-sentiment-analysis)
9. [IPO / Newly Listed Company Analysis](#9-ipo--newly-listed-company-analysis)
10. [Decision Support System](#10-decision-support-system)
11. [Results & Summary](#11-results--summary)
12. [Conclusion & Future Improvements](#12-conclusion--future-improvements)

---

## 1. Project Overview

This project builds a **comprehensive stock analysis system** for the Nepal Stock Exchange (NEPSE). It covers five major companies across three sectors — Banking, Insurance, and Hydropower — and includes a **special focus on MNBBL**, a newly listed IPO company.

### Companies Analyzed
| Symbol | Company | Sector |
|--------|---------|--------|
| NABIL | Nabil Bank Limited | Banking |
| NICA | NIC Asia Bank | Banking |
| NLIC | Nepal Life Insurance Company | Insurance |
| MNBBL | Muktinath Bikas Bank (IPO) | Banking |
| CHCL | Chilime Hydropower Company | Hydropower |

### System Components
```
┌─────────────────────────────────────────────────────────┐
│            NEPSE ANALYSIS SYSTEM ARCHITECTURE           │
├──────────────────┬──────────────────────────────────────┤
│ Data Layer       │ CSV Import → Preprocessing → Features│
├──────────────────┼──────────────────────────────────────┤
│ Analysis Layer   │ EDA → Trend Detection → Volatility   │
├──────────────────┼──────────────────────────────────────┤
│ Prediction Layer │ Linear Regression → 7-Day Forecast   │
├──────────────────┼──────────────────────────────────────┤
│ Sentiment Layer  │ Rule-based → Bullish/Bearish/Neutral │
├──────────────────┼──────────────────────────────────────┤
│ IPO Layer        │ Circuit Analysis → Investor Action   │
├──────────────────┼──────────────────────────────────────┤
│ Decision Layer   │ Buy / Hold / Sell Recommendation     │
└──────────────────┴──────────────────────────────────────┘
```

---

## 2. Data Sources & Structure

### CSV File: `nepse_data.csv`
| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Trading date (YYYY-MM-DD) |
| Company | string | Stock symbol (e.g., NABIL) |
| Sector | string | Industry sector |
| Closing_Price | float | Day's closing price (NPR) |
| Volume | int | Number of shares traded |
| Open_Price | float | Opening price |
| High_Price | float | Day's highest price |
| Low_Price | float | Day's lowest price |

**Total records:** 215 rows | **Date range:** Jan 2023 – Mar 2023

> **Note on Live Data:** Real-time data is a **future enhancement**. NEPSE data can be integrated via:
> - ShareSansar API (https://www.sharesansar.com)
> - MeroShare Web Scraping
> - NEPSE Official Portal (https://www.nepalstock.com)

---

## 3. Data Preprocessing

Steps performed in the preprocessing pipeline:

```python
# 1. Load CSV
df = pd.read_csv("nepse_data.csv")

# 2. Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# 3. Sort chronologically per company
df.sort_values(['Company', 'Date'], inplace=True)

# 4. Handle missing values
df[numeric_cols] = df[numeric_cols].ffill().bfill()

# 5. Drop rows with critical missing data
df.dropna(subset=['Date', 'Company', 'Closing_Price'], inplace=True)
```

**Engineered Features:**
| Feature | Formula | Purpose |
|---------|---------|---------|
| Daily_Return | (Close_t / Close_{t-1} - 1) × 100 | % change per day |
| MA7 | 7-day rolling mean | Short-term trend |
| MA20 | 20-day rolling mean | Medium-term trend |
| MA50 | 50-day rolling mean | Long-term trend |
| Volatility | 7-day rolling std of returns | Risk measure |
| Vol_Spike | Volume / Vol_MA7 | Unusual trading activity |
| Momentum | Close - Close_{t-5} | Price momentum |

---

## 4. Exploratory Data Analysis

### Summary Statistics
| Company | Avg Price (NPR) | High | Low | Avg Volume | Avg Return% | Volatility |
|---------|----------------|------|-----|------------|-------------|------------|
| NABIL | 1,127.58 | 1,230 | 1,022 | 24,295 | 0.343% | 0.863 |
| NICA | 900.67 | 978 | 820 | 17,288 | 0.302% | 0.930 |
| NLIC | 4,392.07 | 4,598 | 4,132 | 11,471 | 0.190% | 0.533 |
| MNBBL | 237.34 | 350 | 100 | 33,734 | 3.926% | 3.796 |
| CHCL | 677.89 | 764 | 583 | 26,324 | 0.427% | 1.307 |

### Key Observations
- **MNBBL** (IPO) has the highest average daily return (3.93%) and highest volatility (3.80) — typical for newly listed stocks
- **NLIC** shows the most stable performance with lowest volatility (0.533)
- **CHCL** (Hydropower) shows moderate volatility with consistent upward bias
- **NABIL and NICA** (established banks) show steady, low-volatility growth

---

## 5. Visualization Charts

Eight professional charts are generated:

| Chart | Filename | Description |
|-------|----------|-------------|
| 1 | `01_price_overview.png` | Time-series prices with MA7/MA20 overlays |
| 2 | `02_volume_price.png` | Volume bars (green=up, red=down) + price |
| 3 | `03_returns_volatility.png` | Daily returns and rolling volatility |
| 4 | `04_trend_detection.png` | MA crossover + support/resistance levels |
| 5 | `05_ipo_analysis.png` | MNBBL post-IPO price and volume journey |
| 6 | `06_predictions.png` | Actual vs predicted + 7-day forecast |
| 7 | `07_sentiment_dashboard.png` | Sentiment gauge + decision cards |
| 8 | `08_correlation_heatmap.png` | Inter-stock price correlation matrix |

---

## 6. Trend Detection

### Methodology
Trend is classified using **Moving Average Crossover** (MA7 vs MA20):

```
IF MA7 > MA20  →  UPTREND  (short-term above long-term)
IF MA7 < MA20  →  DOWNTREND
IF MA7 ≈ MA20  →  SIDEWAYS
```

### Support & Resistance
```
Support    = Minimum Low  of last 20 trading days
Resistance = Maximum High of last 20 trading days
```

### Results
| Company | Trend | Momentum | Support (NPR) | Resistance (NPR) |
|---------|-------|----------|--------------|-----------------|
| NABIL | Uptrend | Bullish | 1,102 | 1,230 |
| NICA | Uptrend | Bullish | 878 | 978 |
| NLIC | Uptrend | Bullish | 4,370 | 4,598 |
| MNBBL | Uptrend | Bullish | 192 | 350 |
| CHCL | Uptrend | Bullish | 660 | 764 |

---

## 7. Prediction Module

### Algorithm: Linear Regression
A simple, interpretable model that fits a straight line through historical prices.

```python
from sklearn.linear_model import LinearRegression

# X = trading day number (0, 1, 2, ...)
# y = closing price
model = LinearRegression()
model.fit(X_train, y_train)
future_prices = model.predict(future_days)
```

### 7-Day Price Forecast
| Company | Current Price | Day 7 Forecast | Change |
|---------|-------------|----------------|--------|
| NABIL | NPR 1,218 | NPR 1,246 | +2.3% |
| NICA | NPR 968 | NPR 980 | +1.2% |
| NLIC | NPR 4,562 | NPR 4,644 | +1.8% |
| MNBBL | NPR 345 | NPR 365 | +5.8% |
| CHCL | NPR 744 | NPR 769 | +3.4% |

> **Limitation:** Linear regression assumes prices follow a straight-line trend. Real markets are non-linear. This is a baseline model; ARIMA or LSTM would be more accurate for production use.

---

## 8. Investor Sentiment Analysis

### Rule-Based Sentiment Engine

```
BULLISH  if:  Volume Spike (>1.5×avg) AND Price Rise
         OR:  4+ of last 5 days were positive returns
         OR:  Continuous rise after IPO listing

BEARISH  if:  Volume Spike AND Price Fall (selling pressure)
         OR:  4+ of last 5 days were negative returns
         OR:  Sudden drop after rapid rise (panic selling)

NEUTRAL  if:  None of the above conditions met
```

### Sentiment Classification Logic (IPO Context)
| Scenario | Signal | Sentiment |
|----------|--------|-----------|
| IPO Day 1–5: High volume + rising price | Investor excitement | 🟢 Bullish |
| Price rose 100%+, then sharp drop | Profit booking | 🔴 Bearish |
| Gradual fall after peak | Panic selling | 🔴 Bearish |
| Price stabilizing after correction | Consolidation | 🟡 Neutral |
| New all-time high with rising volume | Strong demand | 🟢 Bullish |

---

## 9. IPO / Newly Listed Company Analysis

### MNBBL (Muktinath Bikas Bank) — Case Study

| Metric | Value |
|--------|-------|
| IPO Listing Date | January 16, 2023 |
| IPO Issue Price | NPR 100 (par value) |
| Day 1 Close | NPR 100 |
| Peak Price | NPR 345 (Day 35) |
| Upper Circuit Days | 6 days |
| Sudden Fall Events | 3 events |
| Return from IPO | +245% |

### Pattern Identified
```
Phase 1 (Days 1–12): UPPER CIRCUIT TREND
  → Price rose from 100 → 313 (+213%)
  → 6 days hit upper circuit (~10% daily limit)
  → High demand, low supply (investors holding)

Phase 2 (Days 13–19): PROFIT BOOKING / CORRECTION
  → Sharp drop from 313 → 192 (-39% in 5 days)
  → High sell volume — panic + profit booking
  → Typical post-IPO correction pattern

Phase 3 (Days 20+): STABILIZATION & RECOVERY
  → Gradual recovery and stabilization
  → Long-term investors accumulating
  → Price recovered to 345 by end of dataset
```

### Investor Action Prediction
Based on price vs IPO price (+245%) and position relative to peak (at peak):
> **Recommendation: HOLD** — Still performing well from IPO price; monitor for further correction signals before adding positions.

---

## 10. Decision Support System

Combines three signals into a Buy/Hold/Sell recommendation:

```
Score 3/3 (Uptrend + Bullish + Rising Slope) → BUY
Score 2/3                                     → BUY
Score 1/3                                     → HOLD
Score 0/3 (Downtrend + Bearish + Falling)    → SELL
```

### Final Decisions
| Company | Trend | Sentiment | Prediction | **Decision** |
|---------|-------|-----------|------------|-------------|
| NABIL | Uptrend | Bullish | ↑ NPR 1,246 | 🟢 **BUY** |
| NICA | Uptrend | Bullish | ↑ NPR 980 | 🟢 **BUY** |
| NLIC | Uptrend | Bullish | ↑ NPR 4,644 | 🟢 **BUY** |
| MNBBL | Uptrend | Bullish | ↑ NPR 365 | 🟢 **BUY** |
| CHCL | Uptrend | Bullish | ↑ NPR 769 | 🟢 **BUY** |

---

## 11. Results & Summary

### Key Findings
1. **All five companies** showed an upward trend during Jan–Mar 2023
2. **MNBBL (IPO)** demonstrated classic Nepali IPO behavior: rapid upper circuit rise → profit booking → recovery
3. **Banking sector** (NABIL, NICA) shows steady, low-risk growth suitable for conservative investors
4. **NLIC (Insurance)** had the most stable performance — lowest volatility
5. **CHCL (Hydropower)** shows seasonal price sensitivity with moderate risk

### Correlation Analysis
- NABIL and NICA are highly correlated (same sector, same market forces)
- NLIC moves somewhat independently (Insurance sector dynamics)
- MNBBL shows lowest correlation with others due to IPO effect

---

## 12. Conclusion & Future Improvements

### Conclusion
This NEPSE analysis system successfully demonstrates:
- ✅ Complete data preprocessing pipeline
- ✅ Exploratory data analysis with statistical insights
- ✅ Moving average based trend detection
- ✅ Linear regression price forecasting
- ✅ Rule-based investor sentiment classification
- ✅ IPO-specific pattern recognition (upper circuit, profit booking)
- ✅ Automated Buy/Hold/Sell decision support
- ✅ 8 professional visualization charts

### Future Improvements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **Real-time Data** | API integration with ShareSansar/NEPSE portal | High |
| **ARIMA/LSTM** | Advanced time-series models for better prediction | High |
| **Fundamental Analysis** | P/E ratio, EPS, book value integration | Medium |
| **Sector Dashboard** | Compare all sectors simultaneously | Medium |
| **Alert System** | Email/SMS alerts for buy/sell signals | Medium |
| **Candlestick Charts** | Japanese candlestick visualization | Low |
| **Portfolio Tracking** | Multi-stock portfolio P&L calculator | Low |
| **News Sentiment** | NLP on Nepali financial news | Future |
| **Mobile App** | Flutter/React Native app for retail investors | Future |

### Live Data Integration (Future Code Sketch)
```python
import requests

def fetch_nepse_live(symbol):
    """Future: Fetch live NEPSE data from API"""
    url = f"https://www.nepalstock.com/api/stocks/{symbol}"
    response = requests.get(url)
    return response.json()

# Alternative: Web scraping from ShareSansar
def scrape_sharesansar(symbol):
    """Future: Scrape historical data from ShareSansar"""
    url = f"https://www.sharesansar.com/company/{symbol}"
    # Use BeautifulSoup to parse HTML tables
    pass
```

---

## Project Deliverables

```
nepse_project/
├── nepse_analysis.py          ← Main source code
├── nepse_data.csv             ← Sample dataset (215 records, 5 companies)
├── PROJECT_REPORT.md          ← This document
└── outputs/
    ├── 01_price_overview.png
    ├── 02_volume_price.png
    ├── 03_returns_volatility.png
    ├── 04_trend_detection.png
    ├── 05_ipo_analysis.png
    ├── 06_predictions.png
    ├── 07_sentiment_dashboard.png
    └── 08_correlation_heatmap.png
```

---

*This project is designed for educational purposes and viva presentation. All investment decisions should be made with professional financial advice. Past performance does not guarantee future results.*
