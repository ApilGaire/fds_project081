"""
============================================================
NEPSE LIVE DATA FETCHER  (v4)
============================================================
HOW TO FETCH DATA — choose ONE method:

METHOD 1 (Recommended): nepse pip package
  pip3 install nepse
  python3 fetch_nepse_data.py --auto

METHOD 2: Manual entry (always works, no internet needed)
  python3 fetch_nepse_data.py --manual

METHOD 3: MeroLagani CSV download (free, no login)
  1. Go to https://merolagani.com/StockQuote.aspx
  2. Search each symbol, click "Download CSV"
  3. python3 fetch_nepse_data.py --csv /path/to/file.csv SYMBOL

  OR use the bulk download page:
  https://merolagani.com/handlers/webutility.aspx?req=stock_price_dowload
  python3 fetch_nepse_data.py --merolagani /path/to/downloaded.csv

METHOD 4: ShareSansar today-price page (backup scrape)
  python3 fetch_nepse_data.py --sharesansar

Run with --show to print last few rows of your CSV:
  python3 fetch_nepse_data.py --show
============================================================
"""

import requests
import pandas as pd
import os
import sys
import json
from datetime import datetime, date
from io import StringIO

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
COMPANIES = [
    "NABIL", "NICA",  "NLIC",  "MNBBL", "CHCL",
    "SCB",   "EBL",   "SBI",   "UPPER", "NHPC",
]

SECTOR_MAP = {
    "NABIL":  "Banking",    "NICA":  "Banking",
    "NLIC":   "Insurance",  "MNBBL": "Banking",
    "CHCL":   "Hydropower", "SCB":   "Banking",
    "EBL":    "Banking",    "SBI":   "Banking",
    "UPPER":  "Hydropower", "NHPC":  "Hydropower",
}

OUTPUT_FILE = "nepse_data.csv"
TODAY       = datetime.today().strftime("%Y-%m-%d")
HEADERS     = {"User-Agent": "Mozilla/5.0 (compatible; NEPSEFetcher/4.0)"}


# ─────────────────────────────────────────────────────────
# METHOD 1: nepse pip package (most reliable)
# ─────────────────────────────────────────────────────────
def fetch_via_nepse_package():
    """
    Uses the unofficial nepse package which handles
    NEPSE's WebAssembly token auth automatically.
    Install: pip3 install nepse
    """
    try:
        from nepse import Nepse
    except ImportError:
        print("  ✘  nepse package not installed.")
        print("     Run: pip3 install nepse")
        return None

    print("  Using nepse package (nepalstock.com) ...")
    try:
        nepse = Nepse()
        nepse.setTLSVerification(False)
        market_data = nepse.getLiveMarket()   # returns list of dicts
        if not market_data:
            print("  ✘  No live market data — market may be closed.")
            return None

        # Build lookup by symbol
        lookup = {}
        for item in market_data:
            sym = item.get("symbol") or item.get("securityName", "")
            if not sym:
                continue
            lookup[sym.upper()] = item

        rows = []
        for symbol in COMPANIES:
            item = lookup.get(symbol)
            if not item:
                print(f"  {symbol:6s}: not found in live market")
                continue
            close  = float(item.get("lastTradedPrice") or item.get("closingPrice") or 0)
            open_  = float(item.get("openPrice")  or close)
            high   = float(item.get("highPrice")  or close)
            low    = float(item.get("lowPrice")   or close)
            volume = int(float(item.get("totalTradeQuantity") or item.get("volume") or 0))
            if close <= 0:
                print(f"  {symbol:6s}: zero price — skipped")
                continue
            rows.append(_row(symbol, close, open_, high, low, volume))
            print(f"  {symbol:6s}: NPR {close:8.2f}  Vol {volume:,}")
        return rows if rows else None

    except Exception as e:
        print(f"  ✘  nepse package error: {e}")
        return None


# ─────────────────────────────────────────────────────────
# METHOD 2: Manual entry
# ─────────────────────────────────────────────────────────
def fetch_manual():
    """
    Prompt the user to type in prices from any source
    (MeroLagani, ShareSansar, NEPSE website, etc.)
    """
    print(f"\n  Manual entry mode — enter today's closing prices.")
    print(f"  (Press Enter to skip a stock)\n")
    rows = []
    for symbol in COMPANIES:
        try:
            val = input(f"  {symbol:6s} Closing Price (NPR): ").strip()
            if not val:
                print(f"         Skipped.")
                continue
            close = float(val)
            if close <= 0:
                print(f"         Invalid — skipped.")
                continue
            open_  = input(f"         Open   (Enter to use {close}): ").strip()
            high   = input(f"         High   (Enter to use {close}): ").strip()
            low    = input(f"         Low    (Enter to use {close}): ").strip()
            volume = input(f"         Volume (Enter to use 0):      ").strip()
            rows.append(_row(
                symbol,
                close,
                float(open_)  if open_  else close,
                float(high)   if high   else close,
                float(low)    if low    else close,
                int(volume)   if volume else 0,
            ))
        except (ValueError, KeyboardInterrupt):
            print(f"         Skipped.")
    return rows if rows else None


# ─────────────────────────────────────────────────────────
# METHOD 3a: MeroLagani single CSV
# ─────────────────────────────────────────────────────────
def fetch_from_csv(filepath, symbol):
    """
    Parse a CSV downloaded from MeroLagani for one symbol.
    Expected columns: Date, Open, High, Low, Close, Volume (or similar)
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip().lower() for c in df.columns]

        # Flexible column name matching
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "close" in cl:            col_map["close"] = col
            elif "open" in cl:           col_map["open"]  = col
            elif "high" in cl:           col_map["high"]  = col
            elif "low" in cl:            col_map["low"]   = col
            elif "vol" in cl:            col_map["vol"]   = col
            elif "date" in cl:           col_map["date"]  = col

        if "close" not in col_map:
            print(f"  ✘  Could not find 'Close' column in {filepath}")
            return None

        # Use the most recent row
        row = df.iloc[-1]
        close  = float(row[col_map["close"]])
        open_  = float(row[col_map["open"]])  if "open" in col_map else close
        high   = float(row[col_map["high"]])  if "high" in col_map else close
        low    = float(row[col_map["low"]])   if "low"  in col_map else close
        volume = int(float(row[col_map["vol"]])) if "vol" in col_map else 0

        if close <= 0:
            print(f"  ✘  Zero price in CSV — skipped")
            return None

        print(f"  {symbol:6s}: NPR {close:8.2f}  Vol {volume:,}  (from CSV)")
        return [_row(symbol, close, open_, high, low, volume)]

    except Exception as e:
        print(f"  ✘  CSV parse error: {e}")
        return None


# ─────────────────────────────────────────────────────────
# METHOD 3b: MeroLagani bulk download
# ─────────────────────────────────────────────────────────
def fetch_from_merolagani_bulk(filepath):
    """
    Parse the bulk CSV from:
    https://merolagani.com/handlers/webutility.aspx?req=stock_price_dowload
    which contains all listed companies for the latest trading day.
    """
    try:
        df = pd.read_csv(filepath)
        df.columns = [c.strip() for c in df.columns]
        print(f"  Loaded bulk CSV: {len(df)} rows, columns: {list(df.columns)}")

        # MeroLagani bulk CSV uses 'Symbol', 'LTP'/'Close', etc.
        col_sym   = next((c for c in df.columns if "symbol" in c.lower()), None)
        col_close = next((c for c in df.columns if c.lower() in ["ltp", "close", "closing price", "last traded price"]), None)
        col_open  = next((c for c in df.columns if "open" in c.lower()), None)
        col_high  = next((c for c in df.columns if "high" in c.lower()), None)
        col_low   = next((c for c in df.columns if "low" in c.lower()), None)
        col_vol   = next((c for c in df.columns if "vol" in c.lower()), None)

        if not col_sym or not col_close:
            print(f"  ✘  Cannot find Symbol/Close columns. Found: {list(df.columns)}")
            return None

        df[col_sym] = df[col_sym].str.strip().str.upper()
        rows = []
        for symbol in COMPANIES:
            match = df[df[col_sym] == symbol]
            if match.empty:
                print(f"  {symbol:6s}: not in bulk CSV — skipped")
                continue
            r      = match.iloc[-1]
            close  = float(str(r[col_close]).replace(",", ""))
            open_  = float(str(r[col_open]).replace(",", ""))  if col_open  else close
            high   = float(str(r[col_high]).replace(",", ""))  if col_high  else close
            low    = float(str(r[col_low]).replace(",", ""))   if col_low   else close
            volume = int(float(str(r[col_vol]).replace(",", ""))) if col_vol else 0
            if close <= 0:
                print(f"  {symbol:6s}: zero price — skipped")
                continue
            rows.append(_row(symbol, close, open_, high, low, volume))
            print(f"  {symbol:6s}: NPR {close:8.2f}  Vol {volume:,}")
        return rows if rows else None

    except Exception as e:
        print(f"  ✘  MeroLagani bulk CSV error: {e}")
        return None


# ─────────────────────────────────────────────────────────
# METHOD 4: ShareSansar today-share-price AJAX endpoint
# ─────────────────────────────────────────────────────────
def fetch_sharesansar():
    """
    ShareSansar exposes a DataTables server-side endpoint for today's prices.
    This may break if ShareSansar changes their backend.
    """
    url = "https://www.sharesansar.com/today-share-price"
    ajax_url = "https://www.sharesansar.com/stockdatas"

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # First get the page to set cookies + CSRF token
        resp = session.get(url, timeout=15)
        resp.raise_for_status()

        # Try the AJAX endpoint used by the DataTable on the page
        payload = {
            "draw": "1", "start": "0", "length": "500",
            "search[value]": "", "search[regex]": "false",
        }
        ajax_headers = {
            **HEADERS,
            "X-Requested-With": "XMLHttpRequest",
            "Referer": url,
        }
        ajax_resp = session.post(ajax_url, data=payload, headers=ajax_headers, timeout=15)
        ajax_resp.raise_for_status()
        data = ajax_resp.json()

        records = data.get("data") or data.get("aaData") or []
        if not records:
            print("  ✘  ShareSansar AJAX returned no records")
            return None

        # Each record is a list: [symbol, open, high, low, close, volume, ...]
        # Column order may vary — detect by header
        print(f"  ShareSansar: got {len(records)} records")
        rows = []
        for rec in records:
            if isinstance(rec, dict):
                sym   = str(rec.get("symbol", rec.get("stock_symbol", ""))).strip().upper()
                close = float(str(rec.get("ltp", rec.get("close_price", 0))).replace(",",""))
                open_ = float(str(rec.get("open_price", close)).replace(",",""))
                high  = float(str(rec.get("high_price", close)).replace(",",""))
                low   = float(str(rec.get("low_price",  close)).replace(",",""))
                vol   = int(float(str(rec.get("volume", rec.get("total_quantity", 0))).replace(",","")))
            elif isinstance(rec, list) and len(rec) >= 5:
                # Positional: symbol, open, high, low, close, volume
                sym   = str(rec[0]).strip().upper()
                open_ = float(str(rec[1]).replace(",",""))
                high  = float(str(rec[2]).replace(",",""))
                low   = float(str(rec[3]).replace(",",""))
                close = float(str(rec[4]).replace(",",""))
                vol   = int(float(str(rec[5]).replace(",",""))) if len(rec) > 5 else 0
            else:
                continue

            if sym not in COMPANIES or close <= 0:
                continue
            rows.append(_row(sym, close, open_, high, low, vol))
            print(f"  {sym:6s}: NPR {close:8.2f}  Vol {vol:,}")

        return rows if rows else None

    except Exception as e:
        print(f"  ✘  ShareSansar error: {e}")
        return None


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def _row(symbol, close, open_, high, low, volume):
    return {
        "Date":          TODAY,
        "Company":       symbol,
        "Sector":        SECTOR_MAP.get(symbol, "Unknown"),
        "Closing_Price": round(float(close), 2),
        "Volume":        int(volume),
        "Open_Price":    round(float(open_), 2),
        "High_Price":    round(float(high), 2),
        "Low_Price":     round(float(low), 2),
    }


def save_rows(rows):
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        existing = existing[existing["Closing_Price"] > 0]   # purge old zeros
        existing = existing[existing["Date"] != TODAY]        # avoid duplicates
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.sort_values(["Company", "Date"], inplace=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  ✔  Saved {len(rows)} stocks → '{OUTPUT_FILE}'")
    print(f"     Total rows : {len(combined)}")
    print(f"     Date range : {combined['Date'].min()} → {combined['Date'].max()}")
    print(f"\n  Now run: python3 nepse_analysis.py\n")


def show_csv():
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(df.tail(len(COMPANIES) * 3).to_string(index=False))
    else:
        print(f"No file found: {OUTPUT_FILE}")


def print_help():
    print("""
Usage:
  python3 fetch_nepse_data.py --auto          # Use nepse package (pip3 install nepse)
  python3 fetch_nepse_data.py --manual        # Type prices in manually
  python3 fetch_nepse_data.py --sharesansar   # Scrape ShareSansar
  python3 fetch_nepse_data.py --merolagani /path/to/bulk.csv
  python3 fetch_nepse_data.py --csv /path/to/file.csv SYMBOL
  python3 fetch_nepse_data.py --show          # Print last rows of CSV

Recommended workflow:
  1. pip3 install nepse
  2. python3 fetch_nepse_data.py --auto       (run daily after 3 PM NPT)
  3. python3 nepse_analysis.py
""")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    print(f"\n{'=' * 60}")
    print(f"  NEPSE DATA FETCHER  (v4)")
    print(f"  Date : {TODAY}  |  Stocks : {', '.join(COMPANIES)}")
    print(f"{'=' * 60}\n")

    if not args or "--help" in args or "-h" in args:
        print_help()
        return

    if "--show" in args:
        show_csv()
        return

    rows = None

    if "--auto" in args:
        print("  [Method 1] nepse package ...")
        rows = fetch_via_nepse_package()

    elif "--manual" in args:
        rows = fetch_manual()

    elif "--sharesansar" in args:
        print("  [Method 4] ShareSansar scrape ...")
        rows = fetch_sharesansar()

    elif "--merolagani" in args:
        idx = args.index("--merolagani")
        if idx + 1 >= len(args):
            print("  ✘  Provide path: --merolagani /path/to/file.csv")
            return
        print(f"  [Method 3b] MeroLagani bulk CSV ...")
        rows = fetch_from_merolagani_bulk(args[idx + 1])

    elif "--csv" in args:
        idx = args.index("--csv")
        if idx + 2 >= len(args):
            print("  ✘  Usage: --csv /path/to/file.csv SYMBOL")
            return
        filepath = args[idx + 1]
        symbol   = args[idx + 2].upper()
        print(f"  [Method 3a] Single CSV for {symbol} ...")
        rows = fetch_from_csv(filepath, symbol)

    else:
        print("  Unknown option. Run with --help for usage.")
        return

    if not rows:
        print("""
  ✘  No data saved.

  Try:
    --auto        pip3 install nepse  then  python3 fetch_nepse_data.py --auto
    --manual      enter prices yourself from sharesansar.com / merolagani.com
    --merolagani  download the bulk CSV from merolagani.com and pass the path
""")
        return

    save_rows(rows)


if __name__ == "__main__":
    main()
