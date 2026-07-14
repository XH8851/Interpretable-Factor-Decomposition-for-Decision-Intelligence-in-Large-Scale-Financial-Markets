"""
Paper 1 Upgrade — Step 2: Daily factor data from baostock
Pull peTTM, pbMRQ, psTTM, pcfNcfTTM, turnover for all constituents (2009-2019).

This is a large pull (~3,600 stocks × 2,572 trading days).
Uses chunked pulls with CSV caching to allow resumption.
"""

import baostock as bs
import pandas as pd
import os
import time
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "daily_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

START_DATE = "2009-01-01"
END_DATE = "2019-07-30"

FIELDS = "date,code,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"

print("=" * 60)
print("Paper 1: Daily factor pull (baostock)")
print("=" * 60)

# Load stock universe (from RAR + akshare union)
universe_path = os.path.join(OUTPUT_DIR, "paper1_stock_universe.csv")
if not os.path.exists(universe_path):
    print("ERROR: Run 03_paper1_constituents.py first")
    sys.exit(1)

df = pd.read_csv(universe_path)
bs_codes = df['bs_code'].tolist()
print(f"Total stocks to pull: {len(bs_codes)}")

# Check cache for already-pulled stocks
cached = set()
for f in os.listdir(CACHE_DIR):
    if f.endswith('.csv'):
        cached.add(f.replace('.csv', '').replace('_', '.'))

remaining = [c for c in bs_codes if c not in cached]
print(f"Already cached: {len(cached)}")
print(f"Remaining: {len(remaining)}")

if not remaining:
    print("All stocks already cached!")
    sys.exit(0)

lg = bs.login()
print(f"Login: {lg.error_msg}")

failed = []
start_time = time.time()

for i, code in enumerate(remaining):
    # Re-login every 500 stocks to avoid session timeout
    if i > 0 and i % 500 == 0:
        try:
            bs.logout()
        except:
            pass
        bs.login()

    if (i + 1) % 100 == 0 or i == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - i - 1) / rate / 60 if rate > 0 else 0
        print(f"  [{i+1}/{len(remaining)}] {code} | {rate:.1f} stocks/s | ETA: {eta:.0f}min", flush=True)

    rs = bs.query_history_k_data_plus(
        code, FIELDS,
        start_date=START_DATE, end_date=END_DATE,
        frequency="d", adjustflag="3"
    )

    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())

    if rows:
        stock_df = pd.DataFrame(rows, columns=rs.fields)
        cache_file = os.path.join(CACHE_DIR, f"{code.replace('.', '_')}.csv")
        stock_df.to_csv(cache_file, index=False)
    else:
        failed.append(code)

    time.sleep(0.02)

bs.logout()

print(f"\nCompleted: {len(remaining) - len(failed)}")
print(f"Failed: {len(failed)}")
if failed:
    pd.DataFrame({'code': failed}).to_csv(
        os.path.join(OUTPUT_DIR, "paper1_failed_pulls.csv"), index=False)
    print(f"Failed codes saved to paper1_failed_pulls.csv")

# Combine all cached files
print("\n" + "=" * 60)
print("Combining cached files...")
print("=" * 60)

all_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.csv')]
if all_files:
    chunks = []
    for f in all_files:
        chunks.append(pd.read_csv(f, dtype=str))
    combined = pd.concat(chunks, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "paper1_daily_factors.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved combined: {out_path}")
    print(f"Shape: {combined.shape}")
    print(f"Stocks: {combined['code'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
