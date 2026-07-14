"""
Paper 1 Upgrade — Step 3: Quarterly fundamentals (ROE as ROIC proxy)
Pull quarterly profit data for all constituents, 2009-2019.
Resilient version: re-logins every 100 stocks, detects hangs via timeout.
"""

import baostock as bs
import pandas as pd
import os
import time
import sys
import signal

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
CACHE_DIR = os.path.join(OUTPUT_DIR, "quarterly_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Timeout handler for hung API calls
class APITimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise APITimeout("API call timed out")

def safe_login():
    try:
        bs.logout()
    except:
        pass
    lg = bs.login()
    return lg

print("=" * 60)
print("Paper 1: Quarterly fundamentals (baostock) — resilient mode")
print("=" * 60)

universe_path = os.path.join(OUTPUT_DIR, "paper1_stock_universe.csv")
if not os.path.exists(universe_path):
    print("ERROR: Run 03_paper1_constituents.py first")
    sys.exit(1)

df = pd.read_csv(universe_path)
bs_codes = df['bs_code'].tolist()
print(f"Total stocks: {len(bs_codes)}")

# Check cache
cached = set()
for f in os.listdir(CACHE_DIR):
    if f.endswith('.csv'):
        cached.add(f.replace('.csv', '').replace('_', '.'))

remaining = [c for c in bs_codes if c not in cached]
print(f"Already cached: {len(cached)}")
print(f"Remaining: {len(remaining)}")

if not remaining:
    print("All stocks already cached!")
else:
    lg = safe_login()
    print(f"Login: {lg.error_msg}")

    failed = []
    start_time = time.time()
    consecutive_empty = 0  # track consecutive empty responses (sign of dead connection)

    for i, code in enumerate(remaining):
        # Re-login every 100 stocks to prevent stale connections
        if i > 0 and i % 100 == 0:
            safe_login()
            consecutive_empty = 0
            elapsed = time.time() - start_time
            rate = (i) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i}/{len(remaining)}] Re-login. {rate:.2f} stocks/s | ETA: {eta:.0f}min", flush=True)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(remaining)}] {code} | {rate:.2f} stocks/s | ETA: {eta:.0f}min", flush=True)

        all_rows = []
        try:
            # Set a 30-second timeout for each stock's full quarterly pull
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)

            for year in range(2009, 2020):
                for quarter in range(1, 5):
                    rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
                    while rs.error_code == '0' and rs.next():
                        all_rows.append(rs.get_row_data())

            signal.alarm(0)  # cancel timeout
        except APITimeout:
            print(f"  TIMEOUT on {code} — re-logging in", flush=True)
            signal.alarm(0)
            safe_login()
            failed.append(code)
            continue
        except Exception as e:
            print(f"  ERROR on {code}: {e} — re-logging in", flush=True)
            signal.alarm(0)
            safe_login()
            failed.append(code)
            continue

        if all_rows:
            stock_df = pd.DataFrame(all_rows, columns=['code', 'pubDate', 'statDate',
                                                         'roeAvg', 'npMargin', 'gpMargin',
                                                         'netProfit', 'epsTTM', 'MBRevenue',
                                                         'totalShare', 'liqaShare'])
            cache_file = os.path.join(CACHE_DIR, f"{code.replace('.', '_')}.csv")
            stock_df.to_csv(cache_file, index=False)
            consecutive_empty = 0
        else:
            failed.append(code)
            consecutive_empty += 1
            # If 20 consecutive stocks return nothing, connection is probably dead
            if consecutive_empty >= 20:
                print(f"  20 consecutive empty — forcing re-login", flush=True)
                safe_login()
                consecutive_empty = 0

        time.sleep(0.03)

    try:
        bs.logout()
    except:
        pass

    print(f"\nCompleted: {len(remaining) - len(failed)}")
    print(f"Failed: {len(failed)}")

# Combine all cached files
print("\nCombining cached files...")
all_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.csv')]
if all_files:
    chunks = [pd.read_csv(f, dtype=str) for f in all_files]
    combined = pd.concat(chunks, ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "paper1_quarterly_fundamentals.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {combined.shape}")
    print(f"Stocks: {combined['code'].nunique()}")
