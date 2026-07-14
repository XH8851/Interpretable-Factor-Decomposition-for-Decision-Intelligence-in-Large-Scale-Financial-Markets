"""
Paper 1 Upgrade — Step 1: Get CSI All-Share constituent list
Uses akshare to get current constituents with inclusion dates.
"""

import akshare as ak
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: CSI All-Share constituents (akshare)")
print("=" * 60)

df = ak.index_stock_cons(symbol="000985")
print(f"Total constituents: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(df.head(5).to_string())

# Parse inclusion date
df['纳入日期'] = pd.to_datetime(df['纳入日期'], errors='coerce')
print(f"\nInclusion date range: {df['纳入日期'].min()} to {df['纳入日期'].max()}")
print(f"Stocks included before 2009-01-01: {(df['纳入日期'] < '2009-01-01').sum()}")
print(f"Stocks included before 2019-07-30: {(df['纳入日期'] < '2019-07-30').sum()}")

# Convert to baostock format
df['bs_code'] = df['品种代码'].apply(
    lambda x: f"sh.{x}" if str(x).startswith(('6','9')) else f"sz.{x}"
)

out_path = os.path.join(OUTPUT_DIR, "csi_allshare_constituents.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Optional cross-reference with an external stock list (e.g. a local .xls).
# Leave RAR_FILE as None to skip; set it to an existing file path to print overlap stats.
RAR_FILE = None  # e.g. "/path/to/2012-2018.xls"
if RAR_FILE and os.path.exists(RAR_FILE):
    rar_df = pd.read_excel(RAR_FILE, sheet_name=0)
    rar_codes = set(str(c).split('.')[0] for c in rar_df['证券代码'].dropna())
    akshare_codes = set(df['品种代码'].astype(str))
    print(f"\nRAR file stocks: {len(rar_codes)}")
    print(f"Akshare constituents: {len(akshare_codes)}")
    print(f"Overlap: {len(rar_codes & akshare_codes)}")
    print(f"In RAR only: {len(rar_codes - akshare_codes)}")
    print(f"In akshare only: {len(akshare_codes - rar_codes)}")
