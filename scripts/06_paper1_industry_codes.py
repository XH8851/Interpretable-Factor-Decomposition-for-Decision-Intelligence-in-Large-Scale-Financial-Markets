"""
Paper 1: Pull industry codes for all stocks in the universe.
"""

import baostock as bs
import pandas as pd
import os
import time

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

universe = pd.read_csv(os.path.join(OUTPUT_DIR, "paper1_stock_universe.csv"))
bs_codes = universe['bs_code'].tolist()
print(f"Total stocks: {len(bs_codes)}")

lg = bs.login()
print(f"Login: {lg.error_msg}")

results = []
failed = []

for i, code in enumerate(bs_codes):
    if i > 0 and i % 500 == 0:
        try:
            bs.logout()
        except:
            pass
        bs.login()

    if (i + 1) % 200 == 0 or i == 0:
        print(f"  [{i+1}/{len(bs_codes)}] {code}", flush=True)

    rs = bs.query_stock_industry(code=code)
    while rs.error_code == '0' and rs.next():
        row = rs.get_row_data()
        results.append(row)

    time.sleep(0.01)

bs.logout()

if results:
    df = pd.DataFrame(results, columns=['updateDate', 'code', 'code_name',
                                         'industry', 'industryClassification'])
    out_path = os.path.join(OUTPUT_DIR, "paper1_industry_codes.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Stocks with industry: {df['code'].nunique()}")
    print(f"Unique industries: {df['industry'].nunique()}")
    print(f"\nTop industries:")
    print(df['industry'].value_counts().head(15).to_string())
else:
    print("No results!")
