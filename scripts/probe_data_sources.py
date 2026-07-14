"""
Probe akshare and baostock to check what's available for Paper 1:
- CSI All-Share (中证全指, code: 000985) constituents
- Daily prices + valuation factors (P/E, P/B, P/S, dividend yield, market cap)
- EV/EBIT, ROIC (harder — check if available)
- Date range: 2009-01-01 to 2019-07-30
"""

import akshare as ak
import baostock as bs
import pandas as pd
import traceback

SEPARATOR = "\n" + "="*60 + "\n"

def probe(label, fn):
    print(f"\n--- {label} ---")
    try:
        result = fn()
        if isinstance(result, pd.DataFrame):
            print(f"Shape: {result.shape}")
            print(f"Columns: {list(result.columns)}")
            print(result.head(3).to_string())
        else:
            print(result)
        return result
    except Exception as e:
        print(f"FAILED: {e}")
        return None

print(SEPARATOR + "1. CSI ALL-SHARE CONSTITUENTS (中证全指 000985)" + SEPARATOR)

# Current constituents (to understand structure)
probe("CSI All-Share current constituents (akshare)",
      lambda: ak.index_stock_cons(symbol="000985"))

# Historical constituent changes
probe("CSI All-Share constituent changes (akshare)",
      lambda: ak.index_stock_hist(symbol="000985"))

print(SEPARATOR + "2. VALUATION FACTORS — SAMPLE STOCK (600519 贵州茅台)" + SEPARATOR)

# Test with one well-known stock: Kweichow Moutai (2009-2019)
SAMPLE = "600519"

probe("Daily price + basic info (akshare)",
      lambda: ak.stock_zh_a_hist(symbol=SAMPLE, period="daily",
                                  start_date="20090101", end_date="20090131",
                                  adjust="qfq"))

probe("P/E P/B P/S daily (akshare stock_a_lg_indicator)",
      lambda: ak.stock_a_lg_indicator(symbol=SAMPLE))

# Check date range on the indicator
def check_indicator_range():
    df = ak.stock_a_lg_indicator(symbol=SAMPLE)
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    print(f"Date range: {df['交易日期'].min()} to {df['交易日期'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(df[df['交易日期'] >= '2009-01-01'].head(3).to_string())
    return df

probe("Indicator date range check", check_indicator_range)

print(SEPARATOR + "3. MARKET CAP (akshare)" + SEPARATOR)

probe("Daily market cap (akshare stock_zh_a_hist)",
      lambda: ak.stock_zh_a_hist(symbol=SAMPLE, period="daily",
                                  start_date="20090101", end_date="20090131",
                                  adjust="").rename(columns=str)[['日期','收盘','成交量','成交额']].head())

print(SEPARATOR + "4. DIVIDEND YIELD (akshare)" + SEPARATOR)

probe("Dividend data (akshare stock_history_dividend)",
      lambda: ak.stock_history_dividend())

print(SEPARATOR + "5. EV/EBIT AND ROIC (akshare)" + SEPARATOR)

probe("Financial indicators — EV/EBIT/ROIC check (akshare stock_financial_analysis_indicator)",
      lambda: ak.stock_financial_analysis_indicator(symbol=SAMPLE, start_year="2009"))

print(SEPARATOR + "6. BAOSTOCK — FACTOR DATA" + SEPARATOR)

lg = bs.login()
print(f"Baostock login: {lg.error_msg}")

def probe_baostock_daily():
    # baostock uses bs.6-digit prefix: sh.600519
    rs = bs.query_history_k_data_plus(
        "sh.600519",
        "date,code,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM",
        start_date="2009-01-05", end_date="2009-01-31",
        frequency="d", adjustflag="3"
    )
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)
    return df

probe("Baostock daily OHLCV + peTTM, pbMRQ, psTTM, pcfNcfTTM", probe_baostock_daily)

def probe_baostock_profit():
    rs = bs.query_profit_data(code="sh.600519", year=2009, quarter=1)
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)

probe("Baostock quarterly profit (ROIC proxy)", probe_baostock_profit)

def probe_baostock_growth():
    rs = bs.query_growth_data(code="sh.600519", year=2009, quarter=1)
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)

probe("Baostock quarterly growth data", probe_baostock_growth)

bs.logout()

print(SEPARATOR + "7. INDEX CONSTITUENT LIST — HISTORICAL (baostock)" + SEPARATOR)

lg = bs.login()

def probe_index_constituents():
    # CSI All-Share = sh.000985
    rs = bs.query_stock_basic(code="sh.000985")
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)

probe("Baostock index constituent lookup (sh.000985)", probe_index_constituents)

def probe_hs300_constituents():
    # Try CSI 300 as a known working example
    rs = bs.query_hs300_stocks()
    data = []
    while rs.error_code == '0' and rs.next():
        data.append(rs.get_row_data())
    return pd.DataFrame(data, columns=rs.fields)

probe("Baostock CSI 300 constituent list (sanity check)", probe_hs300_constituents)

bs.logout()

print(SEPARATOR + "SUMMARY" + SEPARATOR)
print("Check results above to determine what's available and what needs workarounds.")
