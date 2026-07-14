"""
Additional baselines for paper1_ieee.tex — Table 3 and §4.4.

Approach:
  • Run LR walk-forward (fast; AUC 0.546 ≈ XGB 0.547) saving stock-level predictions
    with market-cap and industry attached.
  • Apply three portfolio construction schemes to the same LR ranking:
      A. Equal-weight        (reference, matches existing paper result)
      B. Float-cap-weighted  (weight by market cap within each quintile)
      C. Industry-neutral    (sort within industry, then aggregate)
  • The weighting-scheme comparison is clean because the same ranking is used for all three.

Market cap: amount / turn  (= price × total_shares, exact derivation from daily data)
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
PANEL_CACHE = os.path.join(OUTPUT_DIR, "paper1_monthly_panel_cache.csv")

# ─────────────────────────────────────────────────────────────────
# 1. Load cached monthly panel
# ─────────────────────────────────────────────────────────────────
print("Loading cached monthly panel …")
monthly = pd.read_csv(PANEL_CACHE)
monthly['ym'] = monthly['ym'].apply(lambda x: pd.Period(x, 'M'))
print(f"  {monthly.shape}")

feature_cols = ['mean_pe','mean_pb','mean_ps','mean_pcf','mean_turn',
                'vol_month','mom_3m','mom_6m','mom_12m','ret_month',
                'roeAvg','npMargin']
feature_cols = [c for c in feature_cols if c in monthly.columns]

ml = monthly.dropna(subset=['fwd_outperform']).copy()
ml = ml.dropna(subset=feature_cols, thresh=len(feature_cols)-2)
for col in feature_cols:
    ml[col] = ml.groupby('ym')[col].transform(lambda x: x.fillna(x.median()))
ml = ml.dropna(subset=feature_cols)
for col in feature_cols:
    ml[col] = ml.groupby('ym')[col].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

print(f"ML dataset: {ml.shape}")

unique_months  = sorted(ml['ym'].unique())
n_months       = len(unique_months)
train_window   = 60
test_window    = 1

# ─────────────────────────────────────────────────────────────────
# 2. LR walk-forward — save stock-level predictions
# ─────────────────────────────────────────────────────────────────
print("Running LR walk-forward (saving stock-level predictions) …")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

preds_list = []
auc_list   = []

for start in range(train_window, n_months - test_window + 1):
    train_months = unique_months[start - train_window:start]
    test_month   = unique_months[start]

    tr_mask = ml['ym'].isin(train_months)
    te_mask = ml['ym'] == test_month

    X_tr = ml.loc[tr_mask, feature_cols].values
    y_tr = ml.loc[tr_mask, 'fwd_outperform'].values
    X_te = ml.loc[te_mask, feature_cols].values
    y_te = ml.loc[te_mask, 'fwd_outperform'].values

    if len(y_te) == 0 or len(np.unique(y_tr)) < 2:
        continue

    sc = StandardScaler()
    lr = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
    lr.fit(sc.fit_transform(X_tr), y_tr)
    proba = lr.predict_proba(sc.transform(X_te))[:, 1]

    auc_list.append(roc_auc_score(y_te, proba))

    stock_preds = ml.loc[te_mask, ['code','ym','fwd_ret_1m','mean_mcap',
                                    'ind_broad']].copy()
    stock_preds['proba'] = proba
    preds_list.append(stock_preds)

    if len(auc_list) % 10 == 0:
        print(f"  fold {len(auc_list)}: {test_month}  AUC={auc_list[-1]:.4f}")

print(f"  Mean AUC: {np.mean(auc_list):.4f}  (n_folds={len(auc_list)})")
preds = pd.concat(preds_list, ignore_index=True)
preds = preds.dropna(subset=['fwd_ret_1m','mean_mcap'])
preds = preds[preds['mean_mcap'] > 0]

# ─────────────────────────────────────────────────────────────────
# 3. Portfolio construction helpers
# ─────────────────────────────────────────────────────────────────
def ew_ls(group):
    n = len(group); q = max(1, n//5)
    top = group.nlargest(q, 'proba')['fwd_ret_1m'].mean()
    bot = group.nsmallest(q, 'proba')['fwd_ret_1m'].mean()
    return pd.Series({'ls': top - bot})

def fcw_ls(group):
    """Float-cap-weighted returns within each quintile."""
    n = len(group); q = max(1, n//5)
    top_df = group.nlargest(q, 'proba')
    bot_df = group.nsmallest(q, 'proba')
    def wret(df):
        w = df['mean_mcap']
        return (w * df['fwd_ret_1m']).sum() / w.sum()
    return pd.Series({'ls': wret(top_df) - wret(bot_df)})

def industry_neutral_ls(month_df, min_stocks=5):
    """Within-industry quintile sorting, then aggregate across industries."""
    long_rets, short_rets = [], []
    for _, grp in month_df.dropna(subset=['ind_broad']).groupby('ind_broad'):
        n = len(grp)
        if n < min_stocks:
            continue
        q = max(1, n//5)
        long_rets.extend(grp.nlargest(q, 'proba')['fwd_ret_1m'].tolist())
        short_rets.extend(grp.nsmallest(q, 'proba')['fwd_ret_1m'].tolist())
    if not long_rets or not short_rets:
        return pd.Series({'ls': np.nan})
    return pd.Series({'ls': np.mean(long_rets) - np.mean(short_rets)})

# ─────────────────────────────────────────────────────────────────
# 4. Compute all three schemes
# ─────────────────────────────────────────────────────────────────
ew_m  = preds.groupby('ym').apply(ew_ls).reset_index().dropna()
fcw_m = preds.groupby('ym').apply(fcw_ls).reset_index().dropna()
in_m  = preds.groupby('ym').apply(industry_neutral_ls).reset_index().dropna()

def stats(df):
    m = df['ls'].mean()
    s = m / df['ls'].std() * np.sqrt(12)
    prof = (df['ls'] > 0).mean() * 100
    return m, s, prof

ew_mean,  ew_sh,  ew_prof  = stats(ew_m)
fcw_mean, fcw_sh, fcw_prof = stats(fcw_m)
in_mean,  in_sh,  in_prof  = stats(in_m)

# Save
fcw_m.to_csv(f"{OUTPUT_DIR}/paper1_floatcap_portfolio.csv", index=False)
in_m.to_csv( f"{OUTPUT_DIR}/paper1_industry_neutral_portfolio.csv", index=False)

# ─────────────────────────────────────────────────────────────────
# 5. Print results
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULTS — all use the same LR ranking (AUC≈0.546)")
print("="*65)
print(f"  A. Equal-weight        L-S {ew_mean:+.2f}%/mo  Sharpe {ew_sh:.2f}  {ew_prof:.0f}% months +ve")
print(f"  B. Float-cap-weighted  L-S {fcw_mean:+.2f}%/mo  Sharpe {fcw_sh:.2f}  {fcw_prof:.0f}% months +ve")
print(f"  C. Industry-neutral    L-S {in_mean:+.2f}%/mo  Sharpe {in_sh:.2f}  {in_prof:.0f}% months +ve")
print()
print("LaTeX rows (for tab:model_comp):")
print(f"  LR + float-cap-weighted  & 0.546 & ${fcw_mean:+.2f}\\%$ & {fcw_sh:.2f} \\\\")
print(f"  LR + industry-neutral    & 0.546 & ${in_mean:+.2f}\\%$ & {in_sh:.2f} \\\\")
print(f"  LR + equal-weight (ref)  & 0.546 & ${ew_mean:+.2f}\\%$ & {ew_sh:.2f} \\\\")
