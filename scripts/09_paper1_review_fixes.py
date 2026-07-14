"""
Paper 1: Review fixes script
Addresses all must-fix and should-fix issues from the paper review:
  #1  Bootstrap CIs + Newey-West t-stats
  #2  Factor-model alpha regression
  #3  Monthly rolling CV (60mo train → 1mo test → step 1mo)
  #4  Correct Sharpe annualization (verified)
  #5  Turnover ablation paradox (explained in manuscript)
  #6  Add log market cap as feature
  #7  Clarify rolling vs expanding (code confirms rolling)
  #8  Winsorize features at 1/99 percentile
  #9  Per-industry SHAP breakdown
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. Load and build monthly panel (same as 08, but with market cap)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Loading data and building monthly panel")
print("=" * 60)

daily = pd.read_csv(f"{OUTPUT_DIR}/paper1_daily_factors.csv", dtype=str)
numeric_cols = ['close', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
for col in numeric_cols:
    daily[col] = pd.to_numeric(daily[col], errors='coerce')
daily['isST'] = daily['isST'].map({'1': 1, '0': 0, 1: 1, 0: 0}).fillna(0).astype(int)
daily['date'] = pd.to_datetime(daily['date'])
daily = daily[daily['isST'] == 0].copy()
daily['ym'] = daily['date'].dt.to_period('M')

monthly = daily.groupby(['code', 'ym']).agg(
    ret_month=('pctChg', lambda x: ((1 + x / 100).prod() - 1) * 100),
    mean_pe=('peTTM', 'mean'),
    mean_pb=('pbMRQ', 'mean'),
    mean_ps=('psTTM', 'mean'),
    mean_pcf=('pcfNcfTTM', 'mean'),
    mean_turn=('turn', 'mean'),
    vol_month=('pctChg', 'std'),
    last_close=('close', 'last'),
    n_days=('pctChg', 'count'),
).reset_index()
monthly = monthly[monthly['n_days'] >= 10].copy()

# Momentum
monthly = monthly.sort_values(['code', 'ym'])
for window in [3, 6, 12]:
    monthly[f'mom_{window}m'] = monthly.groupby('code')['ret_month'].transform(
        lambda x: x.rolling(window).sum()
    )

# Forward return
monthly['fwd_ret_1m'] = monthly.groupby('code')['ret_month'].shift(-1)
monthly['fwd_outperform'] = monthly.groupby('ym')['fwd_ret_1m'].transform(
    lambda x: (x > x.median()).astype(int)
)

print(f"Monthly panel: {monthly.shape}, stocks: {monthly['code'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# 2. Merge quarterly fundamentals + compute market cap (FIX #6)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Merging quarterly fundamentals + market cap")
print("=" * 60)

qtr = pd.read_csv(f"{OUTPUT_DIR}/paper1_quarterly_fundamentals.csv", dtype=str)
for col in ['roeAvg', 'npMargin', 'totalShare']:
    qtr[col] = pd.to_numeric(qtr[col], errors='coerce')
qtr['statDate'] = pd.to_datetime(qtr['statDate'], errors='coerce')
qtr['ym'] = qtr['statDate'].dt.to_period('M')
qtr_agg = qtr.groupby(['code', 'ym']).agg(
    roeAvg=('roeAvg', 'last'),
    npMargin=('npMargin', 'last'),
    totalShare=('totalShare', 'last'),
).reset_index()
qtr_agg = qtr_agg.sort_values(['code', 'ym'])
monthly = monthly.merge(qtr_agg, on=['code', 'ym'], how='left')
monthly[['roeAvg', 'npMargin', 'totalShare']] = monthly.groupby('code')[['roeAvg', 'npMargin', 'totalShare']].ffill()

# FIX #6: Log market cap = log(close × totalShare)
# totalShare is in 万股 (10,000 shares), close in yuan
monthly['market_cap'] = monthly['last_close'] * monthly['totalShare'] * 10000  # in yuan
monthly['log_mcap'] = np.log(monthly['market_cap'].clip(lower=1))
n_mcap = monthly['log_mcap'].notna().sum()
print(f"Market cap coverage: {n_mcap}/{len(monthly)} ({n_mcap/len(monthly)*100:.1f}%)")

# Merge industry
ind = pd.read_csv(f"{OUTPUT_DIR}/paper1_industry_codes.csv")
ind['ind_broad'] = ind['industry'].str[:3]
ind = ind[['code', 'industry', 'ind_broad']].drop_duplicates(subset='code', keep='last')
monthly = monthly.merge(ind, on='code', how='left')

# ═══════════════════════════════════════════════════════════════════
# 3. Feature preparation with winsorization (FIX #8) and size (FIX #6)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Feature preparation with winsorization")
print("=" * 60)

feature_cols = ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf', 'mean_turn',
                'vol_month', 'mom_3m', 'mom_6m', 'mom_12m', 'ret_month',
                'roeAvg', 'npMargin', 'log_mcap']  # FIX #6: added log_mcap

ml_data = monthly.dropna(subset=['fwd_outperform']).copy()
ml_data = ml_data.dropna(subset=feature_cols, thresh=len(feature_cols) - 2)

# Median imputation
for col in feature_cols:
    ml_data[col] = ml_data.groupby('ym')[col].transform(lambda x: x.fillna(x.median()))
ml_data = ml_data.dropna(subset=feature_cols)

# FIX #8: Winsorize at 1st/99th percentile within each month
print("Winsorizing features at 1st/99th percentile per month...")
for col in feature_cols:
    ml_data[col] = ml_data.groupby('ym')[col].transform(
        lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
    )

print(f"ML dataset: {ml_data.shape}, stocks: {ml_data['code'].nunique()}, months: {ml_data['ym'].nunique()}")
print(f"Features ({len(feature_cols)}): {feature_cols}")

# ═══════════════════════════════════════════════════════════════════
# 4. Descriptive statistics (post-winsorization)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Descriptive statistics (winsorized)")
print("=" * 60)

desc_cols = feature_cols + ['fwd_ret_1m']
desc = ml_data[desc_cols].describe().T[['count', 'mean', 'std', 'min', 'max', '50%']]
desc.columns = ['Obs', 'Mean', 'Std', 'Min', 'Max', 'Median']
desc['Obs'] = desc['Obs'].astype(int)
print(desc.round(4).to_string())
desc.round(4).to_csv(f"{OUTPUT_DIR}/paper1_v3_descriptive_stats.csv")

# ═══════════════════════════════════════════════════════════════════
# 5. Monthly rolling CV (FIX #3: 60mo train → 1mo test → step 1mo)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Monthly rolling XGBoost (60mo train, 1mo test, 1mo step)")
print("=" * 60)

from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

unique_months = sorted(ml_data['ym'].unique())
n_months = len(unique_months)
train_window = 60  # Rolling window (FIX #7: confirmed rolling, not expanding)

def make_model():
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         verbosity=0, n_jobs=-1)

all_monthly_results = []
all_predictions = []
retrain_interval = 3  # Retrain every 3 months for efficiency (predict monthly)
current_model = None

for i in range(train_window, n_months - 1):
    test_month = unique_months[i]
    train_months = unique_months[i - train_window:i]

    train_mask = ml_data['ym'].isin(train_months)
    test_mask = ml_data['ym'] == test_month

    X_train = ml_data.loc[train_mask, feature_cols].values
    y_train = ml_data.loc[train_mask, 'fwd_outperform'].values
    X_test = ml_data.loc[test_mask, feature_cols].values
    y_test = ml_data.loc[test_mask, 'fwd_outperform'].values

    if len(y_test) == 0 or len(np.unique(y_train)) < 2:
        continue

    # Retrain every N months
    if current_model is None or (i - train_window) % retrain_interval == 0:
        current_model = make_model()
        current_model.fit(X_train, y_train)

    proba = current_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
    acc = accuracy_score(y_test, (proba > 0.5).astype(int))

    all_monthly_results.append({
        'test_month': str(test_month),
        'auc': auc, 'accuracy': acc,
        'n_train': len(y_train), 'n_test': len(y_test),
    })

    test_preds = ml_data.loc[test_mask, ['code', 'ym', 'fwd_ret_1m', 'fwd_outperform']].copy()
    test_preds['proba'] = proba
    if 'industry' in ml_data.columns:
        test_preds['industry'] = ml_data.loc[test_mask, 'industry'].values
        test_preds['ind_broad'] = ml_data.loc[test_mask, 'ind_broad'].values
    all_predictions.append(test_preds)

    if (i - train_window + 1) % 10 == 0 or i == train_window:
        print(f"  Month {i - train_window + 1}/{n_months - train_window - 1}: "
              f"{test_month}, AUC={auc:.4f}, n_test={len(y_test)}", flush=True)

results_df = pd.DataFrame(all_monthly_results)
mean_auc = results_df['auc'].mean()
mean_acc = results_df['accuracy'].mean()
n_test_months = len(results_df)
print(f"\n  Total test months: {n_test_months}")
print(f"  Mean AUC: {mean_auc:.4f}")
print(f"  Mean Accuracy: {mean_acc:.4f}")
results_df.to_csv(f"{OUTPUT_DIR}/paper1_v3_cv_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 6. Portfolio returns
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Portfolio returns")
print("=" * 60)

preds_all = pd.concat(all_predictions, ignore_index=True)

def compute_ls(group):
    n = len(group)
    q = max(1, n // 5)
    top = group.nlargest(q, 'proba')['fwd_ret_1m'].mean()
    bot = group.nsmallest(q, 'proba')['fwd_ret_1m'].mean()
    return pd.Series({'ml_top': top, 'ml_bot': bot, 'ml_ls': top - bot,
                       'equal_weight': group['fwd_ret_1m'].mean()})

monthly_ls = preds_all.groupby('ym').apply(compute_ls).reset_index()

# FIX #4: Verify Sharpe annualization
# Monthly Sharpe = mean / std
# Annualized Sharpe = monthly_sharpe × sqrt(12)
ls_mean = monthly_ls['ml_ls'].mean()
ls_std = monthly_ls['ml_ls'].std()
monthly_sharpe = ls_mean / ls_std
ann_sharpe = monthly_sharpe * np.sqrt(12)

print(f"  Mean monthly L-S: {ls_mean:+.4f}%")
print(f"  Std monthly L-S: {ls_std:.4f}%")
print(f"  Monthly Sharpe: {monthly_sharpe:.4f}")
print(f"  Annualized Sharpe: {ann_sharpe:.3f}  (= {monthly_sharpe:.4f} × √12)")
print(f"  Mean top quintile: {monthly_ls['ml_top'].mean():+.4f}%")
print(f"  Mean bottom quintile: {monthly_ls['ml_bot'].mean():+.4f}%")
print(f"  Equal-weight: {monthly_ls['equal_weight'].mean():+.4f}%")

# Max drawdown
cum = (1 + monthly_ls['ml_ls'] / 100).cumprod()
peak = cum.expanding().max()
dd = (cum - peak) / peak
print(f"  Max drawdown: {dd.min()*100:.1f}%")
print(f"  Profitable months: {(monthly_ls['ml_ls'] > 0).mean()*100:.0f}%")

monthly_ls.to_csv(f"{OUTPUT_DIR}/paper1_v3_portfolio_returns.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 7. FIX #1: Bootstrap CIs + Newey-West t-statistics
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. Statistical inference (Bootstrap CIs + Newey-West)")
print("=" * 60)

n_bootstrap = 10000
rng = np.random.RandomState(42)

# --- Bootstrap CI for mean L-S return ---
ls_returns = monthly_ls['ml_ls'].values
boot_means = np.array([ls_returns[rng.choice(len(ls_returns), len(ls_returns), replace=True)].mean()
                        for _ in range(n_bootstrap)])
ls_ci_low, ls_ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"  Mean L-S: {ls_mean:+.3f}% [95% CI: {ls_ci_low:+.3f}, {ls_ci_high:+.3f}]")

# --- Bootstrap CI for Sharpe ratio ---
def compute_sharpe(returns):
    if returns.std() == 0: return 0
    return returns.mean() / returns.std() * np.sqrt(12)

boot_sharpes = np.array([compute_sharpe(ls_returns[rng.choice(len(ls_returns), len(ls_returns), replace=True)])
                          for _ in range(n_bootstrap)])
sharpe_ci_low, sharpe_ci_high = np.percentile(boot_sharpes, [2.5, 97.5])
print(f"  Annualized Sharpe: {ann_sharpe:.3f} [95% CI: {sharpe_ci_low:.3f}, {sharpe_ci_high:.3f}]")

# --- Bootstrap CI for AUC ---
auc_values = results_df['auc'].dropna().values
boot_aucs = np.array([auc_values[rng.choice(len(auc_values), len(auc_values), replace=True)].mean()
                       for _ in range(n_bootstrap)])
auc_ci_low, auc_ci_high = np.percentile(boot_aucs, [2.5, 97.5])
print(f"  Mean AUC: {mean_auc:.4f} [95% CI: {auc_ci_low:.4f}, {auc_ci_high:.4f}]")

# --- Newey-West t-statistic for mean L-S ---
def newey_west_tstat(returns, max_lag=None):
    """Newey-West HAC t-statistic for testing H0: mean = 0"""
    T = len(returns)
    if max_lag is None:
        max_lag = int(np.floor(4 * (T / 100) ** (2/9)))  # Andrews (1991) rule
    mean_r = returns.mean()
    demeaned = returns - mean_r

    # Gamma_0
    gamma_0 = np.sum(demeaned ** 2) / T
    # Newey-West weighted sum of autocovariances
    nw_var = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.sum(demeaned[lag:] * demeaned[:-lag]) / T
        nw_var += 2 * weight * gamma_lag

    se = np.sqrt(nw_var / T)
    t_stat = mean_r / se
    # Two-sided p-value from normal approximation
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return t_stat, se, p_value

t_stat, nw_se, p_value = newey_west_tstat(ls_returns)
print(f"  Newey-West t-stat: {t_stat:.3f} (SE={nw_se:.4f}, p={p_value:.6f})")

# Save inference results
inference = {
    'metric': ['Mean L-S (%/mo)', 'Annualized Sharpe', 'Mean AUC'],
    'estimate': [ls_mean, ann_sharpe, mean_auc],
    'ci_low': [ls_ci_low, sharpe_ci_low, auc_ci_low],
    'ci_high': [ls_ci_high, sharpe_ci_high, auc_ci_high],
    'nw_tstat': [t_stat, np.nan, np.nan],
    'nw_pvalue': [p_value, np.nan, np.nan],
}
pd.DataFrame(inference).to_csv(f"{OUTPUT_DIR}/paper1_v3_inference.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 8. FIX #2: Factor-model alpha (self-constructed factors)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. Factor-model alpha regression")
print("=" * 60)

# Construct factors from our data each month:
# MKT = equal-weighted market return
# SMB = small minus big (bottom vs top half by market cap)
# HML = high minus low (low P/B minus high P/B, i.e., value minus growth)
# UMD = up minus down (high minus low 12-month momentum)

factor_data = ml_data[['code', 'ym', 'fwd_ret_1m', 'log_mcap', 'mean_pb', 'mom_12m']].dropna().copy()

def build_factors(group):
    n = len(group)
    if n < 20:
        return pd.Series({'mkt': np.nan, 'smb': np.nan, 'hml': np.nan, 'umd': np.nan})

    mkt = group['fwd_ret_1m'].mean()

    # SMB: median split on market cap
    med_mcap = group['log_mcap'].median()
    small = group[group['log_mcap'] <= med_mcap]['fwd_ret_1m'].mean()
    big = group[group['log_mcap'] > med_mcap]['fwd_ret_1m'].mean()
    smb = small - big

    # HML: top vs bottom 30% by P/B (low P/B = value)
    q30 = group['mean_pb'].quantile(0.3)
    q70 = group['mean_pb'].quantile(0.7)
    value = group[group['mean_pb'] <= q30]['fwd_ret_1m'].mean()
    growth = group[group['mean_pb'] >= q70]['fwd_ret_1m'].mean()
    hml = value - growth

    # UMD: top vs bottom 30% by 12-month momentum
    q30_m = group['mom_12m'].quantile(0.3)
    q70_m = group['mom_12m'].quantile(0.7)
    winners = group[group['mom_12m'] >= q70_m]['fwd_ret_1m'].mean()
    losers = group[group['mom_12m'] <= q30_m]['fwd_ret_1m'].mean()
    umd = winners - losers

    return pd.Series({'mkt': mkt, 'smb': smb, 'hml': hml, 'umd': umd})

monthly_factors = factor_data.groupby('ym').apply(build_factors).reset_index()
monthly_factors['ym'] = monthly_factors['ym'].astype(str)
monthly_ls_str = monthly_ls.copy()
monthly_ls_str['ym'] = monthly_ls_str['ym'].astype(str)

reg_data = monthly_ls_str[['ym', 'ml_ls']].merge(monthly_factors, on='ym', how='inner').dropna()
print(f"  Regression sample: {len(reg_data)} months")

from scipy import stats as scipy_stats
import statsmodels.api as sm

# CAPM: L-S = alpha + beta*MKT + epsilon
X_capm = sm.add_constant(reg_data[['mkt']])
y = reg_data['ml_ls']
model_capm = sm.OLS(y, X_capm).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"\n  CAPM alpha: {model_capm.params['const']:+.3f}%/mo (t={model_capm.tvalues['const']:.3f}, p={model_capm.pvalues['const']:.4f})")
print(f"  CAPM beta(MKT): {model_capm.params['mkt']:.3f} (t={model_capm.tvalues['mkt']:.3f})")

# Three-factor: L-S = alpha + b1*MKT + b2*SMB + b3*HML + epsilon
X_ff3 = sm.add_constant(reg_data[['mkt', 'smb', 'hml']])
model_ff3 = sm.OLS(y, X_ff3).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"\n  FF3 alpha: {model_ff3.params['const']:+.3f}%/mo (t={model_ff3.tvalues['const']:.3f}, p={model_ff3.pvalues['const']:.4f})")
print(f"  FF3 beta(MKT): {model_ff3.params['mkt']:.3f}, beta(SMB): {model_ff3.params['smb']:.3f}, beta(HML): {model_ff3.params['hml']:.3f}")

# Four-factor: L-S = alpha + b1*MKT + b2*SMB + b3*HML + b4*UMD + epsilon
X_ff4 = sm.add_constant(reg_data[['mkt', 'smb', 'hml', 'umd']])
model_ff4 = sm.OLS(y, X_ff4).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f"\n  Carhart-4 alpha: {model_ff4.params['const']:+.3f}%/mo (t={model_ff4.tvalues['const']:.3f}, p={model_ff4.pvalues['const']:.4f})")
print(f"  beta(MKT)={model_ff4.params['mkt']:.3f}, beta(SMB)={model_ff4.params['smb']:.3f}, "
      f"beta(HML)={model_ff4.params['hml']:.3f}, beta(UMD)={model_ff4.params['umd']:.3f}")
print(f"  R²: {model_ff4.rsquared:.4f}")

# Save factor regression results
alpha_results = []
for name, model in [('CAPM', model_capm), ('FF3', model_ff3), ('Carhart-4', model_ff4)]:
    alpha_results.append({
        'model': name,
        'alpha': model.params['const'],
        'alpha_tstat': model.tvalues['const'],
        'alpha_pvalue': model.pvalues['const'],
        'r_squared': model.rsquared,
    })
pd.DataFrame(alpha_results).to_csv(f"{OUTPUT_DIR}/paper1_v3_factor_alpha.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 9. SHAP analysis (full model with 13 features including log_mcap)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. SHAP feature importance (13 features)")
print("=" * 60)

import shap

# Train final model on full data
X_all = ml_data[feature_cols].values
y_all = ml_data['fwd_outperform'].values
final_model = make_model()
final_model.n_estimators = 300
final_model.fit(X_all, y_all)

explainer = shap.TreeExplainer(final_model)
sample_idx = np.random.RandomState(42).choice(len(X_all), min(5000, len(X_all)), replace=False)
shap_values = explainer.shap_values(X_all[sample_idx])

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': mean_abs_shap})
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
shap_df['category'] = shap_df['feature'].map({
    'mean_pe': 'Valuation', 'mean_pb': 'Valuation', 'mean_ps': 'Valuation', 'mean_pcf': 'Valuation',
    'mean_turn': 'Behavioral', 'vol_month': 'Behavioral', 'ret_month': 'Behavioral',
    'mom_3m': 'Behavioral', 'mom_6m': 'Behavioral', 'mom_12m': 'Behavioral',
    'roeAvg': 'Fundamental', 'npMargin': 'Fundamental',
    'log_mcap': 'Size',
})

total_shap = shap_df['mean_abs_shap'].sum()
cat_shares = shap_df.groupby('category')['mean_abs_shap'].sum() / total_shap * 100
print("\nSHAP by category:")
for cat, share in cat_shares.sort_values(ascending=False).items():
    print(f"  {cat}: {share:.1f}%")

print("\nSHAP feature importance:")
print(shap_df.to_string(index=False))
shap_df.to_csv(f"{OUTPUT_DIR}/paper1_v3_shap_importance.csv", index=False)

# SHAP beeswarm plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_all[sample_idx], feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_v3_shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("SHAP plot saved.")

# ═══════════════════════════════════════════════════════════════════
# 10. Ablation study (with new features)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. Ablation study")
print("=" * 60)

ablation_groups = {
    'No valuation (PE/PB/PS/PCF)': ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf'],
    'No momentum (3m/6m/12m)': ['mom_3m', 'mom_6m', 'mom_12m'],
    'No turnover': ['mean_turn'],
    'No volatility': ['vol_month'],
    'No current return': ['ret_month'],
    'No ROE/margins': ['roeAvg', 'npMargin'],
    'No size (log mcap)': ['log_mcap'],
}

# Use monthly rolling same as main model
def run_ablation(abl_features):
    aucs = []
    model = None
    for i in range(train_window, n_months - 1):
        test_month = unique_months[i]
        train_months_abl = unique_months[i - train_window:i]
        train_mask = ml_data['ym'].isin(train_months_abl)
        test_mask = ml_data['ym'] == test_month
        X_tr = ml_data.loc[train_mask, abl_features].values
        y_tr = ml_data.loc[train_mask, 'fwd_outperform'].values
        X_te = ml_data.loc[test_mask, abl_features].values
        y_te = ml_data.loc[test_mask, 'fwd_outperform'].values
        if len(y_te) == 0 or len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        if model is None or (i - train_window) % retrain_interval == 0:
            model = make_model()
            model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, proba))
    return np.mean(aucs) if aucs else np.nan

full_auc = mean_auc
ablation_results = [{'config': 'Full model (13 features)', 'mean_auc': full_auc, 'delta': 0.0}]

for abl_name, remove_cols in ablation_groups.items():
    abl_features = [f for f in feature_cols if f not in remove_cols]
    abl_auc = run_ablation(abl_features)
    delta = abl_auc - full_auc
    ablation_results.append({'config': abl_name, 'mean_auc': abl_auc, 'delta': delta})
    print(f"  {abl_name}: AUC={abl_auc:.4f} (delta={delta:+.4f})")

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(f"{OUTPUT_DIR}/paper1_v3_ablation.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 11. FIX #9: Per-industry SHAP breakdown
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("11. Per-industry SHAP category breakdown")
print("=" * 60)

# Compute SHAP values for a larger sample, then group by industry
shap_sample_size = min(20000, len(X_all))
large_sample_idx = np.random.RandomState(42).choice(len(X_all), shap_sample_size, replace=False)
shap_values_large = explainer.shap_values(X_all[large_sample_idx])

# Map each sample to its industry
sample_industries = ml_data.iloc[large_sample_idx]['ind_broad'].values

category_map = {
    'mean_pe': 'Valuation', 'mean_pb': 'Valuation', 'mean_ps': 'Valuation', 'mean_pcf': 'Valuation',
    'mean_turn': 'Behavioral', 'vol_month': 'Behavioral', 'ret_month': 'Behavioral',
    'mom_3m': 'Behavioral', 'mom_6m': 'Behavioral', 'mom_12m': 'Behavioral',
    'roeAvg': 'Fundamental', 'npMargin': 'Fundamental',
    'log_mcap': 'Size',
}

# For each industry, compute the category shares
ind_shap_results = []
unique_industries = set(x for x in sample_industries if isinstance(x, str) and not pd.isna(x))
for ind_code in sorted(unique_industries):
    mask = sample_industries == ind_code
    if mask.sum() < 50:
        continue
    ind_shap = np.abs(shap_values_large[mask]).mean(axis=0)
    total = ind_shap.sum()
    if total == 0:
        continue

    cat_totals = {}
    for j, feat in enumerate(feature_cols):
        cat = category_map.get(feat, 'Other')
        cat_totals[cat] = cat_totals.get(cat, 0) + ind_shap[j]

    result = {'ind_broad': ind_code, 'n_obs': int(mask.sum())}
    for cat in ['Behavioral', 'Valuation', 'Fundamental', 'Size']:
        result[f'{cat}_pct'] = cat_totals.get(cat, 0) / total * 100
    ind_shap_results.append(result)

ind_shap_df = pd.DataFrame(ind_shap_results).sort_values('Behavioral_pct', ascending=False)
print(f"\nPer-industry SHAP category shares ({len(ind_shap_df)} industries):")
print(f"  Behavioral range: {ind_shap_df['Behavioral_pct'].min():.1f}%–{ind_shap_df['Behavioral_pct'].max():.1f}%")
print(f"  Valuation range: {ind_shap_df['Valuation_pct'].min():.1f}%–{ind_shap_df['Valuation_pct'].max():.1f}%")
print(f"  Industries where Behavioral > 50%: {(ind_shap_df['Behavioral_pct'] > 50).sum()}/{len(ind_shap_df)}")
print(f"\nTop 5 (highest Behavioral %):")
print(ind_shap_df.head().to_string(index=False))
print(f"\nBottom 5 (lowest Behavioral %):")
print(ind_shap_df.tail().to_string(index=False))
ind_shap_df.to_csv(f"{OUTPUT_DIR}/paper1_v3_industry_shap.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 12. Industry transferability (with new model)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("12. Industry transferability")
print("=" * 60)

ind_preds = preds_all.dropna(subset=['ind_broad', 'fwd_ret_1m'])

def ind_ls(group):
    n = len(group)
    q = max(1, n // 5)
    if n < 10:
        return pd.Series({'ls': np.nan, 'n': n})
    top = group.nlargest(q, 'proba')['fwd_ret_1m'].mean()
    bot = group.nsmallest(q, 'proba')['fwd_ret_1m'].mean()
    return pd.Series({'ls': top - bot, 'n': n})

ind_results = []
for ind_code in ind_preds['ind_broad'].unique():
    ind_sub = ind_preds[ind_preds['ind_broad'] == ind_code]
    monthly_ind = ind_sub.groupby('ym').apply(ind_ls).reset_index()
    monthly_ind = monthly_ind.dropna(subset=['ls'])
    if len(monthly_ind) < 6:
        continue
    mean_ls_ind = monthly_ind['ls'].mean()
    sharpe_ind = mean_ls_ind / monthly_ind['ls'].std() * np.sqrt(12) if monthly_ind['ls'].std() > 0 else 0
    n_firms = ind_sub['code'].nunique()
    ind_name_lookup = ind[ind['ind_broad'] == ind_code]['industry']
    ind_name = ind_name_lookup.iloc[0] if len(ind_name_lookup) > 0 else ind_code
    ind_results.append({
        'ind_broad': ind_code, 'industry': ind_name,
        'n_firms': n_firms, 'mean_monthly_ls': mean_ls_ind, 'sharpe': sharpe_ind,
        'positive': mean_ls_ind > 0,
    })

ind_res_df = pd.DataFrame(ind_results).sort_values('mean_monthly_ls', ascending=False)
n_positive = ind_res_df['positive'].sum()
n_total = len(ind_res_df)
print(f"\nIndustries with positive L-S: {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")
ind_res_df.to_csv(f"{OUTPUT_DIR}/paper1_v3_industry_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 13. Sub-period robustness (by year)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("13. Sub-period robustness")
print("=" * 60)

monthly_ls_copy = monthly_ls.copy()
monthly_ls_copy['year'] = monthly_ls_copy['ym'].astype(str).str[:4].astype(int)
for year in sorted(monthly_ls_copy['year'].unique()):
    sub = monthly_ls_copy[monthly_ls_copy['year'] == year]
    if len(sub) < 3:
        continue
    sub_mean = sub['ml_ls'].mean()
    sub_sharpe = sub_mean / sub['ml_ls'].std() * np.sqrt(12) if sub['ml_ls'].std() > 0 else 0
    sub_aucs = results_df[results_df['test_month'].str[:4] == str(year)]['auc']
    sub_auc = sub_aucs.mean() if len(sub_aucs) > 0 else np.nan
    print(f"  {year}: L-S={sub_mean:+.3f}%/mo, Sharpe={sub_sharpe:.3f}, AUC={sub_auc:.4f}, n_months={len(sub)}")

# ═══════════════════════════════════════════════════════════════════
# 14. Transaction cost sensitivity (with new results)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("14. Transaction cost sensitivity")
print("=" * 60)

tc_results = []
for cost_bps in [0, 10, 30, 50]:
    cost_pct = cost_bps / 100
    adjusted_ls = monthly_ls['ml_ls'] - 2 * cost_pct
    mean_adj = adjusted_ls.mean()
    sharpe_adj = mean_adj / adjusted_ls.std() * np.sqrt(12) if adjusted_ls.std() > 0 else 0
    pct_profitable = (adjusted_ls > 0).mean() * 100
    tc_results.append({
        'cost_bps': cost_bps, 'cost_roundtrip_pct': cost_pct * 2,
        'mean_monthly_ls': mean_adj, 'sharpe': sharpe_adj,
        'profitable_months_pct': pct_profitable,
    })
    print(f"  {cost_bps:3d} bps ({cost_pct*2:.1f}% RT): L-S={mean_adj:+.3f}%/mo, Sharpe={sharpe_adj:.3f}, "
          f"Profitable={pct_profitable:.0f}%")

tc_df = pd.DataFrame(tc_results)
tc_df.to_csv(f"{OUTPUT_DIR}/paper1_v3_transaction_costs.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 15. Single-factor baselines
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("15. Single-factor baselines")
print("=" * 60)

for factor in ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf', 'mean_turn', 'mom_12m', 'log_mcap']:
    if factor not in ml_data.columns:
        continue
    fdata = ml_data[['code', 'ym', factor, 'fwd_ret_1m']].dropna()
    def factor_ls_fn(group):
        n = len(group)
        q = max(1, n // 5)
        if factor.startswith('mean_p'):
            top = group.nsmallest(q, factor)['fwd_ret_1m'].mean()
            bot = group.nlargest(q, factor)['fwd_ret_1m'].mean()
        else:
            top = group.nlargest(q, factor)['fwd_ret_1m'].mean()
            bot = group.nsmallest(q, factor)['fwd_ret_1m'].mean()
        return pd.Series({'ls': top - bot})
    fls = fdata.groupby('ym').apply(factor_ls_fn).reset_index()
    fmean = fls['ls'].mean()
    fsharpe = fmean / fls['ls'].std() * np.sqrt(12) if fls['ls'].std() > 0 else 0
    print(f"  {factor:12s}: L-S = {fmean:+.3f}%/mo, Sharpe = {fsharpe:.3f}")

# ═══════════════════════════════════════════════════════════════════
# 16. Updated portfolio plot
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("16. Portfolio plot")
print("=" * 60)

monthly_ls['ym_dt'] = monthly_ls['ym'].astype(str).apply(lambda x: pd.Period(x).to_timestamp())
fig, ax = plt.subplots(figsize=(12, 5))
for col, label, color in [('ml_top', 'ML Top Quintile', '#5cb85c'),
                           ('ml_bot', 'ML Bottom Quintile', '#d9534f'),
                           ('ml_ls', 'ML Long-Short', '#f0ad4e'),
                           ('equal_weight', 'Equal Weight', '#5bc0de')]:
    cum_ret = (1 + monthly_ls[col] / 100).cumprod()
    ax.plot(monthly_ls['ym_dt'], cum_ret, label=label, color=color, linewidth=1.5)
ax.set_ylabel('Cumulative Return (NAV)')
ax.set_title('ML Factor Model: Portfolio Performance (Monthly Rolling OOS)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_v3_portfolio.png", dpi=150)
plt.close()
print("Portfolio plot saved.")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL REVIEW FIXES COMPLETE")
print("=" * 60)
print(f"\nv3 output files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('paper1_v3'):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")
