"""
Paper 1: Full analysis pipeline with ROE, industry codes, and extended analyses.
Builds on 07_paper1_ml_pipeline.py but adds:
  - ROE and net profit margin as features
  - Industry codes for transferability analysis
  - Sub-period robustness
  - Transaction cost sensitivity
  - Single-factor baselines
  - Descriptive statistics for the paper
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
# 1. Load daily factors and build monthly panel
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Loading daily factors and building monthly panel")
print("=" * 60)

daily = pd.read_csv(f"{OUTPUT_DIR}/paper1_daily_factors.csv", dtype=str)
print(f"Raw daily: {daily.shape}")

numeric_cols = ['close', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
for col in numeric_cols:
    daily[col] = pd.to_numeric(daily[col], errors='coerce')
daily['isST'] = daily['isST'].map({'1': 1, '0': 0, 1: 1, 0: 0}).fillna(0).astype(int)
daily['date'] = pd.to_datetime(daily['date'])

# Filter ST stocks
daily = daily[daily['isST'] == 0].copy()
print(f"After removing ST: {daily.shape}")

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
    mean_volume=('volume', 'mean'),
    n_days=('pctChg', 'count'),
).reset_index()

monthly = monthly[monthly['n_days'] >= 10].copy()
print(f"Monthly panel: {monthly.shape}, stocks: {monthly['code'].nunique()}, months: {monthly['ym'].nunique()}")

# Momentum features
monthly = monthly.sort_values(['code', 'ym'])
for window in [3, 6, 12]:
    monthly[f'mom_{window}m'] = monthly.groupby('code')['ret_month'].transform(
        lambda x: x.rolling(window).sum()
    )

# Forward return (target)
monthly['fwd_ret_1m'] = monthly.groupby('code')['ret_month'].shift(-1)
monthly['fwd_outperform'] = monthly.groupby('ym')['fwd_ret_1m'].transform(
    lambda x: (x > x.median()).astype(int)
)

# ═══════════════════════════════════════════════════════════════════
# 2. Merge quarterly fundamentals
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Merging quarterly fundamentals")
print("=" * 60)

qtr = pd.read_csv(f"{OUTPUT_DIR}/paper1_quarterly_fundamentals.csv", dtype=str)
for col in ['roeAvg', 'npMargin', 'gpMargin', 'epsTTM']:
    qtr[col] = pd.to_numeric(qtr[col], errors='coerce')
qtr['statDate'] = pd.to_datetime(qtr['statDate'], errors='coerce')
qtr['ym'] = qtr['statDate'].dt.to_period('M')
qtr_agg = qtr.groupby(['code', 'ym']).agg(
    roeAvg=('roeAvg', 'last'),
    npMargin=('npMargin', 'last'),
).reset_index()
qtr_agg = qtr_agg.sort_values(['code', 'ym'])
monthly = monthly.merge(qtr_agg, on=['code', 'ym'], how='left')
monthly[['roeAvg', 'npMargin']] = monthly.groupby('code')[['roeAvg', 'npMargin']].ffill()
n_roe = monthly['roeAvg'].notna().sum()
print(f"Merged quarterly: {n_roe}/{len(monthly)} rows have ROE ({n_roe/len(monthly)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════
# 3. Merge industry codes
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Merging industry codes")
print("=" * 60)

ind = pd.read_csv(f"{OUTPUT_DIR}/paper1_industry_codes.csv")
# Extract broad industry code (first letter + 2 digits)
ind['ind_broad'] = ind['industry'].str[:3]
ind = ind[['code', 'industry', 'ind_broad']].drop_duplicates(subset='code', keep='last')
monthly = monthly.merge(ind, on='code', how='left')
n_ind = monthly['industry'].notna().sum()
print(f"Merged industry: {n_ind}/{len(monthly)} rows ({n_ind/len(monthly)*100:.1f}%)")
print(f"Unique industries (broad): {monthly['ind_broad'].nunique()}")

# ═══════════════════════════════════════════════════════════════════
# 4. Prepare ML features
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Preparing ML features")
print("=" * 60)

feature_cols = ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf', 'mean_turn',
                'vol_month', 'mom_3m', 'mom_6m', 'mom_12m', 'ret_month']

# Add ROE and npMargin
if monthly['roeAvg'].notna().mean() > 0.3:
    feature_cols.extend(['roeAvg', 'npMargin'])
    print("Including ROE and npMargin")

ml_data = monthly.dropna(subset=['fwd_outperform']).copy()
ml_data = ml_data.dropna(subset=feature_cols, thresh=len(feature_cols) - 2)

for col in feature_cols:
    ml_data[col] = ml_data.groupby('ym')[col].transform(lambda x: x.fillna(x.median()))
ml_data = ml_data.dropna(subset=feature_cols)

print(f"ML dataset: {ml_data.shape}, stocks: {ml_data['code'].nunique()}, months: {ml_data['ym'].nunique()}")
print(f"Features: {feature_cols}")

# ═══════════════════════════════════════════════════════════════════
# 5. Descriptive statistics (for paper Table 3.1)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Descriptive statistics")
print("=" * 60)

desc_cols = feature_cols + ['fwd_ret_1m']
desc = ml_data[desc_cols].describe().T[['count', 'mean', 'std', 'min', 'max', '50%']]
desc.columns = ['Obs', 'Mean', 'Std', 'Min', 'Max', 'Median']
desc['Obs'] = desc['Obs'].astype(int)
print(desc.round(4).to_string())
desc.round(4).to_csv(f"{OUTPUT_DIR}/paper1_descriptive_stats.csv")

# ═══════════════════════════════════════════════════════════════════
# 6. XGBoost walk-forward CV
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. XGBoost walk-forward CV")
print("=" * 60)

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

unique_months = sorted(ml_data['ym'].unique())
n_months = len(unique_months)
train_window = 60
test_window = 12

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    from sklearn.ensemble import GradientBoostingClassifier

def make_model():
    if has_xgb:
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbosity=0, n_jobs=-1)
    else:
        return GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, random_state=42)

fold_results = []
all_predictions = []

for start in range(0, n_months - train_window - test_window, test_window):
    train_months = unique_months[start:start + train_window]
    test_months = unique_months[start + train_window:start + train_window + test_window]

    train_mask = ml_data['ym'].isin(train_months)
    test_mask = ml_data['ym'].isin(test_months)

    X_train = ml_data.loc[train_mask, feature_cols].values
    y_train = ml_data.loc[train_mask, 'fwd_outperform'].values
    X_test = ml_data.loc[test_mask, feature_cols].values
    y_test = ml_data.loc[test_mask, 'fwd_outperform'].values

    if len(y_test) == 0 or len(np.unique(y_train)) < 2:
        continue

    model = make_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, (proba > 0.5).astype(int))
    ap = average_precision_score(y_test, proba)

    fold_results.append({
        'train_start': str(train_months[0]), 'train_end': str(train_months[-1]),
        'test_start': str(test_months[0]), 'test_end': str(test_months[-1]),
        'auc': auc, 'accuracy': acc, 'avg_precision': ap,
        'n_train': len(y_train), 'n_test': len(y_test),
    })

    test_preds = ml_data.loc[test_mask, ['code', 'ym', 'fwd_ret_1m', 'fwd_outperform']].copy()
    test_preds['proba'] = proba
    if 'industry' in ml_data.columns:
        test_preds['industry'] = ml_data.loc[test_mask, 'industry'].values
        test_preds['ind_broad'] = ml_data.loc[test_mask, 'ind_broad'].values
    all_predictions.append(test_preds)

    print(f"  Fold {len(fold_results)}: {train_months[0]}–{train_months[-1]} → "
          f"{test_months[0]}–{test_months[-1]}, AUC={auc:.4f}, Acc={acc:.4f}")

results_df = pd.DataFrame(fold_results)
print(f"\nOverall: mean AUC={results_df['auc'].mean():.4f}, mean Acc={results_df['accuracy'].mean():.4f}")
results_df.to_csv(f"{OUTPUT_DIR}/paper1_v2_cv_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 6b. Logistic Regression baseline (same rolling-window protocol)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6b. Logistic Regression baseline (walk-forward)")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

lr_fold_results = []
lr_all_predictions = []

for start in range(0, n_months - train_window - test_window, test_window):
    train_months = unique_months[start:start + train_window]
    test_months = unique_months[start + train_window:start + train_window + test_window]

    train_mask = ml_data['ym'].isin(train_months)
    test_mask = ml_data['ym'].isin(test_months)

    X_train = ml_data.loc[train_mask, feature_cols].values
    y_train = ml_data.loc[train_mask, 'fwd_outperform'].values
    X_test = ml_data.loc[test_mask, feature_cols].values
    y_test = ml_data.loc[test_mask, 'fwd_outperform'].values

    if len(y_test) == 0 or len(np.unique(y_train)) < 2:
        continue

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr.fit(X_train_s, y_train)
    proba_lr = lr.predict_proba(X_test_s)[:, 1]

    auc_lr = roc_auc_score(y_test, proba_lr)
    acc_lr = accuracy_score(y_test, (proba_lr > 0.5).astype(int))

    lr_fold_results.append({
        'train_start': str(train_months[0]), 'train_end': str(train_months[-1]),
        'test_start': str(test_months[0]), 'test_end': str(test_months[-1]),
        'auc': auc_lr, 'accuracy': acc_lr,
        'n_train': len(y_train), 'n_test': len(y_test),
    })

    test_preds_lr = ml_data.loc[test_mask, ['code', 'ym', 'fwd_ret_1m', 'fwd_outperform']].copy()
    test_preds_lr['proba'] = proba_lr
    lr_all_predictions.append(test_preds_lr)

lr_results_df = pd.DataFrame(lr_fold_results)
lr_mean_auc = lr_results_df['auc'].mean()
print(f"Logistic Regression — mean AUC: {lr_mean_auc:.4f}")

lr_preds_all = pd.concat(lr_all_predictions, ignore_index=True)
lr_monthly_ls = lr_preds_all.groupby('ym').apply(compute_ls).reset_index()
lr_mean_ls = lr_monthly_ls['ml_ls'].mean()
lr_sharpe = lr_mean_ls / lr_monthly_ls['ml_ls'].std() * np.sqrt(12)
print(f"Logistic Regression — L-S: {lr_mean_ls:+.3f}%/mo, Sharpe: {lr_sharpe:.3f}")
print(f"\n>>> TABLE VALUES for paper1_ieee.tex Table 3 (tab:model_comp):")
print(f"    Logistic Regression | AUC: {lr_mean_auc:.3f} | L-S: {lr_mean_ls:+.2f}%/mo | Sharpe: {lr_sharpe:.2f}")

lr_results_df.to_csv(f"{OUTPUT_DIR}/paper1_lr_cv_results.csv", index=False)
lr_monthly_ls.to_csv(f"{OUTPUT_DIR}/paper1_lr_portfolio_returns.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 7. SHAP analysis
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. SHAP feature importance")
print("=" * 60)

X_all = ml_data[feature_cols].values
y_all = ml_data['fwd_outperform'].values

final_model = make_model()
final_model.n_estimators = 300
final_model.fit(X_all, y_all)

try:
    import shap
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
    })

    # Compute category shares
    total_shap = shap_df['mean_abs_shap'].sum()
    cat_shares = shap_df.groupby('category')['mean_abs_shap'].sum() / total_shap * 100
    print("\nSHAP by category:")
    for cat, share in cat_shares.sort_values(ascending=False).items():
        print(f"  {cat}: {share:.1f}%")

    print("\nSHAP feature importance:")
    print(shap_df.to_string(index=False))
    shap_df.to_csv(f"{OUTPUT_DIR}/paper1_v2_shap_importance.csv", index=False)

    # SHAP plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_all[sample_idx], feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_v2_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved.")
except ImportError:
    print("shap not installed — skipping")

# ═══════════════════════════════════════════════════════════════════
# 8. Portfolio returns and baselines
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. Portfolio returns")
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

# Single-factor baselines
print("\nSingle-factor baselines (top vs bottom quintile):")
factor_baselines = {}
for factor in ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf', 'mean_turn', 'mom_12m']:
    if factor not in ml_data.columns:
        continue
    # Merge factor into predictions
    factor_data = ml_data[['code', 'ym', factor, 'fwd_ret_1m']].dropna()

    def factor_ls(group):
        n = len(group)
        q = max(1, n // 5)
        # For valuation: low = cheap = long; for momentum/turnover: high = long
        if factor.startswith('mean_p'):  # valuation — low is good
            top = group.nsmallest(q, factor)['fwd_ret_1m'].mean()
            bot = group.nlargest(q, factor)['fwd_ret_1m'].mean()
        else:  # behavioral — high is good
            top = group.nlargest(q, factor)['fwd_ret_1m'].mean()
            bot = group.nsmallest(q, factor)['fwd_ret_1m'].mean()
        return pd.Series({'ls': top - bot})

    fls = factor_data.groupby('ym').apply(factor_ls).reset_index()
    mean_ls = fls['ls'].mean()
    sharpe = fls['ls'].mean() / fls['ls'].std() * np.sqrt(12) if fls['ls'].std() > 0 else 0
    factor_baselines[factor] = {'mean_monthly_ls': mean_ls, 'sharpe': sharpe}
    print(f"  {factor:12s}: L-S = {mean_ls:+.3f}%/mo, Sharpe = {sharpe:.3f}")

# ML performance
ml_mean = monthly_ls['ml_ls'].mean()
ml_sharpe = monthly_ls['ml_ls'].mean() / monthly_ls['ml_ls'].std() * np.sqrt(12)
ml_top_mean = monthly_ls['ml_top'].mean()
ml_bot_mean = monthly_ls['ml_bot'].mean()
ew_mean = monthly_ls['equal_weight'].mean()

print(f"\nML Long-Short:")
print(f"  Mean monthly L-S: {ml_mean:+.4f}%")
print(f"  Mean top quintile: {ml_top_mean:+.4f}%")
print(f"  Mean bottom quintile: {ml_bot_mean:+.4f}%")
print(f"  Equal-weight: {ew_mean:+.4f}%")
print(f"  L-S Sharpe (ann): {ml_sharpe:.3f}")

# Max drawdown
cum = (1 + monthly_ls['ml_ls'] / 100).cumprod()
peak = cum.expanding().max()
dd = (cum - peak) / peak
print(f"  Max drawdown: {dd.min()*100:.1f}%")
print(f"  Profitable months: {(monthly_ls['ml_ls'] > 0).mean()*100:.0f}%")

monthly_ls.to_csv(f"{OUTPUT_DIR}/paper1_v2_portfolio_returns.csv", index=False)
pd.DataFrame(factor_baselines).T.to_csv(f"{OUTPUT_DIR}/paper1_v2_factor_baselines.csv")

# ═══════════════════════════════════════════════════════════════════
# 9. Ablation study
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. Ablation study")
print("=" * 60)

ablation_groups = {
    'No valuation (PE/PB/PS/PCF)': ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf'],
    'No momentum': ['mom_3m', 'mom_6m', 'mom_12m'],
    'No turnover': ['mean_turn'],
    'No volatility': ['vol_month'],
    'No current return': ['ret_month'],
    'No ROE/margins': ['roeAvg', 'npMargin'],
}

ablation_results = [{'config': 'Full model', 'mean_auc': results_df['auc'].mean()}]

for abl_name, remove_cols in ablation_groups.items():
    abl_features = [f for f in feature_cols if f not in remove_cols]
    if len(abl_features) == 0:
        continue
    abl_aucs = []
    for start in range(0, n_months - train_window - test_window, test_window):
        train_months = unique_months[start:start + train_window]
        test_months = unique_months[start + train_window:start + train_window + test_window]
        train_mask = ml_data['ym'].isin(train_months)
        test_mask = ml_data['ym'].isin(test_months)
        X_tr = ml_data.loc[train_mask, abl_features].values
        y_tr = ml_data.loc[train_mask, 'fwd_outperform'].values
        X_te = ml_data.loc[test_mask, abl_features].values
        y_te = ml_data.loc[test_mask, 'fwd_outperform'].values
        if len(y_te) == 0 or len(np.unique(y_tr)) < 2:
            continue
        m = make_model()
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        abl_aucs.append(roc_auc_score(y_te, proba))
    mean_auc = np.mean(abl_aucs) if abl_aucs else np.nan
    delta = mean_auc - results_df['auc'].mean()
    ablation_results.append({'config': abl_name, 'mean_auc': mean_auc, 'delta': delta})
    print(f"  {abl_name}: AUC={mean_auc:.4f} (delta={delta:+.4f})")

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(f"{OUTPUT_DIR}/paper1_v2_ablation.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 10. Industry transferability
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. Industry transferability")
print("=" * 60)

if 'ind_broad' in preds_all.columns:
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
        mean_ls = monthly_ind['ls'].mean()
        sharpe = mean_ls / monthly_ind['ls'].std() * np.sqrt(12) if monthly_ind['ls'].std() > 0 else 0
        n_firms = ind_sub['code'].nunique()
        # Get full industry name
        ind_name = ind[ind['ind_broad'] == ind_code]['industry'].iloc[0] if len(ind[ind['ind_broad'] == ind_code]) > 0 else ind_code
        ind_results.append({
            'ind_broad': ind_code, 'industry': ind_name,
            'n_firms': n_firms, 'mean_monthly_ls': mean_ls, 'sharpe': sharpe,
            'positive': mean_ls > 0,
        })

    ind_df = pd.DataFrame(ind_results).sort_values('mean_monthly_ls', ascending=False)
    n_positive = ind_df['positive'].sum()
    n_total = len(ind_df)
    print(f"\nIndustries with positive L-S: {n_positive}/{n_total} ({n_positive/n_total*100:.0f}%)")
    print(f"\nTop 5:")
    print(ind_df.head().to_string(index=False))
    print(f"\nBottom 5:")
    print(ind_df.tail().to_string(index=False))
    ind_df.to_csv(f"{OUTPUT_DIR}/paper1_v2_industry_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 11. Sub-period robustness
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("11. Sub-period robustness")
print("=" * 60)

for fold in fold_results:
    test_start = fold['test_start']
    test_end = fold['test_end']
    test_preds = preds_all[(preds_all['ym'].astype(str) >= test_start) &
                           (preds_all['ym'].astype(str) <= test_end)]
    if len(test_preds) == 0:
        continue
    sub_ls = test_preds.groupby('ym').apply(compute_ls).reset_index()
    mean_ls = sub_ls['ml_ls'].mean()
    sharpe = mean_ls / sub_ls['ml_ls'].std() * np.sqrt(12) if sub_ls['ml_ls'].std() > 0 else 0
    print(f"  {test_start} to {test_end}: L-S={mean_ls:+.3f}%/mo, Sharpe={sharpe:.3f}, AUC={fold['auc']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 12. Transaction cost sensitivity
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("12. Transaction cost sensitivity")
print("=" * 60)

# Estimate monthly turnover in top/bottom quintile
tc_results = []
for cost_bps in [0, 10, 30, 50]:
    cost_pct = cost_bps / 100  # convert bps to %
    # Approximate: each month we turn over ~100% of the quintile portfolio
    # So cost = 2 * cost_pct (buy + sell) per month, applied to L-S
    # More conservative: assume 100% turnover each side
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
tc_df.to_csv(f"{OUTPUT_DIR}/paper1_v2_transaction_costs.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 13. Cumulative return plot (updated)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("13. Saving updated plots")
print("=" * 60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

monthly_ls['ym_dt'] = monthly_ls['ym'].astype(str).apply(lambda x: pd.Period(x).to_timestamp())
fig, ax = plt.subplots(figsize=(12, 5))
for col, label, color in [('ml_top', 'ML Top Quintile', '#5cb85c'),
                           ('ml_bot', 'ML Bottom Quintile', '#d9534f'),
                           ('ml_ls', 'ML Long-Short', '#f0ad4e'),
                           ('equal_weight', 'Equal Weight', '#5bc0de')]:
    cum = (1 + monthly_ls[col] / 100).cumprod()
    ax.plot(monthly_ls['ym_dt'], cum, label=label, color=color, linewidth=1.5)
ax.set_ylabel('Cumulative Return (NAV)')
ax.set_title('ML Factor Model: Portfolio Performance (Walk-Forward OOS)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_v2_portfolio.png", dpi=150)
plt.close()
print("Portfolio plot saved.")

# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FULL ANALYSIS PIPELINE COMPLETE")
print("=" * 60)
print(f"\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith('paper1_v2'):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")
