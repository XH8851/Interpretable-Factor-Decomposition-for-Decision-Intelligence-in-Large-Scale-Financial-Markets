"""
Paper 1: ML/SHAP pipeline for interpretable factor decomposition.

Builds monthly firm×factor panel from daily data, trains XGBoost to predict
next-month outperformance, and runs SHAP for factor importance.

Designed to work with daily factors immediately; quarterly ROE and industry
codes are merged when available.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
import os
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

# Month-end label
daily['ym'] = daily['date'].dt.to_period('M')

# Monthly aggregation per stock
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

# Filter months with too few trading days
monthly = monthly[monthly['n_days'] >= 10].copy()
print(f"Monthly panel: {monthly.shape}, stocks: {monthly['code'].nunique()}, months: {monthly['ym'].nunique()}")

# ── Momentum features ────────────────────────────────────────────
# 1-month return is ret_month itself
# 3-month, 6-month, 12-month momentum
monthly = monthly.sort_values(['code', 'ym'])
for window in [3, 6, 12]:
    monthly[f'mom_{window}m'] = monthly.groupby('code')['ret_month'].transform(
        lambda x: x.rolling(window).sum()
    )

# ── Forward return (target) ──────────────────────────────────────
monthly['fwd_ret_1m'] = monthly.groupby('code')['ret_month'].shift(-1)

# Cross-sectional rank: outperform = above median
monthly['fwd_outperform'] = monthly.groupby('ym')['fwd_ret_1m'].transform(
    lambda x: (x > x.median()).astype(int)
)

# ═══════════════════════════════════════════════════════════════════
# 2. Merge quarterly ROE if available
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Merging quarterly fundamentals (if available)")
print("=" * 60)

quarterly_path = f"{OUTPUT_DIR}/paper1_quarterly_fundamentals.csv"
if os.path.exists(quarterly_path):
    qtr = pd.read_csv(quarterly_path, dtype=str)
    for col in ['roeAvg', 'npMargin', 'gpMargin', 'epsTTM']:
        qtr[col] = pd.to_numeric(qtr[col], errors='coerce')
    qtr['statDate'] = pd.to_datetime(qtr['statDate'], errors='coerce')
    qtr['ym'] = qtr['statDate'].dt.to_period('M')
    qtr_agg = qtr.groupby(['code', 'ym']).agg(
        roeAvg=('roeAvg', 'last'),
        npMargin=('npMargin', 'last'),
    ).reset_index()
    # Forward-fill quarterly data to monthly
    qtr_agg = qtr_agg.sort_values(['code', 'ym'])
    monthly = monthly.merge(qtr_agg, on=['code', 'ym'], how='left')
    monthly[['roeAvg', 'npMargin']] = monthly.groupby('code')[['roeAvg', 'npMargin']].ffill()
    n_roe = monthly['roeAvg'].notna().sum()
    print(f"Merged quarterly: {n_roe}/{len(monthly)} rows have ROE ({n_roe/len(monthly)*100:.1f}%)")
else:
    monthly['roeAvg'] = np.nan
    monthly['npMargin'] = np.nan
    print("Quarterly file not found — proceeding without ROE")

# ═══════════════════════════════════════════════════════════════════
# 3. Merge industry codes if available
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Merging industry codes (if available)")
print("=" * 60)

industry_path = f"{OUTPUT_DIR}/paper1_industry_codes.csv"
if os.path.exists(industry_path):
    ind = pd.read_csv(industry_path)
    ind = ind[['code', 'industry']].drop_duplicates(subset='code', keep='last')
    monthly = monthly.merge(ind, on='code', how='left')
    n_ind = monthly['industry'].notna().sum()
    print(f"Merged industry: {n_ind}/{len(monthly)} rows ({n_ind/len(monthly)*100:.1f}%)")
    print(f"Unique industries: {monthly['industry'].nunique()}")
else:
    monthly['industry'] = np.nan
    print("Industry file not found — proceeding without")

# ═══════════════════════════════════════════════════════════════════
# 4. Prepare ML features and train XGBoost
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. XGBoost training with time-series CV")
print("=" * 60)

feature_cols = ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf', 'mean_turn',
                'vol_month', 'mom_3m', 'mom_6m', 'mom_12m', 'ret_month']

# Add ROE if available
if monthly['roeAvg'].notna().mean() > 0.3:
    feature_cols.extend(['roeAvg', 'npMargin'])
    print("Including ROE and npMargin")

# Drop rows with NaN target or all-NaN features
ml_data = monthly.dropna(subset=['fwd_outperform']).copy()
ml_data = ml_data.dropna(subset=feature_cols, thresh=len(feature_cols) - 2)

# Fill remaining NaN with cross-sectional median
for col in feature_cols:
    ml_data[col] = ml_data.groupby('ym')[col].transform(lambda x: x.fillna(x.median()))
ml_data = ml_data.dropna(subset=feature_cols)

print(f"ML dataset: {ml_data.shape}, months: {ml_data['ym'].nunique()}")
print(f"Features: {feature_cols}")
print(f"Target balance: {ml_data['fwd_outperform'].mean():.3f}")

# Time-series split (walk-forward)
unique_months = sorted(ml_data['ym'].unique())
n_months = len(unique_months)
print(f"Total months: {n_months}")

# Use 60-month training, 12-month test, rolling
train_window = 60
test_window = 12

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False
    print("WARNING: xgboost not installed. Using sklearn GradientBoosting.")
    from sklearn.ensemble import GradientBoostingClassifier

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

    if has_xgb:
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, (proba > 0.5).astype(int))
    ap = average_precision_score(y_test, proba)

    fold_results.append({
        'train_start': str(train_months[0]),
        'train_end': str(train_months[-1]),
        'test_start': str(test_months[0]),
        'test_end': str(test_months[-1]),
        'auc': auc, 'accuracy': acc, 'avg_precision': ap,
        'n_train': len(y_train), 'n_test': len(y_test),
    })

    test_preds = ml_data.loc[test_mask, ['code', 'ym', 'fwd_ret_1m', 'fwd_outperform']].copy()
    test_preds['proba'] = proba
    all_predictions.append(test_preds)

    print(f"  Fold {len(fold_results)}: train {train_months[0]}–{train_months[-1]}, "
          f"test {test_months[0]}–{test_months[-1]}, AUC={auc:.3f}, Acc={acc:.3f}")

results_df = pd.DataFrame(fold_results)
print(f"\nOverall: mean AUC={results_df['auc'].mean():.3f}, mean Acc={results_df['accuracy'].mean():.3f}")

results_df.to_csv(f"{OUTPUT_DIR}/paper1_ml_cv_results.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 5. SHAP analysis (on full model)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. SHAP feature importance")
print("=" * 60)

# Train final model on all data
X_all = ml_data[feature_cols].values
y_all = ml_data['fwd_outperform'].values

if has_xgb:
    final_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=-1
    )
else:
    final_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

final_model.fit(X_all, y_all)

try:
    import shap
    has_shap = True
except ImportError:
    has_shap = False
    print("WARNING: shap not installed. Using feature_importances_ instead.")

if has_shap:
    explainer = shap.TreeExplainer(final_model)
    sample_idx = np.random.RandomState(42).choice(len(X_all), min(5000, len(X_all)), replace=False)
    shap_values = explainer.shap_values(X_all[sample_idx])

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nSHAP feature importance:")
    print(shap_df.to_string(index=False))
    shap_df.to_csv(f"{OUTPUT_DIR}/paper1_shap_importance.csv", index=False)

    # SHAP summary plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_all[sample_idx], feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to {FIG_DIR}/fig_shap_summary.png")
else:
    importances = final_model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    print("\nFeature importance (gain):")
    print(imp_df.to_string(index=False))
    imp_df.to_csv(f"{OUTPUT_DIR}/paper1_feature_importance.csv", index=False)

# ═══════════════════════════════════════════════════════════════════
# 6. Baseline comparisons
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Baseline portfolio comparisons")
print("=" * 60)

if all_predictions:
    preds_all = pd.concat(all_predictions, ignore_index=True)

    # ML long-short: top quintile vs bottom quintile by predicted proba
    def ml_ls_return(group):
        n = len(group)
        q = max(1, n // 5)
        top = group.nlargest(q, 'proba')['fwd_ret_1m'].mean()
        bot = group.nsmallest(q, 'proba')['fwd_ret_1m'].mean()
        return pd.Series({'ml_top': top, 'ml_bot': bot, 'ml_ls': top - bot, 'equal_weight': group['fwd_ret_1m'].mean()})

    monthly_ls = preds_all.groupby('ym').apply(ml_ls_return).reset_index()

    print(f"\nML Long-Short (top vs bottom quintile):")
    print(f"  Mean monthly L-S return: {monthly_ls['ml_ls'].mean():.4f}%")
    print(f"  Mean top quintile:       {monthly_ls['ml_top'].mean():.4f}%")
    print(f"  Mean bottom quintile:    {monthly_ls['ml_bot'].mean():.4f}%")
    print(f"  Mean equal-weight:       {monthly_ls['equal_weight'].mean():.4f}%")
    print(f"  L-S Sharpe (monthly):    {monthly_ls['ml_ls'].mean() / monthly_ls['ml_ls'].std() * np.sqrt(12):.3f}")

    monthly_ls.to_csv(f"{OUTPUT_DIR}/paper1_ml_portfolio_returns.csv", index=False)

    # Cumulative return plot
    import matplotlib.pyplot as plt
    monthly_ls['ym_dt'] = monthly_ls['ym'].astype(str).apply(lambda x: pd.Period(x).to_timestamp())
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label, color in [('ml_top', 'ML Top Quintile', '#5cb85c'),
                               ('ml_bot', 'ML Bottom Quintile', '#d9534f'),
                               ('equal_weight', 'Equal Weight', '#5bc0de')]:
        cum = (1 + monthly_ls[col] / 100).cumprod()
        ax.plot(monthly_ls['ym_dt'], cum, label=label, color=color, linewidth=1.5)
    ax.set_ylabel('Cumulative Return (NAV)')
    ax.set_title('ML Factor Model: Portfolio Performance (Walk-Forward OOS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig_ml_portfolio.png", dpi=150)
    plt.close()
    print(f"Portfolio plot saved.")

# ═══════════════════════════════════════════════════════════════════
# 7. Ablation study
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. Ablation study")
print("=" * 60)

ablation_results = []
# Full model result
ablation_results.append({'config': 'Full model', 'mean_auc': results_df['auc'].mean()})

# Remove each feature group
ablation_groups = {
    'No valuation (PE/PB/PS/PCF)': ['mean_pe', 'mean_pb', 'mean_ps', 'mean_pcf'],
    'No momentum': ['mom_3m', 'mom_6m', 'mom_12m'],
    'No turnover': ['mean_turn'],
    'No volatility': ['vol_month'],
    'No current return': ['ret_month'],
}

if 'roeAvg' in feature_cols:
    ablation_groups['No ROE/margins'] = ['roeAvg', 'npMargin']

for ablation_name, remove_cols in ablation_groups.items():
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
        if has_xgb:
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
        else:
            m = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                            subsample=0.8, random_state=42)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        abl_aucs.append(roc_auc_score(y_te, proba))

    mean_auc = np.mean(abl_aucs) if abl_aucs else np.nan
    ablation_results.append({'config': ablation_name, 'mean_auc': mean_auc})
    print(f"  {ablation_name}: AUC={mean_auc:.3f} (Δ={mean_auc - results_df['auc'].mean():.3f})")

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(f"{OUTPUT_DIR}/paper1_ablation.csv", index=False)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
