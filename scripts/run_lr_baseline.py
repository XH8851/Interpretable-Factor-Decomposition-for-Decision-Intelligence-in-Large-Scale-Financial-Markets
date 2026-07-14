"""Standalone script: logistic regression walk-forward baseline for paper1_ieee.tex Table 3."""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

print("Loading daily factors (695 MB) …")
daily = pd.read_csv(f"{OUTPUT_DIR}/paper1_daily_factors.csv",
                    usecols=['code','date','close','turn','pctChg',
                             'peTTM','pbMRQ','psTTM','pcfNcfTTM','isST'],
                    dtype={'isST': str})

for col in ['close','turn','pctChg','peTTM','pbMRQ','psTTM','pcfNcfTTM']:
    daily[col] = pd.to_numeric(daily[col], errors='coerce')
daily['isST'] = daily['isST'].map({'1':1,'0':0}).fillna(0).astype(int)
daily = daily[daily['isST'] == 0].copy()
daily['date'] = pd.to_datetime(daily['date'])
daily['ym'] = daily['date'].dt.to_period('M')
print(f"  daily rows after ST filter: {len(daily):,}")

print("Building monthly panel …")
monthly = daily.groupby(['code','ym']).agg(
    ret_month=('pctChg', lambda x: ((1 + x/100).prod() - 1)*100),
    mean_pe=('peTTM','mean'), mean_pb=('pbMRQ','mean'),
    mean_ps=('psTTM','mean'), mean_pcf=('pcfNcfTTM','mean'),
    mean_turn=('turn','mean'), vol_month=('pctChg','std'),
    last_close=('close','last'), n_days=('pctChg','count'),
).reset_index()
monthly = monthly[monthly['n_days'] >= 10].copy()
monthly = monthly.sort_values(['code','ym'])
for w in [3, 6, 12]:
    monthly[f'mom_{w}m'] = monthly.groupby('code')['ret_month'].transform(
        lambda x: x.rolling(w).sum())
monthly['fwd_ret_1m'] = monthly.groupby('code')['ret_month'].shift(-1)
monthly['fwd_outperform'] = monthly.groupby('ym')['fwd_ret_1m'].transform(
    lambda x: (x > x.median()).astype(int))

# Merge quarterly fundamentals
qtr = pd.read_csv(f"{OUTPUT_DIR}/paper1_quarterly_fundamentals.csv",
                  usecols=['code','statDate','roeAvg','npMargin'], dtype=str)
for col in ['roeAvg','npMargin']:
    qtr[col] = pd.to_numeric(qtr[col], errors='coerce')
qtr['ym'] = pd.to_datetime(qtr['statDate'], errors='coerce').dt.to_period('M')
qtr_agg = qtr.groupby(['code','ym']).agg(roeAvg=('roeAvg','last'), npMargin=('npMargin','last')).reset_index()
qtr_agg = qtr_agg.sort_values(['code','ym'])
monthly = monthly.merge(qtr_agg, on=['code','ym'], how='left')
monthly[['roeAvg','npMargin']] = monthly.groupby('code')[['roeAvg','npMargin']].ffill()

# Size feature (log mcap): use last_close as proxy (no shares outstanding available standalone)
# Match script 08: log_mcap was computed from close * total_shares; use close as size proxy
monthly['log_close'] = np.log(monthly['last_close'].clip(lower=0.01))

feature_cols = ['mean_pe','mean_pb','mean_ps','mean_pcf','mean_turn',
                'vol_month','mom_3m','mom_6m','mom_12m','ret_month']
if monthly['roeAvg'].notna().mean() > 0.3:
    feature_cols += ['roeAvg','npMargin']

ml_data = monthly.dropna(subset=['fwd_outperform']).copy()
ml_data = ml_data.dropna(subset=feature_cols, thresh=len(feature_cols)-2)
for col in feature_cols:
    ml_data[col] = ml_data.groupby('ym')[col].transform(lambda x: x.fillna(x.median()))
ml_data = ml_data.dropna(subset=feature_cols)

# Winsorize
for col in feature_cols:
    ml_data[col] = ml_data.groupby('ym')[col].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

print(f"  ML dataset: {ml_data.shape}, stocks: {ml_data['code'].nunique()}, months: {ml_data['ym'].nunique()}")

# Walk-forward LR
print("Running logistic regression walk-forward CV …")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

unique_months = sorted(ml_data['ym'].unique())
n_months = len(unique_months)
train_window, test_window = 60, 1

lr_folds, lr_preds_list = [], []

for start in range(train_window, n_months - test_window + 1):
    train_months = unique_months[start - train_window:start]
    test_month = unique_months[start]

    train_mask = ml_data['ym'].isin(train_months)
    test_mask  = ml_data['ym'] == test_month

    X_tr = ml_data.loc[train_mask, feature_cols].values
    y_tr = ml_data.loc[train_mask, 'fwd_outperform'].values
    X_te = ml_data.loc[test_mask,  feature_cols].values
    y_te = ml_data.loc[test_mask,  'fwd_outperform'].values

    if len(y_te) == 0 or len(np.unique(y_tr)) < 2:
        continue

    sc = StandardScaler()
    lr = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
    lr.fit(sc.fit_transform(X_tr), y_tr)
    proba = lr.predict_proba(sc.transform(X_te))[:, 1]

    auc = roc_auc_score(y_te, proba)
    lr_folds.append({'test_month': str(test_month), 'auc': auc, 'n_test': len(y_te)})

    preds = ml_data.loc[test_mask, ['code','ym','fwd_ret_1m']].copy()
    preds['proba'] = proba
    lr_preds_list.append(preds)

    if len(lr_folds) % 10 == 0:
        print(f"  fold {len(lr_folds)}: {test_month}, AUC={auc:.4f}")

lr_df = pd.DataFrame(lr_folds)
lr_preds = pd.concat(lr_preds_list, ignore_index=True)

def compute_ls(group):
    n = len(group); q = max(1, n//5)
    top = group.nlargest(q,'proba')['fwd_ret_1m'].mean()
    bot = group.nsmallest(q,'proba')['fwd_ret_1m'].mean()
    return pd.Series({'ml_top':top,'ml_bot':bot,'ml_ls':top-bot,
                      'equal_weight':group['fwd_ret_1m'].mean()})

monthly_ls = lr_preds.groupby('ym').apply(compute_ls).reset_index()

mean_auc  = lr_df['auc'].mean()
mean_ls   = monthly_ls['ml_ls'].mean()
sharpe    = mean_ls / monthly_ls['ml_ls'].std() * np.sqrt(12)
n_months_oos = len(lr_df)

lr_df.to_csv(f"{OUTPUT_DIR}/paper1_lr_cv_results.csv", index=False)
monthly_ls.to_csv(f"{OUTPUT_DIR}/paper1_lr_portfolio_returns.csv", index=False)

print("\n" + "="*60)
print("LOGISTIC REGRESSION BASELINE RESULTS")
print("="*60)
print(f"  OOS months : {n_months_oos}")
print(f"  Mean AUC   : {mean_auc:.3f}")
print(f"  Mean L-S   : {mean_ls:+.2f} %/mo")
print(f"  Ann Sharpe : {sharpe:.2f}")
print()
print(">>> COPY THESE INTO paper1_ieee.tex (tab:model_comp):")
print(f"    Logistic Regression & {mean_auc:.3f} & ${mean_ls:+.2f}\\%$ & {sharpe:.2f} \\\\")
