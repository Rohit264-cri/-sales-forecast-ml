"""
train_model.py — Sales Forecasting ML Pipeline
Models: Linear Regression | Random Forest | XGBoost
Run: python src/train_model.py
"""
import pandas as pd, numpy as np, pickle, os, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from xgboost                 import XGBRegressor

DATA_PATH='data/sales_data.csv'; OUT='outputs'; os.makedirs(OUT,exist_ok=True)

print("="*60)
print("  SALES REVENUE FORECASTING — ML PIPELINE")
print("="*60)

df=pd.read_csv(DATA_PATH,parse_dates=['date'])
df=df.sort_values('date').reset_index(drop=True)

# Feature Engineering
df['time_idx']    = np.arange(len(df))
df['sin_week']    = np.sin(2*np.pi*df['week_of_year']/52)
df['cos_week']    = np.cos(2*np.pi*df['week_of_year']/52)
df['sin_month']   = np.sin(2*np.pi*df['month']/12)
df['cos_month']   = np.cos(2*np.pi*df['month']/12)
df['lag_1']       = df['sales'].shift(1)
df['lag_2']       = df['sales'].shift(2)
df['roll_4']      = df['sales'].shift(1).rolling(4).mean()
df=df.dropna().reset_index(drop=True)

FEATS=['time_idx','sin_week','cos_week','sin_month','cos_month',
       'marketing_spend','discount_pct','temperature','is_holiday',
       'quarter','year','lag_1','lag_2','roll_4']
X=df[FEATS]; y=df['sales']

# 80/20 time split
split=int(len(df)*0.8)
Xtr,Xte=X.iloc[:split],X.iloc[split:]
ytr,yte=y.iloc[:split],y.iloc[split:]
print(f"\n[1] Data: {len(df)} rows | Train={split} Test={len(df)-split}")

sc=StandardScaler()
Xtr_sc=sc.fit_transform(Xtr); Xte_sc=sc.transform(Xte)

models={'Linear Regression':(LinearRegression(),True),
        'Random Forest':(RandomForestRegressor(n_estimators=300,max_depth=10,random_state=42),False),
        'XGBoost':(XGBRegressor(n_estimators=300,learning_rate=.03,max_depth=5,
                                subsample=.8,colsample_bytree=.8,random_state=42,verbosity=0),False)}

results=[]; best=(0,'',None,False)
print(f"\n[2] Model Results:")
print(f"    {'Model':<22}{'MAE':>10}{'RMSE':>10}{'R²':>8}")
print("    "+"-"*52)

for name,(mdl,use_sc) in models.items():
    Xtr_=Xtr_sc if use_sc else Xtr; Xte_=Xte_sc if use_sc else Xte
    mdl.fit(Xtr_,ytr)
    p=mdl.predict(Xte_)
    mae=mean_absolute_error(yte,p); rmse=np.sqrt(mean_squared_error(yte,p)); r2=r2_score(yte,p)
    results.append({'Model':name,'MAE':round(mae,2),'RMSE':round(rmse,2),'R2':round(r2,4)})
    print(f"    {name:<22}{mae:>10,.0f}{rmse:>10,.0f}{r2:>8.4f}")
    if r2>best[0]: best=(r2,name,mdl,use_sc)

print(f"\n    Best: {best[1]} (R²={best[0]:.4f})")

# Predict on test set for chart
best_model=best[2]; Xte_best=Xte_sc if best[3] else Xte
test_preds=best_model.predict(Xte_best)
test_df=pd.DataFrame({'date':df['date'].iloc[split:].values,
                       'actual':yte.values,'predicted':test_preds})
test_df.to_csv(f'{OUT}/test_predictions.csv',index=False)

# 6-month rolling forecast
history=df.copy(); last=history.iloc[-1].copy()
forecast=[]; prev_sales=list(history['sales'].tail(4))
for i in range(24):
    fd=df['date'].max()+pd.Timedelta(weeks=i+1)
    row={'time_idx':len(df)+i,'sin_week':np.sin(2*np.pi*fd.isocalendar()[1]/52),
         'cos_week':np.cos(2*np.pi*fd.isocalendar()[1]/52),
         'sin_month':np.sin(2*np.pi*fd.month/12),
         'cos_month':np.cos(2*np.pi*fd.month/12),
         'marketing_spend':df['marketing_spend'].mean()*(1+.005*i),
         'discount_pct':df['discount_pct'].mean(),'temperature':df['temperature'].mean(),
         'is_holiday':1 if fd.month in [4,10,12] else 0,'quarter':fd.quarter,'year':fd.year,
         'lag_1':prev_sales[-1],'lag_2':prev_sales[-2],'roll_4':np.mean(prev_sales[-4:])}
    Xf=pd.DataFrame([row])[FEATS]
    if best[3]: Xf=sc.transform(Xf)
    pred=float(best_model.predict(Xf)[0])
    prev_sales.append(pred)
    forecast.append({'date':fd.date(),'predicted_sales':round(pred,2)})

fdf=pd.DataFrame(forecast)
fdf.to_csv(f'{OUT}/forecast_6months.csv',index=False)
pd.DataFrame(results).to_csv(f'{OUT}/model_comparison.csv',index=False)
pickle.dump(best_model,open(f'{OUT}/best_model.pkl','wb'))
pickle.dump(sc,open(f'{OUT}/scaler.pkl','wb'))
print(f"\n[3] Forecast Avg: Rs {fdf['predicted_sales'].mean():,.0f}/week")
print(f"[4] Artifacts saved -> {OUT}/")
print("\nPipeline complete! ✅")
