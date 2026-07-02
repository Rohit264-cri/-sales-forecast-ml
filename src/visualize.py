"""
visualize.py — Generate all charts for Sales Forecasting project
Run: python src/visualize.py
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings, os; warnings.filterwarnings('ignore')

DP='data/sales_data.csv'; FP='outputs/forecast_6months.csv'
CP='outputs/model_comparison.csv'; TP='outputs/test_predictions.csv'
ID='images'; os.makedirs(ID,exist_ok=True)

BG='#0d0f14'; SRF='#151820'; SR2='#1c2030'; BRD='#252a3a'
TX='#e2e8f0'; MU='#64748b'; AC='#6366f1'; CY='#22d3ee'
GR='#10b981'; RD='#f43f5e'; YL='#f59e0b'; PU='#a78bfa'

def dax(ax):
    ax.set_facecolor(SRF); ax.tick_params(colors=MU,labelsize=9)
    ax.spines[:].set_color(BRD)
    for l in ax.get_xticklabels()+ax.get_yticklabels(): l.set_color(MU)

df=pd.read_csv(DP,parse_dates=['date'])
fdf=pd.read_csv(FP,parse_dates=['date'])
cdf=pd.read_csv(CP)
tdf=pd.read_csv(TP,parse_dates=['date'])
monthly=df.groupby(df['date'].dt.to_period('M'))['sales'].sum().reset_index()
monthly['date']=monthly['date'].dt.to_timestamp()

# ════ DASHBOARD ════════════════════════════════════════════════
fig,axes=plt.subplots(2,3,figsize=(18,10))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=.06,right=.97,top=.80,bottom=.09,wspace=.35,hspace=.48)

kax=fig.add_axes([0,.86,1,.11]); kax.set_facecolor(BG); kax.axis('off')
total=df['sales'].sum(); avg=df['sales'].mean(); peak=df['sales'].max()
yoy=((df[df['year']==2024]['sales'].sum()-df[df['year']==2023]['sales'].sum())
     /df[df['year']==2023]['sales'].sum()*100)
for i,(v,l,c) in enumerate([(f"Rs {total/1e6:.1f}M","Total Revenue 3Y",CY),
                              (f"Rs {avg:,.0f}","Avg Weekly Sales",GR),
                              (f"Rs {peak:,.0f}","Peak Week Sales",YL),
                              (f"+{yoy:.1f}%","YoY Growth 23→24",AC)]):
    x=.10+i*.22
    kax.add_patch(FancyBboxPatch((x-.09,.05),.18,.88,boxstyle="round,pad=0.02",
                  linewidth=1,edgecolor=c,facecolor=SRF,transform=kax.transAxes))
    kax.text(x,.64,v,ha='center',va='center',fontsize=15,fontweight='bold',color=c,transform=kax.transAxes)
    kax.text(x,.22,l,ha='center',va='center',fontsize=9,color=MU,transform=kax.transAxes)

fig.text(.04,.97,'📈 Sales Revenue Forecasting — ML Project Dashboard',
         fontsize=14,fontweight='bold',color=TX,va='top')
fig.text(.04,.93,'Dataset: 3 Years Weekly Retail Data | Python · Scikit-learn · XGBoost | Tools: Power BI',
         fontsize=9,color=MU,va='top')

# [0,0] Monthly Revenue Trend
ax=axes[0,0]; dax(ax)
ax.plot(monthly['date'],monthly['sales']/1000,color=CY,lw=2.2,
        marker='o',ms=4,markerfacecolor=CY,markeredgecolor=BG,markeredgewidth=1.5)
ax.fill_between(monthly['date'],monthly['sales']/1000,alpha=.12,color=CY)
ax.set_title('Monthly Revenue Trend (2022–2024)',color=TX,fontsize=11,pad=8)
ax.set_ylabel('Sales (Rs K)',color=MU,fontsize=9)
ax.grid(color=SR2,lw=.8); ax.tick_params(axis='x',rotation=40,labelsize=7)

# [0,1] Actual vs Predicted (test set)
ax=axes[0,1]; dax(ax)
ax.plot(range(len(tdf)),tdf['actual']/1000,color=GR,lw=2,
        marker='o',ms=4,markerfacecolor=GR,markeredgecolor=BG,markeredgewidth=1.5,label='Actual')
ax.plot(range(len(tdf)),tdf['predicted']/1000,color=YL,lw=2,ls='--',
        marker='s',ms=4,markerfacecolor=YL,markeredgecolor=BG,markeredgewidth=1.5,label='Predicted')
r2=cdf['R2'].max()
ax.set_title(f'Actual vs Predicted (Test Set, R²={r2:.3f})',color=TX,fontsize=11,pad=8)
ax.legend(fontsize=9,labelcolor=MU,facecolor=SR2,edgecolor=BRD)
ax.grid(color=SR2,lw=.8); ax.set_xlabel('Test Weeks',color=MU,fontsize=9)
ax.set_ylabel('Sales (Rs K)',color=MU,fontsize=9)

# [0,2] Model Comparison
ax=axes[0,2]; dax(ax)
names=cdf['Model'].tolist(); r2s=cdf['R2'].tolist(); maes=cdf['MAE'].tolist()
mcols=[AC,GR,YL]
brs=ax.bar(names,r2s,color=mcols,edgecolor=BG,lw=1.5,width=.55)
for bar,r,m in zip(brs,r2s,maes):
    ax.text(bar.get_x()+bar.get_width()/2,r+.01,
            f'R²={r:.3f}\nMAE=Rs{m/1000:.1f}K',
            ha='center',color=TX,fontsize=9,fontweight='bold',va='bottom')
ax.set_ylim(0,1.05); ax.set_title('3 Models Comparison',color=TX,fontsize=11,pad=8)
ax.set_ylabel('R² Score',color=MU,fontsize=9)
ax.grid(axis='y',color=SR2,lw=.8); ax.tick_params(axis='x',labelsize=8)

# [1,0] Seasonal Pattern (Monthly Avg)
ax=axes[1,0]; dax(ax)
mavg=df.groupby('month')['sales'].mean()/1000
mnames=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mc=[RD if v>mavg.mean()*1.1 else GR if v<mavg.mean()*.9 else CY for v in mavg]
brs2=ax.bar(mnames,mavg.values,color=mc,edgecolor=BG,lw=1.2,width=.7)
for b,v in zip(brs2,mavg): ax.text(b.get_x()+b.get_width()/2,v+.3,f'{v:.0f}K',ha='center',color=TX,fontsize=8,fontweight='bold')
ax.set_title('Seasonal Sales Pattern by Month',color=TX,fontsize=11,pad=8)
ax.grid(axis='y',color=SR2,lw=.8); ax.tick_params(axis='x',labelsize=8)

# [1,1] Marketing vs Sales
ax=axes[1,1]; dax(ax)
ax.scatter(df['marketing_spend']/1000,df['sales']/1000,
           c=df['is_holiday'].map({0:CY,1:YL}),alpha=.6,s=30,edgecolors='none')
m,b=np.polyfit(df['marketing_spend']/1000,df['sales']/1000,1)
xr=np.linspace(df['marketing_spend'].min()/1000,df['marketing_spend'].max()/1000,100)
corr=np.corrcoef(df['marketing_spend'],df['sales'])[0,1]
ax.plot(xr,m*xr+b,color=RD,lw=2,ls='--',label=f'Trend r={corr:.2f}')
ax.set_title('Marketing Spend vs Revenue',color=TX,fontsize=11,pad=8)
ax.set_xlabel('Marketing (Rs K)',color=MU,fontsize=9); ax.set_ylabel('Sales (Rs K)',color=MU,fontsize=9)
ax.legend(fontsize=9,labelcolor=MU,facecolor=SR2,edgecolor=BRD)
ax.grid(color=SR2,lw=.8)

# [1,2] 6-Month Forecast
ax=axes[1,2]; dax(ax)
last16=df.tail(16)
ax.plot(range(16),last16['sales']/1000,color=GR,lw=2.2,
        marker='o',ms=4,markerfacecolor=GR,markeredgecolor=BG,markeredgewidth=1.5,label='Actual (Last 16W)')
fx=range(15,15+len(fdf)+1)
fy=[last16['sales'].iloc[-1]/1000]+list(fdf['predicted_sales']/1000)
ax.plot(fx,fy,color=YL,lw=2.2,ls='--',marker='s',ms=4,
        markerfacecolor=YL,markeredgecolor=BG,markeredgewidth=1.5,label='Forecast (24W)')
ax.fill_between(fx,np.array(fy)*.92,np.array(fy)*1.08,alpha=.15,color=YL)
ax.axvline(x=15.5,color=RD,lw=1.5,ls=':',alpha=.7)
ax.text(16,ax.get_ylim()[1]*.97,'Forecast →',color=RD,fontsize=8)
ax.set_title('24-Week Sales Forecast',color=TX,fontsize=11,pad=8)
ax.legend(fontsize=8,labelcolor=MU,facecolor=SR2,edgecolor=BRD)
ax.grid(color=SR2,lw=.8); ax.set_xlabel('Weeks',color=MU,fontsize=9); ax.set_ylabel('Sales (Rs K)',color=MU,fontsize=9)

plt.savefig(f'{ID}/dashboard.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print(f"✅ {ID}/dashboard.png")

# ════ LINKEDIN HERO (3-panel clean) ════════════════════════════
fig,axes=plt.subplots(1,3,figsize=(18,6))
fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=.06,right=.97,top=.76,bottom=.13,wspace=.32)
fig.text(.5,.92,'🤖  Sales Forecasting using ML — End-to-End Project',ha='center',
         fontsize=16,fontweight='bold',color=TX)
fig.text(.5,.86,'Python  ·  Pandas  ·  Scikit-learn  ·  XGBoost  |  Dataset: 3 Years Retail Data',
         ha='center',fontsize=10,color=MU)

ax=axes[0]; dax(ax)
ax.plot(monthly['date'],monthly['sales']/1000,color=CY,lw=2.5,
        marker='o',ms=5,markerfacecolor=CY,markeredgecolor=BG,markeredgewidth=2)
ax.fill_between(monthly['date'],monthly['sales']/1000,alpha=.12,color=CY)
ax.set_title('3-Year Monthly Revenue Trend',color=TX,fontsize=12,pad=10,fontweight='bold')
ax.set_ylabel('Sales (Rs K)',color=MU,fontsize=10)
ax.grid(color=SR2,lw=.8); ax.tick_params(axis='x',rotation=38,labelsize=7)

ax=axes[1]; dax(ax)
brs3=ax.bar(names,r2s,color=mcols,edgecolor=BG,lw=1.5,width=.55)
for bar,r,m in zip(brs3,r2s,maes):
    ax.text(bar.get_x()+bar.get_width()/2,r+.01,
            f'R²={r:.3f}\nMAE=Rs{m/1000:.0f}K',
            ha='center',color=TX,fontsize=10,fontweight='bold',va='bottom')
ax.set_ylim(0,1.1); ax.set_title('ML Model Comparison (R² Score)',color=TX,fontsize=12,pad=10,fontweight='bold')
ax.set_ylabel('R² Score',color=MU,fontsize=10); ax.grid(axis='y',color=SR2,lw=.8)
ax.tick_params(axis='x',labelsize=9)

ax=axes[2]; dax(ax)
last12=df.tail(12)
ax.plot(range(12),last12['sales']/1000,color=GR,lw=2.5,
        marker='o',ms=5,markerfacecolor=GR,markeredgecolor=BG,markeredgewidth=2,label='Actual')
fx2=range(11,11+len(fdf)+1)
fy2=[last12['sales'].iloc[-1]/1000]+list(fdf['predicted_sales']/1000)
ax.plot(fx2,fy2,color=YL,lw=2.5,ls='--',marker='s',ms=4,
        markerfacecolor=YL,markeredgecolor=BG,markeredgewidth=2,label='Predicted')
ax.fill_between(fx2,np.array(fy2)*.92,np.array(fy2)*1.08,alpha=.18,color=YL)
ax.axvline(x=11.5,color=RD,lw=2,ls=':',alpha=.8)
ax.set_title('24-Week Revenue Forecast',color=TX,fontsize=12,pad=10,fontweight='bold')
ax.legend(fontsize=10,labelcolor=MU,facecolor=SR2,edgecolor=BRD)
ax.grid(color=SR2,lw=.8); ax.set_xlabel('Weeks',color=MU,fontsize=10); ax.set_ylabel('Sales (Rs K)',color=MU,fontsize=10)

plt.savefig(f'{ID}/linkedin_hero.png',dpi=150,bbox_inches='tight',facecolor=BG)
plt.close(); print(f"✅ {ID}/linkedin_hero.png")
print("All images generated! ✅")
