import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
# Reproduce Model.ipynb cell 6 (Goals 2024) EXACTLY up to the feature matrix, then evaluate honestly.
data_full = pd.read_csv("Project Data/Ready_data/FBREF.csv")
for c in ["FW","MF","DF"]:
    data_full[f"is_{c}"]=data_full.pos.str.contains(c,na=False).astype(int)
data_full = data_full.sort_values(["player","season"])
data_full["Performance_Gls_next"] = data_full.groupby("player")["Performance_Gls"].shift(-1)

data_full["Goal_Conversion_Rate"]=data_full["Performance_Gls"]/data_full["Standard_SoT"].replace(0,1)
data_full["xG_Overperformance"]=data_full["Performance_Gls"]-data_full["Expected_npxG"]
data_full["Shot_Accuracy"]=data_full["Standard_SoT"]/data_full["Expected_npxG"].replace(0,1)
data_full["Gls_trend"]=data_full.groupby("player")["Performance_Gls"].pct_change().fillna(0)
data_full["Recent_xG_Form"]=data_full.groupby("player")["Expected_npxG"].rolling(2,min_periods=1).mean().reset_index(level=0,drop=True)
data_full["Adjusted_Goal_Rate"]=(data_full["Performance_Gls"]/data_full["90s"].replace(0,1))*(1+0.3*(data_full["Performance_Gls"]>20).astype(int))
data_full["Goals_Delta"]=data_full.groupby("player")["Performance_Gls"].diff().fillna(0)
data_full["Goals_Delta"]=data_full.groupby("player")["Goals_Delta"].transform("mean").fillna(0)
data_full["Gls_per_90"]=data_full["Performance_Gls"]/data_full["90s"].replace(0,1)
data_full["Gls_per_90_trend"]=data_full.groupby("player")["Gls_per_90"].pct_change().fillna(0)
data_full["Minutes_per_90"]=data_full["Playing Time_Min"]/90
data_full["Minutes_per_90_trend"]=data_full.groupby("player")["Minutes_per_90"].pct_change().fillna(0)
data_full["Breakout_Signal"]=data_full["Gls_per_90_trend"]*data_full["Minutes_per_90_trend"]*data_full["Team Success_onG"]
data_full.loc[data_full["season"]==2023,"Gls_per_90_trend"]*=2.0
data_full.loc[data_full["season"]==2023,"Breakout_Signal"]*=10.0
data_full["Age"]=2024-data_full["born"]
data_full["Age_Decline"]=np.where(data_full["Age"]>32,(data_full["Age"]-32)*-0.1,0)
lag=data_full.groupby("league")["Team Success_onG"].mean()
data_full["Team_Strength"]=data_full["Team Success_onG"]/data_full["league"].map(lag)
data_full.replace([np.inf,-np.inf],np.nan,inplace=True)

real_target = data_full["Performance_Gls_next"].notna().copy()
data_full.fillna(data_full.median(numeric_only=True), inplace=True)   # <-- identity of fake labels
fake_rows = (~real_target) & (data_full["season"]==2023)

drop_f=["Performance_G+A","Performance_G-PK","Standard_Gls","Playing Time_Starts","nation","Unnamed: 0","age"]
data_full=data_full.drop(columns=[c for c in drop_f if c in data_full.columns])
excl=["player","season","pos","alter_position","Performance_Gls","Performance_Gls_next","league","team"]
Xf=[c for c in data_full.columns if c not in excl]
print("features:",len(Xf))

tr=data_full[data_full.season.isin([2019,2020,2021,2022])].dropna(subset=["Performance_Gls_next"]+Xf)
te=data_full[data_full.season==2023].dropna(subset=["Performance_Gls_next"]+Xf)
print("train rows:",len(tr)," test rows:",len(te))
print("test rows whose target was fabricated by the median fill:",int(((~real_target)&(data_full.season==2023)).sum()))

Xtr=tr[Xf].astype(float).values; ytr=tr.Performance_Gls_next.values
Xte=te[Xf].astype(float).values; yte=te.Performance_Gls_next.values
mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd==0]=1
Ztr=np.c_[(Xtr-mu)/sd,np.ones(len(Xtr))]; Zte=np.c_[(Xte-mu)/sd,np.ones(len(Xte))]
lam=10.0; A=Ztr.T@Ztr+lam*np.eye(Ztr.shape[1]); A[-1,-1]-=lam
w=np.linalg.solve(A,Ztr.T@ytr); p=np.clip(Zte@w,0,None)
mask_real = real_target.loc[te.index].values
def mae(y,q): return float(np.mean(np.abs(np.asarray(y)-np.asarray(q))))
print("\nRidge, same feature set & split as the notebook:")
print("  MAE on ALL 2023 rows (notebook's evaluation):     %.3f" % mae(yte,p))
print("  MAE on rows with a REAL next-season target only:  %.3f" % mae(yte[mask_real],p[mask_real]))
print("  rows with fabricated target, n=%d, their MAE:     %.3f" % ((~mask_real).sum(), mae(yte[~mask_real],p[~mask_real])))
print("\n  -> notebook reports 1.21 with GBMs; part of that gap is the fabricated-target rows.")
