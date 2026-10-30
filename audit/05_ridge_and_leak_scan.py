import pandas as pd, numpy as np
df = pd.read_csv("Project Data/Ready_data/FBREF.csv")

for c in ["is_FW","is_MF","is_DF"]: df[c]=df.pos.str.contains(c[3:],na=False).astype(int)
d = df.sort_values(["player","season"]).copy()
d["y"] = d.groupby("player")["Performance_Gls"].shift(-1)
d = d.replace([np.inf,-np.inf], np.nan)

drop = ["Performance_G+A","Performance_G-PK","Standard_Gls","Playing Time_Starts","nation","Unnamed: 0","age"]
d = d.drop(columns=[c for c in drop if c in d.columns])
excl = ["player","season","pos","alter_position","Performance_Gls","y","league","team"]
Xcols = [c for c in d.columns if c not in excl]
print("n features:", len(Xcols), "| any goal-derived name kept?", [c for c in Xcols if "Gls" in c][:5])

tr = d[d.season.isin([2020,2021,2022])].dropna(subset=["y"])
te = d[d.season==2023].dropna(subset=["y"])
Xtr = tr[Xcols].astype(float).fillna(0).values; ytr = tr.y.values
Xte = te[Xcols].astype(float).fillna(0).values; yte = te.y.values

# correlation of each feature with y (train)
cors = [abs(np.corrcoef(Xtr[:,i], ytr)[0,1]) if Xtr[:,i].std()>0 else 0 for i in range(Xtr.shape[1])]
top = sorted(zip(cors, Xcols), reverse=True)[:8]
print("\ntop |corr| with next-season goals (train):", [(round(c,3),n) for c,n in top])

# ridge (closed form) with train-only standardisation
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
Ztr = (Xtr-mu)/sd; Zte = (Xte-mu)/sd
Ztr = np.c_[np.ones(len(Ztr)), Ztr]; Zte = np.c_[np.ones(len(Zte)), Zte]
best=None
for lam in [0.1,1,10,30,100,300,1000]:
    A = Ztr.T@Ztr + lam*np.eye(Ztr.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Ztr.T@ytr)
    p = np.clip(Zte@w, 0, None)
    mae = np.mean(np.abs(yte-p)); r2 = 1 - np.sum((yte-p)**2)/np.sum((yte-yte.mean())**2)
    print(f"  ridge lam={lam:6g}  MAE={mae:.3f}  R2={r2:.3f}")
    if best is None or mae<best[0]: best=(mae,lam,r2)
print("\nbest ridge MAE %.3f (lam=%g, R2=%.3f) vs reported GBM ensemble 1.21 vs best naive 1.733" % best)
