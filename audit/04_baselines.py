import pandas as pd, numpy as np
df = pd.read_csv("Project Data/Ready_data/FBREF.csv")

# replicate Model.ipynb (2024 cell) label construction EXACTLY: sort, then shift
d = df.sort_values(["player","season"]).copy()
d["y"] = d.groupby("player")["Performance_Gls"].shift(-1)
d = d.replace([np.inf,-np.inf], np.nan)

train = d[d.season.isin([2019,2020,2021,2022])].dropna(subset=["y"])
test  = d[d.season==2023].dropna(subset=["y"])
print("their train rows:", len(train), " test rows:", len(test))
print("test target mean %.3f  var %.3f" % (test.y.mean(), test.y.var()))

def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))

br = {}
br["season-mean"] = mae(test.y, np.full(len(test), train.y.mean()))
br["position-season mean"] = mae(test.y, test.groupby("pos")["y"].transform("mean"))
br["last season goals"] = mae(test.y, test["Performance_Gls"])
br["last season npxG"] = mae(test.y, test["Expected_npxG"])
# naive 2-season average of goals
d2 = d.sort_values(["player","season"]).copy()
g_prev = d2.groupby("player")["Performance_Gls"].shift(1)
d2["avg2"] = np.where(g_prev.notna(), (d2.Performance_Gls + g_prev.fillna(0))/np.where(g_prev.notna(),2,1), d2.Performance_Gls)
t2 = d2[d2.season==2023].dropna(subset=["y"])
br["2-season avg goals"] = mae(t2.y, t2.avg2)

print("\nHonest baselines on THEIR exact 2023->2024 test population:")
for k,v in br.items(): print(f"  {k:24s} MAE={v:.3f}")
print("  reported ensemble MAE (notebook) = 1.21")

# restricted to established players
m = test["Playing Time_Min"]>=900
print("\nrestricted to >=900 min (n=%d):" % m.sum())
for k in ["last season goals","last season npxG","season-mean"]:
    if k=="season-mean": p=np.full(m.sum(), train.y.mean())
    elif k=="last season goals": p=test.loc[m,"Performance_Gls"]
    else: p=test.loc[m,"Expected_npxG"]
    print(f"  {k:24s} MAE={mae(test.loc[m,'y'], p):.3f}   (target mean {test.loc[m,'y'].mean():.2f})")
