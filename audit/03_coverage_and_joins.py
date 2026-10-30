import pandas as pd, numpy as np, unicodedata, re
df = pd.read_csv("Project Data/Ready_data/FBREF.csv")

# 1. multi-position overlap
df["is_FW"]=df.pos.str.contains("FW",na=False); df["is_MF"]=df.pos.str.contains("MF",na=False); df["is_DF"]=df.pos.str.contains("DF",na=False)
print("position-dummy sum distribution:", df[["is_FW","is_MF","is_DF"]].sum(axis=1).value_counts().to_dict())
print("rows with 0 position dummies:", (df[["is_FW","is_MF","is_DF"]].sum(axis=1)==0).sum())
print("pos multi-label (e.g. 'DF,FW') rows:", df.pos.str.contains(",").sum())

# 2. label contamination detail
s = df.sort_values(["player","season"], kind="stable").copy()
s["t_next"]=s.groupby("player")["Performance_Gls"].shift(-1)
s["s_next"]=s.groupby("player")["season"].shift(-1)
t = s.dropna(subset=["t_next"])
same_season = (t.s_next==t.season)
gap = (t.s_next-t.season)>1
print("\ntraining labels (non-null):", len(t))
print("  contaminated (target = same season, other club):", same_season.sum(), f"({same_season.mean():.1%})")
print("  skipped a season (gap>1):", gap.sum(), f"({gap.mean():.1%})")

# 3. minutes filter absence
print("\nrows with <450 min (90s<5):", (df["90s"]<5).sum(), f"({(df['90s']<5).mean():.1%})")
print("  of those, goals==0:", ((df["90s"]<5)&(df.Performance_Gls==0)).sum())

# 4. check goals exactly recoverable from kept features
rec = (df["Per 90 Minutes_Gls"]*df["90s"]).round(2)
print("\n'Per 90 Minutes_Gls * 90s' reproduces Performance_Gls exactly:", np.allclose(rec, df.Performance_Gls, atol=0.02))
print("'Per 90 Minutes_Gls' is in the feature set (yes -> current goals are a feature)")

# 5. test2025 vs FBREF name join quality
t25 = pd.read_csv("Project Data/Ready_data/test2025.csv")
def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z ]","",x).strip()
b = set(df[df.season==2024].player.map(norm))
t25["k"]=t25.player.map(norm)
print("\n2025 file players:", len(t25), " normalised join to 2023/24 names:", t25.k.isin(b).sum())
print("  exact-name join would fail on:", len(t25)-t25.player.isin(set(df.player)))

# 6. nulls remaining in shipped tables
for f in ["FBREF.csv","defenders.csv","midfielders.csv","attackers.csv","test2025.csv"]:
    d = pd.read_csv("Project Data/Ready_data/"+f)
    print(f"{f:20s} shape={d.shape} nulls={int(d.isna().sum().sum())} zeros={int((d.select_dtypes('number')==0).sum().sum())}")

# 7. league/season coverage matrix
print("\nleagues x seasons:\n", pd.crosstab(df.league, df.season))
