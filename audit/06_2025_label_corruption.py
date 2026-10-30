import pandas as pd, numpy as np
# Reproduce Model.ipynb cell 1 EXACTLY, step by step, and inspect the training set it builds.
data_full = pd.read_csv("Project Data/Ready_data/FBREF.csv")
data_2025 = pd.read_csv("Project Data/Ready_data/test2025.csv")

for df in [data_full, data_2025]:
    df["is_FW"]=df.pos.str.contains("FW",na=False).astype(int)
    df["is_MF"]=df.pos.str.contains("MF",na=False).astype(int)
    df["is_DF"]=df.pos.str.contains("DF",na=False).astype(int)

data_full["Performance_Gls_next"] = data_full.groupby("player")["Performance_Gls"].shift(-1)
print("target non-null BEFORE the feature loop:", data_full.Performance_Gls_next.notna().sum(), "/", len(data_full))
print("  null targets by season:", data_full.groupby('season').Performance_Gls_next.apply(lambda s: s.isna().sum()).to_dict())

for df in [data_full, data_2025]:
    df["Goal_Conversion_Rate"] = df["Performance_Gls"]/df["Standard_SoT"].replace(0,1)
    df["xG_Overperformance"] = df["Performance_Gls"]-df["Expected_npxG"]
    df["Shot_Accuracy"] = df["Standard_SoT"]/df["Expected_npxG"].replace(0,1)
    df["Gls_trend"] = df.groupby("player")["Performance_Gls"].pct_change().fillna(0)
    df["Gls_per_90"] = df["Performance_Gls"]/df["90s"].replace(0,1)
    df["Gls_per_90_trend"] = df.groupby("player")["Gls_per_90"].pct_change().fillna(0)
    df["Recent_xG_Form"] = df.groupby("player")["Expected_npxG"].rolling(2,min_periods=1).mean().reset_index(level=0,drop=True)
    df["Adjusted_Goal_Rate"] = (df["Performance_Gls"]/df["90s"].replace(0,1))*(1+0.3*(df["Performance_Gls"]>20).astype(int))
    df["Minutes_per_90"] = df["Playing Time_Min"]/90
    df["Minutes_per_90_trend"] = df.groupby("player")["Minutes_per_90"].pct_change().fillna(0)
    df["Breakout_Signal"] = df["Gls_per_90_trend"]*df["Minutes_per_90_trend"]*df["Team Success_onG"]
    df.loc[df["season"]==2024,"Gls_per_90_trend"] *= 2.0
    df.loc[df["season"]==2024,"Breakout_Signal"] *= 10.0
    df["Age"] = 2025-df["born"]
    df["Age_Decline"] = np.where(df["Age"]>32,(df["Age"]-32)*-0.1,0)
    league_avg_goals = df.groupby("league")["Team Success_onG"].mean()
    df["Team_Strength"] = df["Team Success_onG"]/df["league"].map(league_avg_goals)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.fillna(0,inplace=True)          # <-- runs AFTER the target was created

print("\ntarget non-null AFTER fillna(0):", data_full.Performance_Gls_next.notna().sum(), "/", len(data_full))
print("  nulls remaining:", data_full.Performance_Gls_next.isna().sum())

drop_features=["Performance_G+A","Performance_G-PK","Standard_Gls","Playing Time_Starts","nation","Unnamed: 0","age"]
data_full.drop(columns=[c for c in drop_features if c in data_full.columns],inplace=True)
exclude=["player","season","pos","alter_position","Performance_Gls","Performance_Gls_next","league","team"]
Xf=[c for c in data_full.columns if c not in exclude]

train = pd.concat([data_full[data_full['season'].isin([2020,2021])],
                   data_full[data_full['season']==2022]*2,
                   data_full[data_full['season']==2023]*2,
                   data_full[data_full['season']==2024]*4]).dropna(subset=['Performance_Gls_next'])
print("\nTRAIN ROWS actually built:", len(train))
print("  by season (after weighting):\n", train.groupby('season').size().to_dict())
print("  target mean by season:\n", train.groupby('season')['Performance_Gls_next'].mean().to_dict())
print("  share of training rows whose label is a synthetic 0:", float((train.Performance_Gls_next==0).mean()))
print("\n-> 2024 rows carry the 0 label that fillna(0) wrote over the genuinely-missing target.")
