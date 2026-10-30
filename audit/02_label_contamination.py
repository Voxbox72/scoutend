import pandas as pd, numpy as np
df = pd.read_csv("Project Data/Ready_data/FBREF.csv")

# proper sort then shift
u = df.copy()
u["nxt_flat"] = u.groupby("player")["Performance_Gls"].shift(-1)
s = df.sort_values(["player","season"], kind="stable").copy()
s["nxt_sorted"] = s.groupby("player")["Performance_Gls"].shift(-1)
s["next_season"] = s.groupby("player")["season"].shift(-1)
a = u["nxt_flat"]; b = s["nxt_sorted"].reindex(u.index)
d = (a.fillna(-999) != b.fillna(-999))
print("rows where flat order != season-sorted order:", d.sum(), "of", len(df))
print("  ... their season/player dup detail:")
dd = s[d.reindex(s.index).values]
print("  of these, rows inside a duplicated player-season:", s.loc[d.reindex(s.index).values].duplicated(subset=["player","season"], keep=False).sum())

# duplicate player-season examples
dups = df[df.duplicated(subset=["player","season"], keep=False)].sort_values(["player","season"])
print("\nduplicate player-season sample:")
print(dups[["player","season","team","Performance_Gls","Playing Time_Min","pos"]].head(12).to_string())

# same-season target contamination in the 2025 model
u2 = df.copy()
u2["t"] = u2.groupby("player")["Performance_Gls"].shift(-1)
same = u2[u2.duplicated(subset=["player","season"], keep=False)]["t"].notna().sum()
print("\nrows whose 'next' target is actually the same season's other club row:", same)

print("\nrows per player-season after dedupe (keep last):", len(df.drop_duplicates(subset=["player","season"], keep="last")))

# naive baselines for season 2023 -> 2024
s2 = s.drop_duplicates(subset=["player","season"], keep="last")
tr = s2[s2.season==2023].dropna(subset=["nxt_sorted"]).copy()
print("\n2023->2024 test rows:", len(tr), " target mean %.3f median %.3f" % (tr.nxt_sorted.mean(), tr.nxt_sorted.median()))
for name, pred in [("predict season mean", np.full(len(tr), tr.nxt_sorted.mean())),
                   ("predict player's 2023 goals", tr.Performance_Gls.values),
                   ("predict 2023 npxG", tr.Expected_npxG.values),
                   ("predict 2023 goals/90 * 30", (tr["Per 90 Minutes_Gls"]*30).values)]:
    pred = np.clip(pred, 0, None)
    print(f"  {name:35s} MAE={np.mean(np.abs(tr.nxt_sorted.values-pred)):.3f}")
# R2 of constant predictor
y=tr.nxt_sorted.values
print("  var(y)=%.3f  so R2 of mean-predictor=0 by def" % y.var())

t25 = pd.read_csv("Project Data/Ready_data/test2025.csv")
print("\ntest2025 shape", t25.shape, "seasons", t25.season.unique(), "MP mean", t25["Playing Time_MP"].mean())
m = s2[s2.season==2024][["player","Performance_Gls","Playing Time_MP"]].merge(t25[["player","Performance_Gls","Playing Time_MP"]], on="player", suffixes=("_fb24","_t25"))
print("matched players 2025 file:", m.shape[0], "of", t25.shape[0])
print("mean goals in test2025:", t25.Performance_Gls.mean(), " vs 2024:", s2[s2.season==2024].Performance_Gls.mean())
