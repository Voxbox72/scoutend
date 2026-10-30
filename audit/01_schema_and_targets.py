import pandas as pd, numpy as np
pd.set_option("display.width", 200)

df = pd.read_csv("Project Data/Ready_data/FBREF.csv")
print("FBREF shape", df.shape)
print("\nseason counts:\n", df.groupby("season").size())
print("\nleagues:", df.league.unique())
print("\npos values:", df.pos.value_counts(dropna=False).to_dict())
print("\nalter_position:", df.alter_position.value_counts(dropna=False).to_dict() if "alter_position" in df else "missing")
print("\nduplicate player-season rows:", df.duplicated(subset=["player","season"]).sum())
print("exact duplicate rows:", df.duplicated().sum())

g = df.groupby("player").agg(n=("season","nunique"), seasons=("season", lambda s: tuple(sorted(s))))
print("\nplayers with >1 season:", (g.n>1).sum(), "of", len(g))
print("players with all 5 seasons:", (g.n==5).sum())

# gaps: non-consecutive seasons
def maxgap(s):
    s=sorted(s); return max([b-a for a,b in zip(s,s[1:])] or [0])
gaps = g.seasons.map(maxgap)
print("players with a season gap (target shift-1 crosses it):", (gaps>1).sum())

# target reconstruction check
df2 = df.copy()
df2["next_"] = df2.groupby("player")["Performance_Gls"].shift(-1)
ok = df2.dropna(subset=["next_"])
print("\nrows with a next-season target:", len(ok))
# how many of those are actually the immediately following season?
srt = df2.sort_values(["player","season"])
srt["nextseason"] = srt.groupby("player")["season"].shift(-1)
srt["target"] = srt.groupby("player")["Performance_Gls"].shift(-1)
m = srt.dropna(subset=["target"])
print("  of which consecutive season:", ((m.nextseason - m.season)==1).sum())
print("  rows where unsorted shift differs from sorted shift (the 2025 model's bug):", end=" ")
u = df2.copy(); u["nxt_unsorted"]=u.groupby("player")["Performance_Gls"].shift(-1)
print((u["nxt_unsorted"].fillna(-1) != srt.set_index(u.index)["target"].fillna(-1)).sum())
