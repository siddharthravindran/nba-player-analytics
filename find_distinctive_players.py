"""
Find players whose SHAP drivers are MOST DIFFERENT from the league norm.
Great for showing that the feature SET is the same for everyone, but WHICH features
matter varies by player — answering "are the attributes the same for all players?"

Run from the repo root:  python3 find_distinctive_players.py
"""
import pandas as pd, numpy as np

shap = pd.read_parquet("data/shap_v1.parquet")
season = "2025-26"
s = shap[shap["SEASON"] == season].copy()

# normalize each player's |shap| into a distribution over features (their "driver fingerprint")
piv = (s.assign(absshap=s["shap_usd"].abs())
         .pivot_table(index="PLAYER_NAME", columns="feature", values="absshap", aggfunc="mean")
         .fillna(0))
# row-normalize so each player is a probability distribution over features
norm = piv.div(piv.sum(axis=1), axis=0)
league_avg = norm.mean(axis=0)  # the "norm" fingerprint

# distance of each player's fingerprint from the league-average fingerprint
dist = ((norm - league_avg) ** 2).sum(axis=1) ** 0.5
out = dist.sort_values(ascending=False)

print(f"=== Players whose drivers are MOST DIFFERENT from the norm ({season}) ===\n")
for name in out.head(12).index:
    # what makes them distinctive: features where they're most ABOVE the league average
    diff = (norm.loc[name] - league_avg).sort_values(ascending=False)
    top = [f for f in diff.head(3).index]
    print(f"{name:28s}  distinctive drivers: {', '.join(top)}")

print(f"\n=== Most TYPICAL players (drivers closest to the norm) ===\n")
for name in out.tail(5).index:
    print(f"{name:28s}  (prices like a 'standard' profile)")

print("\nTakeaway for your dad: the FEATURE SET is identical for everyone,")
print("but WHICH features drive each player's price varies a lot — a rim-running")
print("big is priced on different attributes than a shot-creating guard.")
