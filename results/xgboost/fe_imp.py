import pandas as pd

# Load feature importance CSV
df = pd.read_csv(
    "tuned_feature_importance_gain.csv",
    index_col=0
)

# Sort by gain (highest first)
df = df.sort_values(by="gain", ascending=False)

print("\nTop 15 features by Gain")
print(df.head(15).to_string())