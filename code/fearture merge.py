import pandas as pd

# Load Excel file and force 'Person' and 'Image' columns as string
df = pd.read_excel("./features_wavelet_train_ortho.xlsx", dtype=str)

# Convert only numeric columns (rest are strings)
numeric_df = df.drop(columns=['Person', 'Image']).apply(pd.to_numeric)

# Identify mean, variance, and energy columns
mean_cols = [col for col in numeric_df.columns if 'Mean' in col]
var_cols = [col for col in numeric_df.columns if 'Variance' in col]
energy_cols = [col for col in numeric_df.columns if 'Energy' in col]

# Compute overall features
df_reduced = pd.DataFrame()
df_reduced['Person'] = df['Person']
df_reduced['Image'] = df['Image']
df_reduced['Overall_Mean'] = numeric_df[mean_cols].mean(axis=1)
df_reduced['Overall_Variance'] = numeric_df[var_cols].mean(axis=1)
df_reduced['Overall_Energy'] = numeric_df[energy_cols].mean(axis=1)

# print(df_reduced)
# Save to Excel while keeping string formatting
df_reduced.to_excel("wavelet_features_ortho_reduced.xlsx", index=False)

print("✅ File saved with leading zeros preserved in 'Person' and 'Image' columns.")
