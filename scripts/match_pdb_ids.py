import os
import pandas as pd

ki_df = pd.read_csv("data/ki_only_data.csv")  # your original Ki CSV
refined_dir = "data/refined-set/"
refined_ids = set(os.listdir(refined_dir))

matched_df = ki_df[ki_df['PDB_ID'].isin(refined_ids)]
matched_df.to_csv("data/filtered_ki_refined.csv", index=False)
print("Matched:", len(matched_df))
