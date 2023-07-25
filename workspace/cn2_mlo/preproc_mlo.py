import numpy as np
import pandas as pd

df = pd.read_csv("2_extracted/MLO_Obs_Stacked.csv.gz")
df["g"] = 9.81  # Add constant g
df["Cn2"] = np.power(10, df["LCn2"])  # Use non-log variant as base
df.to_csv("3_processed/MLO_Obs_Stacked.csv.gz", index=False)
