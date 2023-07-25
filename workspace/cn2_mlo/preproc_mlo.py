import numpy as np
import pandas as pd

df = pd.read_csv("2_extracted/MLO_Obs_Stacked.csv.gz")
df["g"] = 9.81
df["Cn2_tf"] = np.power(np.power(10, df["LCn2"]), 3/2)
df.to_csv("3_processed/MLO_Obs_Stacked.csv.gz", index=False)