import io

import pandas as pd
from PIL import Image

df = pd.read_parquet("/home/adminalp/Desktop/openpi/datasets/ur7e_single_arm/data/chunk-000/episode_000000.parquet")
print(df.columns.tolist())
print(df.head())
print(df.iloc[0]["actions"])   # a single frame's action vector

# Image.open(io.BytesIO(df["wrist_image"][7]["bytes"])).show()
# Image.open(io.BytesIO(df["image"][7]["bytes"])).show()

for i in range(10)[]:
    print(i)

