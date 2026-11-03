import io

import pandas as pd
from PIL import Image

df = pd.read_parquet("/media/adminalp/data/lerobot_db/ur7e_real_robot/data/chunk-000/episode_000000.parquet")
print(df.columns.tolist())
print(df.head())
print(df.iloc[50]["state"])     # a single frame's state vector
print(df.iloc[50]["actions"])   # a single frame's action vector

Image.open(io.BytesIO(df["wrist_image_left"][7]["bytes"])).show()
Image.open(io.BytesIO(df["wrist_image_right"][7]["bytes"])).show()
Image.open(io.BytesIO(df["top_image"][7]["bytes"])).show()