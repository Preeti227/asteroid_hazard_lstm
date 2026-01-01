import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("nasa.csv")

# Drop non-informative columns
df = df.drop(columns=[
    'Neo Reference ID',
    'Name',
    'Orbiting Body',
    'Orbit ID'
])

# Inspect
print(df.head())
print(df.shape)
print(df.info())
