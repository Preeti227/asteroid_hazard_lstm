import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Drop Equinox (single value)
df = df.drop(columns=['Equinox'])

# Correlation analysis
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr)
plt.show()

# Drop redundant features
df = df.drop(columns=[
    'Est Dia in KM(min)', 'Est Dia in KM(max)',
    'Est Dia in M(min)', 'Est Dia in M(max)',
    'Est Dia in Miles(min)', 'Est Dia in Miles(max)',
    'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
    'Relative Velocity km per sec', 'Relative Velocity km per hr',
    'Miss Dist.(lunar)', 'Miss Dist.(kilometers)',
    'Miss Dist.(Astronomical)', 'Orbit Uncertainity',
    'Semi Major Axis', 'Orbital Period'
])
