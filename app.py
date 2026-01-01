import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

y = df['Hazardous'].astype(int)
X = df.drop(columns=['Hazardous'])
X = X.select_dtypes(include='number')

# Boxplot visualization
X_boxplot = X.copy()
X_boxplot['Hazardous'] = y.values

X_melted = pd.melt(X_boxplot.drop(columns='Hazardous'))

plt.figure(figsize=(16, 6))
sns.boxplot(x='variable', y='value', data=X_melted)
plt.xticks(rotation=90)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_lstm = X_train_scaled.reshape(
    X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
)
X_test_lstm = X_test_scaled.reshape(
    X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
)
