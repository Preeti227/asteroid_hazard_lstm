import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import lime
import lime.lime_tabular
import random

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

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

def scheduler(epoch, lr):
    return lr if epoch < 10 else lr * tf.math.exp(-0.1).numpy()

history = lstm_model.fit(
    X_train_lstm, y_train,
    epochs=100,
    validation_data=(X_test_lstm, y_test),
    callbacks=[
        EarlyStopping(patience=10),
        LearningRateScheduler(scheduler)
    ]
)

loss, accuracy = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

y_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_probs = lstm_model.predict(X_test_lstm).ravel()
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], '--')
plt.legend()
plt.show()

class LSTMWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        reshaped = data.reshape((data.shape[0], 1, data.shape[1]))
        prob_pos = self.model.predict(reshaped)
        return np.hstack((1 - prob_pos, prob_pos))

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns.tolist(),
    class_names=['Not Hazardous', 'Hazardous'],
    mode='classification'
)

idx = random.randint(0, X_test_scaled.shape[0] - 1)
explanation = explainer.explain_instance(
    X_test_scaled[idx],
    LSTMWrapper(lstm_model).predict,
    num_features=10
)

print(explanation.as_list())

lstm_model.save("lstm_asteroid_model.h5")
print("Model saved")
