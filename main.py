import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
data_path = "household_power_consumption.txt"  # File extracted from the zip

# Load data
df = pd.read_csv(data_path, sep=";", parse_dates={"datetime": ["Date", "Time"]}, infer_datetime_format=True,
                 na_values=["?"], low_memory=False)

# Data preprocessing
df.dropna(inplace=True)
df["Global_active_power"] = df["Global_active_power"].astype("float")
df.set_index("datetime", inplace=True)

# Downsample to daily data (optional)
daily_df = df["Global_active_power"].resample("D").sum()

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_df.values.reshape(-1, 1))

# Prepare training and test datasets
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 30  # Number of days for prediction
X, y = create_sequences(scaled_data, seq_length)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation="relu", return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.show()
