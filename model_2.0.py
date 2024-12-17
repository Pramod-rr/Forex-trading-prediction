import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt

forex_data = pd.read_csv('data/EUR_USD Historical Data.csv')
forex_data.columns = forex_data.columns.str.strip()
forex_data.rename(columns={'date': 'Date'}, inplace=True)
forex_data['Date'] = pd.to_datetime(forex_data['Date'])
forex_data = forex_data.drop(columns=['Vol.','Change %'])

close_data = forex_data.filter(['High','Low'])
df = close_data.values
train_size = int(np.ceil(len(df) * .7))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[:, :close_data.shape[1]]) 

train_data = scaled_data[0: int(train_size), :]

x_train = []
y_train = []

for i in range(1, len(train_data)):
    x_train.append(train_data[i-1:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1))))
model.add(BatchNormalization())
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(x_train,
                    y_train,
                    epochs=10)

model.save('model_2.keras')

# Load trained model
model = load_model("model_2.keras")

# Load new data for predictions
new_data = pd.read_csv("data/EUR_USD Historical Data.csv")
new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data.set_index('Date', inplace=True)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_data[['Price']].values)

# Prepare data for prediction
X_pred = []
for i in range(1, len(scaled_data)):
    X_pred.append(scaled_data[i-1:i, 0])
X_pred = np.array(X_pred)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[1], 1)


## Predict
# Load trained model
model = load_model("model_1.keras")

# Load new data for predictions
new_data = pd.read_csv("data/EUR_USD Historical Data.csv")
new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data.set_index('Date', inplace=True)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_data[['High', 'Low']].values)

# Prepare data for prediction# Sliding window size
lookback = 1
X_pred = [scaled_data[i-lookback:i, 0] for i in range(lookback, len(scaled_data))]
X_pred = np.array(X_pred).reshape(len(X_pred), lookback, 1)

# Predict
# Predict
try:
    predictions = model.predict(X_pred)
except ValueError as e:
    print("Error during prediction:", e)

# Ensure model output matches scaler input for inverse transformation
if predictions.shape[1] != scaled_data.shape[1]:
    print(f"Model predictions ({predictions.shape}) do not match scaler input shape ({scaled_data.shape}).")
    # Handle single-column predictions by creating a placeholder
    predictions = np.hstack([predictions, np.zeros((predictions.shape[0], 2))])

# Inverse transform predictions
predicted_prices = scaler.inverse_transform(predictions[:, :scaled_data.shape[1]])


# Save predictions
prediction_dates = new_data.index[lookback:]
# Ensure matching lengths
min_length = min(len(prediction_dates), len(predicted_prices))
prediction_dates = prediction_dates[:min_length]
predicted_prices = predicted_prices[:min_length]

# Create DataFrame
results = pd.DataFrame({
    "Date": pd.to_datetime(prediction_dates).strftime('%m-%d-%Y'),
    "Predicted_High": predicted_prices[:, 0].round(4),
    "Predicted_Low": predicted_prices[:, 1].round(4)
})

results.to_csv('predicted_price_3.csv', index=False)
print("Predictions saved to 'predicted_price_3.csv'")