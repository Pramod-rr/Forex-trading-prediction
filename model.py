import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
import joblib

forex_data = pd.read_csv('data/EURUSD_.csv')
forex_data.columns = forex_data.columns.str.strip()
forex_data.rename(columns={'date': 'Date'}, inplace=True)
forex_data['Date'] = pd.to_datetime(forex_data['Date'])

close_data = forex_data.filter(['close'])
df = close_data.values
train_size = int(np.ceil(len(df) * .7))
train_size  #slice data

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df) 

train_data = scaled_data[0: int(train_size), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Scaling & preparing features and labels

model = Sequential()
model.add(Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1))))
model.add(BatchNormalization())
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  #build the neural network

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(x_train,
                    y_train,
                    epochs=10) # model compilationn and Training 
joblib.dump(model, 'model.pkl')
model.save('baseline.keras')
print('Model saved!!')