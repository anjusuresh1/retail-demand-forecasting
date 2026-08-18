from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

LOOKBACK = 28
INPUT_FILE =  Path('data/processed/model_features.parquet')

def create_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X=[]
    y=[]
    for index in range(lookback, len(values)):
        X.append(values[index - lookback: index])
        y.append(values[index])
        
    return np.array(X), np.array(y)

def build_lstm_model(lookback: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimisation = 'adam',
        loss = 'mse',
        metrics = ['mae']
    )
    
    return model

def main() -> None:
    data = pd.read_parquet(INPUT_FILE)
    
    data['Date'] = pd.to_datetime(data['Date'])
    
    selected_product = (data['StockCode'].values_count().index[0])
    
    product_data = data[data['StockCode'] == selected_product].sort_values('Date')
    
    values = product_data['Demand'].values.astype('float32')
    
    split_index = len(values) * 0.8
    
    train_data = values[:split_index]
    
    test_data = values[split_index:]
    
    scaler = MinMaxScaler()
    
    train_scaled = scaler.fit_transform(train_data)
    
    test_context = np.concatenate([
        train_data[-LOOKBACK:],
        test_data
    ])
    
    test_scaled = scaler.transform(test_context)
    
    X_train, X_test = create_sequences(train_scaled, LOOKBACK)
    
    y_train, y_test = create_sequences(test_context, LOOKBACK)
    
    model = build_lstm_model(LOOKBACK)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        restore_best_weights= True
    )
    
    model.fit(X_train, y_train, validation_data= (X_test, y_test), epochs=50, callbacks=[early_stopping], batch_size=32, verbose=1)
    
    model.save('src/models/lstm_demand_moel.korea')
    
    print(f'Trained LSTM for product {selected_product}')
    
if __name__ == '__main__':
    main()