import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_processed_data(file_path='data/processed_data/Bitcoin_Processed_Data.csv'):
    """Ładuje przetworzone dane Bitcoin"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def create_sequences(data, seq_length, pred_horizon=1):
    X, y = [], []
    for i in range(seq_length, len(data) - pred_horizon + 1):
        # Sekwencja wejściowa (wszystkie cechy)
        X.append(data[i-seq_length:i, :-1])  # Wszystkie kolumny oprócz ostatniej (Price)
        # Wartość docelowa (tylko Price)
        y.append(data[i+pred_horizon-1, -1])  # Ostatnia kolumna (Price)
    return np.array(X), np.array(y)

def prepare_lstm_data(df, feature_columns, target_column, sequence_length, prediction_horizon):
    """Przygotowuje dane dla modelu LSTM"""
    # Przygotuj dane
    data = df[feature_columns + [target_column]].values
    
    # Stwórz sekwencje
    X, y = create_sequences(data, sequence_length, prediction_horizon)
    
    # Podział danych na treningowe i testowe
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

def get_feature_columns():
    """Zwraca listę kolumn z cechami"""
    return ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'RSI']

def get_target_column():
    """Zwraca nazwę kolumny docelowej"""
    return 'Price'