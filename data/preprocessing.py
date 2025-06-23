import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import os

def preprocess_bitcoin_data(file_path, normalization_method='minmax'):
    """
    Preprocessuje dane Bitcoin z pliku CSV dla modeli uczenia maszynowego.
    
    Args:
        file_path (str): Ścieżka do pliku CSV
        normalization_method (str): 'minmax' lub 'standard' dla normalizacji
    
    Returns:
        pd.DataFrame: Przetworzone dane gotowe do uczenia maszynowego
    """
    
    # Wczytaj dane
    df = pd.read_csv(file_path)
    
    print(f"Kształt oryginalnych danych: {df.shape}")
    print(f"Kolumny: {df.columns.tolist()}")
    
    # 1. Konwersja daty i sortowanie
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date', ascending=True)
    df = df.reset_index(drop=True)
    
    print(f"Zakres dat: {df['Date'].min()} do {df['Date'].max()}")
    
    # 2. Funkcja do konwersji wartości numerycznych z przecinkami
    def convert_price_to_float(value):
        """Konwertuje stringi z przecinkami na float"""
        if isinstance(value, str):
            return float(value.replace(',', ''))
        return float(value)
    
    # 3. Konwersja kolumn cenowych
    price_columns = ['Price', 'Open', 'High', 'Low']
    for col in price_columns:
        df[col] = df[col].apply(convert_price_to_float)
    
    # 4. Konwersja kolumny Volume (obsługa K, M, B)
    def convert_volume(value):
        """Konwertuje volume z sufiksami K, M, B na liczby"""
        if isinstance(value, str):
            value = value.replace(',', '')
            if value.endswith('K'):
                return float(value[:-1]) * 1000
            elif value.endswith('M'):
                return float(value[:-1]) * 1000000
            elif value.endswith('B'):
                return float(value[:-1]) * 1000000000
            else:
                return float(value)
        return float(value)
    
    df['Vol.'] = df['Vol.'].apply(convert_volume)
    df['Volume'] = df['Vol.']  # Zmiana nazwy kolumny
    df = df.drop('Vol.', axis=1)
    
    # 5. Konwersja kolumny Change %
    def convert_percentage(value):
        """Konwertuje procenty na float (usuwa % i dzieli przez 100)"""
        if isinstance(value, str):
            value = value.replace('%', '')
            return float(value) / 100
        return float(value) / 100
    
    df['Change %'] = df['Change %'].apply(convert_percentage)
    df['Change_Pct'] = df['Change %']  # Zmiana nazwy kolumny
    df = df.drop('Change %', axis=1)
    
    # 6. Dodanie dodatkowych features
    # Volatile range (High - Low)
    df['Daily_Range'] = df['High'] - df['Low']
    
    # Price change from open to close
    df['Open_Close_Change'] = df['Price'] - df['Open']
    
    # Moving averages (jeśli mamy wystarczająco danych)
    if len(df) >= 7:
        df['MA_7'] = df['Price'].rolling(window=7).mean()
    if len(df) >= 30:
        df['MA_30'] = df['Price'].rolling(window=30).mean()
    
    # RSI (Relative Strength Index) - uproszczona wersja
    if len(df) >= 14:
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # 7. Obsługa wartości NaN (dla moving averages)
    # Wypełnij NaN dla moving averages poprzednimi wartościami
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    
    # 8. Normalizacja danych numerycznych
    numeric_columns = ['Price', 'Open', 'High', 'Low', 'Volume', 'Change_Pct', 
                      'Daily_Range', 'Open_Close_Change']
    
    # Dodaj kolumny MA i RSI jeśli istnieją
    if 'MA_7' in df.columns:
        numeric_columns.append('MA_7')
    if 'MA_30' in df.columns:
        numeric_columns.append('MA_30')
    if 'RSI' in df.columns:
        numeric_columns.append('RSI')
    
    # Stwórz kopię dla oryginalnych wartości
    df_original = df.copy()
    
    # Normalizacja
    if normalization_method == 'minmax':
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    elif normalization_method == 'standard':
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # 9. Dodanie features czasowych
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # 10. Sprawdzenie jakości danych
    print(f"\nPodsumowanie po preprocessingu:")
    print(f"Kształt danych: {df.shape}")
    print(f"Wartości NaN: {df.isnull().sum().sum()}")
    print(f"Zakres dat: {df['Date'].min()} do {df['Date'].max()}")
    print(f"Kolumny numeryczne: {numeric_columns}")
    
    return df, df_original, scaler

def create_sequences_for_lstm(data, target_column='Price', sequence_length=60):
    """
    Tworzy sekwencje danych dla modeli LSTM/RNN.
    
    Args:
        data (pd.DataFrame): Przetworzone dane
        target_column (str): Kolumna docelowa do predykcji
        sequence_length (int): Długość sekwencji
    
    Returns:
        np.array, np.array: X (sequences), y (targets)
    """
    
    # Wybierz kolumny numeryczne (bez daty i features czasowych)
    feature_columns = ['Price', 'Open', 'High', 'Low', 'Volume', 'Change_Pct', 
                      'Daily_Range', 'Open_Close_Change']
    
    # Dodaj MA i RSI jeśli istnieją
    if 'MA_7' in data.columns:
        feature_columns.append('MA_7')
    if 'MA_30' in data.columns:
        feature_columns.append('MA_30')
    if 'RSI' in data.columns:
        feature_columns.append('RSI')
    
    # Przygotuj dane
    values = data[feature_columns].values
    target_values = data[target_column].values
    
    X, y = [], []
    
    for i in range(sequence_length, len(values)):
        X.append(values[i-sequence_length:i])
        y.append(target_values[i])
    
    return np.array(X), np.array(y)

# Przykład użycia:
if __name__ == "__main__":
    # Stwórz katalog processed_data jeśli nie istnieje
    os.makedirs('processed_data', exist_ok=True)
    
    # Preprocessuj dane - poprawione ścieżki
    df_processed, df_original, scaler = preprocess_bitcoin_data(
        'data/raw/Bitcoin_Historical_Data.csv',  
        normalization_method='minmax'
    )
    
    # Wyświetl podstawowe statystyki
    print("\nPierwsze 5 wierszy przetworzonych danych:")
    print(df_processed.head())
    
    print("\nStatystyki opisowe dla głównych kolumn:")
    print(df_processed[['Price', 'Volume', 'Change_Pct']].describe())
    
    # Zapisz przetworzone dane
    df_processed.to_csv('data/processed_data/Bitcoin_Processed_Data.csv', index=False)
    df_original.to_csv('data/processed_data/Bitcoin_Original_Data.csv', index=False)
    
    print(f"\nDane zostały zapisane do:")
    print(f"- data/processed_data/Bitcoin_Processed_Data.csv (znormalizowane)")
    print(f"- data/processed_data/Bitcoin_Original_Data.csv (oryginalne wartości, ale przetworzone)")
    
    # Stwórz sekwencje dla LSTM (opcjonalnie)
    X_sequences, y_sequences = create_sequences_for_lstm(df_processed, sequence_length=30)
    print(f"\nSekwencje LSTM:")
    print(f"X shape: {X_sequences.shape}")
    print(f"y shape: {y_sequences.shape}")
    
    # Zapisz informacje o scalerze dla późniejszego użycia
    import joblib
    joblib.dump(scaler, 'data/processed_data/scaler.joblib')
    print(f"- data/processed_data/scaler.joblib (scaler do denormalizacji)")