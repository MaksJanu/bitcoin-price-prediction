import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Oblicza metryki ewaluacji modelu"""
    # Spłaszcz predykcje jeśli potrzeba
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': float(rmse), 
        'r2': r2
    }

def predict_future(model, last_sequence, days_ahead=7):
    """
    Predykcja przyszłych cen na podstawie ostatniej sekwencji
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Predykcja następnego dnia
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
        predictions.append(next_pred[0, 0])
        
        # Aktualizuj sekwencję (prosty sposób - kopiuj ostatni wiersz)
        new_row = current_sequence[-1].copy()
        new_row[-1] = next_pred[0, 0]  # Zaktualizuj cenę
        
        # Przesuń sekwencję
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(predictions)

def assess_model_quality(r2_score, mae=None):
    """
    Ocenia jakość modelu na podstawie R² i opcjonalnie MAE
    
    Args:
        r2_score (float): Współczynnik determinacji R²
        mae (float, optional): Mean Absolute Error
    
    Returns:
        dict: Słownik z oceną jakości modelu
    """
    # Ocena na podstawie R²
    if r2_score > 0.8:
        r2_quality = "Excellent"
        r2_icon = "✅"
    elif r2_score > 0.6:
        r2_quality = "Good"
        r2_icon = "✅"
    elif r2_score > 0.4:
        r2_quality = "Moderate"
        r2_icon = "⚠️"
    else:
        r2_quality = "Poor"
        r2_icon = "❌"
    
    # Ocena na podstawie MAE (jeśli podane)
    mae_quality = "Unknown"
    mae_icon = "❓"
    if mae is not None:
        if mae < 0.01:
            mae_quality = "Excellent"
            mae_icon = "✅"
        elif mae < 0.05:
            mae_quality = "Good"
            mae_icon = "✅"
        elif mae < 0.1:
            mae_quality = "Moderate"
            mae_icon = "⚠️"
        else:
            mae_quality = "Poor"
            mae_icon = "❌"
    
    # Ogólna ocena
    if r2_score > 0.7 and (mae is None or mae < 0.05):
        overall = "Excellent"
    elif r2_score > 0.5 and (mae is None or mae < 0.1):
        overall = "Good"
    elif r2_score > 0.3:
        overall = "Moderate"
    else:
        overall = "Poor"
    
    return {
        'overall': overall,
        'r2_quality': r2_quality,
        'r2_icon': r2_icon,
        'mae_quality': mae_quality,
        'mae_icon': mae_icon,
        'r2_score': r2_score,
        'mae_score': mae if mae is not None else 'N/A'
    }

# Zachowaj kompatybilność wsteczną dla LSTM
def assess_model_quality_simple(r2_score):
    """Ocenia jakość modelu na podstawie R² - wersja dla LSTM"""
    if r2_score > 0.8:
        return "Excellent performance (R² > 0.8)", "✅"
    elif r2_score > 0.6:
        return "Good performance (R² > 0.6)", "✅"
    elif r2_score > 0.4:
        return "Moderate performance (R² > 0.4)", "⚠️"
    else:
        return "Poor performance (R² < 0.4)", "❌"