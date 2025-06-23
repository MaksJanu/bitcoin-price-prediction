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
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
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
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Aktualizuj sekwencję (prosty sposób - kopiuj ostatni wiersz)
        new_row = current_sequence[-1].copy()
        new_row[-1] = next_pred[0, 0]  # Zaktualizuj cenę
        
        # Przesuń sekwencję
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return np.array(predictions)

def assess_model_quality(r2_score):
    """Ocenia jakość modelu na podstawie R²"""
    if r2_score > 0.8:
        return "Excellent performance (R² > 0.8)", "✅"
    elif r2_score > 0.6:
        return "Good performance (R² > 0.6)", "✅"
    elif r2_score > 0.4:
        return "Moderate performance (R² > 0.4)", "⚠️"
    else:
        return "Poor performance (R² < 0.4)", "❌"