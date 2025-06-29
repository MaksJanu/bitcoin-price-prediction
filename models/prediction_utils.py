import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def calculate_metrics(y_true, y_pred):
    """Oblicza metryki ewaluacji modelu dla regresji (LSTM, Transformer)"""
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

def calculate_classification_metrics(y_true, y_pred, class_names=None):
    """Oblicza metryki ewaluacji modelu dla klasyfikacji (Naive Bayes)"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Szczegółowy raport klasyfikacji
    class_report = classification_report(
        y_true, y_pred, 
        target_names=class_names if class_names else None,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report
    }

def predict_future(model, last_sequence, days_ahead=7):
    """
    Przewiduje przyszłe wartości używając wytrenowanego modelu regresji
    
    Args:
        model: Wytrenowany model (LSTM lub Transformer)
        last_sequence: Ostatnia sekwencja danych
        days_ahead: Liczba dni do przewidzenia w przyszłość
    
    Returns:
        array: Przewidywane wartości na kolejne dni
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Przewiduj następny dzień
        next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)[0][0]
        predictions.append(next_pred)
        
        # Aktualizuj sekwencję (przesuń okno czasowe)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, -1] = next_pred  # Zastąp ostatnią wartość predykcją
    
    return np.array(predictions)

def assess_model_quality(r2_score, mae_score):
    """Ocenia jakość modelu regresji na podstawie R² i MAE"""
    
    # Ocena R²
    if r2_score >= 0.9:
        r2_quality = "Excellent"
    elif r2_score >= 0.8:
        r2_quality = "Very Good"
    elif r2_score >= 0.7:
        r2_quality = "Good"
    elif r2_score >= 0.6:
        r2_quality = "Fair"
    elif r2_score >= 0.4:
        r2_quality = "Poor"
    else:
        r2_quality = "Very Poor"
    
    # Ocena MAE (zakładając znormalizowane dane 0-1)
    if mae_score <= 0.02:
        mae_quality = "Excellent"
    elif mae_score <= 0.05:
        mae_quality = "Very Good"
    elif mae_score <= 0.1:
        mae_quality = "Good"
    elif mae_score <= 0.15:
        mae_quality = "Fair"
    elif mae_score <= 0.25:
        mae_quality = "Poor"
    else:
        mae_quality = "Very Poor"
    
    # Ogólna ocena (średnia ważona)
    r2_weight = 0.6
    mae_weight = 0.4
    
    # Konwertuj MAE na skalę 0-1 (odwrócona - mniejsza MAE = lepsza jakość)
    mae_normalized = max(0, 1 - (mae_score / 0.3))  # 0.3 jako maksymalny akceptowalny MAE
    
    overall_score = r2_weight * r2_score + mae_weight * mae_normalized
    
    if overall_score >= 0.85:
        overall = "Excellent"
    elif overall_score >= 0.75:
        overall = "Very Good"
    elif overall_score >= 0.65:
        overall = "Good"
    elif overall_score >= 0.55:
        overall = "Fair"
    elif overall_score >= 0.40:
        overall = "Poor"
    else:
        overall = "Very Poor"
    
    return {
        'overall': overall,
        'r2_quality': r2_quality,
        'mae_quality': mae_quality,
        'overall_score': overall_score
    }

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

def assess_naive_bayes_quality(accuracy, f1_score):
    """Ocenia jakość modelu Naive Bayes na podstawie dokładności i F1-score"""
    
    # Ocena dokładności
    if accuracy >= 0.9:
        accuracy_quality = "Excellent"
    elif accuracy >= 0.8:
        accuracy_quality = "Very Good"
    elif accuracy >= 0.7:
        accuracy_quality = "Good"
    elif accuracy >= 0.6:
        accuracy_quality = "Fair"
    elif accuracy >= 0.5:
        accuracy_quality = "Poor"
    else:
        accuracy_quality = "Very Poor"
    
    # Ocena F1-score
    if f1_score >= 0.9:
        f1_quality = "Excellent"
    elif f1_score >= 0.8:
        f1_quality = "Very Good"
    elif f1_score >= 0.7:
        f1_quality = "Good"
    elif f1_score >= 0.6:
        f1_quality = "Fair"
    elif f1_score >= 0.5:
        f1_quality = "Poor"
    else:
        f1_quality = "Very Poor"
    
    # Ogólna ocena (średnia ważona)
    accuracy_weight = 0.4
    f1_weight = 0.6  # F1 ważniejsze dla klasyfikacji
    
    overall_score = accuracy_weight * accuracy + f1_weight * f1_score
    
    if overall_score >= 0.85:
        overall = "Excellent"
    elif overall_score >= 0.75:
        overall = "Very Good"
    elif overall_score >= 0.65:
        overall = "Good"
    elif overall_score >= 0.55:
        overall = "Fair"
    elif overall_score >= 0.45:
        overall = "Poor"
    else:
        overall = "Very Poor"
    
    return {
        'overall': overall,
        'accuracy_quality': accuracy_quality,
        'f1_quality': f1_quality,
        'overall_score': overall_score
    }

def predict_price_direction_with_naive_bayes(model, last_sequence, scaler=None):
    """
    Przewiduje kierunek zmiany ceny używając modelu Naive Bayes
    
    Args:
        model: Wytrenowany model BitcoinNaiveBayesModel
        last_sequence: Ostatnia sekwencja danych (60 dni)
        scaler: Scaler do denormalizacji (opcjonalnie)
    
    Returns:
        dict: Słownik z predykcją kierunku i prawdopodobieństwami
    """
    try:
        # Reshape dla pojedynczej predykcji
        X_pred = last_sequence.reshape(1, *last_sequence.shape)
        
        # Przewiduj kierunek
        direction_pred = model.predict(X_pred)[0]
        probabilities = model.predict_proba(X_pred)[0]
        
        predicted_direction = model.class_names[direction_pred]
        confidence = probabilities[direction_pred]
        
        result = {
            'predicted_direction': predicted_direction,
            'direction_index': direction_pred,
            'confidence': confidence,
            'all_probabilities': {
                class_name: prob 
                for class_name, prob in zip(model.class_names, probabilities)
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error in direction prediction: {e}")
        return None

def get_model_recommendations(model_type, performance_metrics):
    """
    Zwraca rekomendacje dla modelu na podstawie jego wydajności
    
    Args:
        model_type: Typ modelu ('lstm', 'transformer', 'naive_bayes')
        performance_metrics: Słownik z metrykami wydajności
    
    Returns:
        dict: Słownik z rekomendacjami
    """
    recommendations = {
        'trading_strategy': '',
        'confidence_level': '',
        'risk_assessment': '',
        'improvement_suggestions': []
    }
    
    if model_type.lower() in ['lstm', 'transformer']:
        # Dla modeli regresji
        r2 = performance_metrics.get('r2', 0)
        mae = performance_metrics.get('mae', 1)
        
        if r2 >= 0.8:
            recommendations['trading_strategy'] = "High confidence signals - suitable for automated trading"
            recommendations['confidence_level'] = "High"
            recommendations['risk_assessment'] = "Low to Medium"
        elif r2 >= 0.6:
            recommendations['trading_strategy'] = "Medium confidence - use with additional analysis"
            recommendations['confidence_level'] = "Medium"
            recommendations['risk_assessment'] = "Medium"
        else:
            recommendations['trading_strategy'] = "Low confidence - use only for trend indication"
            recommendations['confidence_level'] = "Low"
            recommendations['risk_assessment'] = "High"
        
        # Sugestie poprawy dla modeli regresji
        if r2 < 0.7:
            recommendations['improvement_suggestions'].extend([
                "Consider adding more features (technical indicators, sentiment analysis)",
                "Experiment with different sequence lengths",
                "Try ensemble methods"
            ])
        
        if mae > 0.1:
            recommendations['improvement_suggestions'].extend([
                "Improve data preprocessing and normalization",
                "Add more training data",
                "Fine-tune hyperparameters"
            ])
    
    elif model_type.lower() == 'naive_bayes':
        # Dla modelu klasyfikacji
        accuracy = performance_metrics.get('accuracy', 0)
        f1 = performance_metrics.get('f1_score', 0)
        
        if accuracy >= 0.75 and f1 >= 0.75:
            recommendations['trading_strategy'] = "Reliable directional signals - good for trend following"
            recommendations['confidence_level'] = "High"
            recommendations['risk_assessment'] = "Medium"
        elif accuracy >= 0.6 and f1 >= 0.6:
            recommendations['trading_strategy'] = "Moderate directional signals - combine with other indicators"
            recommendations['confidence_level'] = "Medium"
            recommendations['risk_assessment'] = "Medium to High"
        else:
            recommendations['trading_strategy'] = "Weak signals - use with extreme caution"
            recommendations['confidence_level'] = "Low"
            recommendations['risk_assessment'] = "High"
        
        # Sugestie poprawy dla Naive Bayes
        if accuracy < 0.7:
            recommendations['improvement_suggestions'].extend([
                "Engineer better statistical features from price sequences",
                "Consider different classification thresholds",
                "Add more diverse features (volume, volatility patterns)"
            ])
        
        if f1 < 0.7:
            recommendations['improvement_suggestions'].extend([
                "Balance the dataset if classes are imbalanced",
                "Consider different classification strategies (multi-class vs binary)",
                "Experiment with feature selection techniques"
            ])
    
    return recommendations

def compare_models_performance(lstm_metrics=None, transformer_metrics=None, naive_bayes_metrics=None):
    """
    Porównuje wydajność różnych modeli
    
    Returns:
        dict: Słownik z wynikami porównania
    """
    comparison = {
        'best_regression_model': None,
        'best_classification_model': None,
        'overall_recommendation': '',
        'model_strengths': {}
    }
    
    # Porównaj modele regresji (LSTM vs Transformer)
    regression_models = {}
    
    if lstm_metrics:
        regression_models['LSTM'] = lstm_metrics.get('r2', 0)
    
    if transformer_metrics:
        regression_models['Transformer'] = transformer_metrics.get('r2', 0)
    
    if regression_models:
        best_regression = max(regression_models, key=regression_models.get)
        comparison['best_regression_model'] = best_regression
        comparison['model_strengths'][best_regression] = f"Best R² score: {regression_models[best_regression]:.4f}"
    
    # Ocena Naive Bayes
    if naive_bayes_metrics:
        comparison['best_classification_model'] = 'Naive Bayes'
        nb_score = naive_bayes_metrics.get('f1_score', 0)
        comparison['model_strengths']['Naive Bayes'] = f"Direction prediction F1: {nb_score:.4f}"
    
    # Ogólne rekomendacje
    if len(regression_models) > 0:
        best_r2 = max(regression_models.values())
        if best_r2 >= 0.8:
            comparison['overall_recommendation'] = f"Use {comparison['best_regression_model']} for price prediction - excellent performance"
        elif best_r2 >= 0.6:
            comparison['overall_recommendation'] = f"Use {comparison['best_regression_model']} for price prediction with caution"
        else:
            comparison['overall_recommendation'] = "Consider ensemble approach - individual models show limited performance"
    
    if naive_bayes_metrics and naive_bayes_metrics.get('f1_score', 0) >= 0.7:
        if comparison['overall_recommendation']:
            comparison['overall_recommendation'] += " + Use Naive Bayes for directional confirmation"
        else:
            comparison['overall_recommendation'] = "Use Naive Bayes for directional analysis only"
    
    return comparison

# Funkcje pomocnicze do kompatybilności wstecznej
def create_performance_summary(metrics, model_type):
    """Tworzy podsumowanie wydajności modelu"""
    if model_type.lower() in ['lstm', 'transformer']:
        quality = assess_model_quality(metrics.get('r2', 0), metrics.get('mae', 1))
        return {
            'model_type': model_type,
            'primary_metric': f"R² = {metrics.get('r2', 0):.4f}",
            'secondary_metric': f"MAE = {metrics.get('mae', 1):.6f}",
            'quality_assessment': quality['overall'],
            'performance_level': quality
        }
    elif model_type.lower() == 'naive_bayes':
        quality = assess_naive_bayes_quality(
            metrics.get('accuracy', 0), 
            metrics.get('f1_score', 0)
        )
        return {
            'model_type': model_type,
            'primary_metric': f"Accuracy = {metrics.get('accuracy', 0):.4f}",
            'secondary_metric': f"F1-Score = {metrics.get('f1_score', 0):.4f}",
            'quality_assessment': quality['overall'],
            'performance_level': quality
        }