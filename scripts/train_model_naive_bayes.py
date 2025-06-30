import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import warnings
import joblib
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# Sprawdź importy
try:
    from models.data_utils import (
        load_processed_data, prepare_lstm_data, 
        get_feature_columns, get_target_column
    )
    print("✅ data_utils imported successfully")
except ImportError as e:
    print(f"❌ Error importing data_utils: {e}")
    sys.exit(1)

try:
    from models.naive_bayes_model import BitcoinNaiveBayesModel
    print("✅ naive_bayes_model imported successfully")
except ImportError as e:
    print(f"❌ Error importing naive_bayes_model: {e}")
    sys.exit(1)

try:
    from models.prediction_utils import assess_naive_bayes_quality
    print("✅ prediction_utils imported successfully")
except ImportError as e:
    print(f"❌ Error importing prediction_utils: {e}")
    sys.exit(1)

# Flaga dla wizualizacji
VISUALIZATION_AVAILABLE = True
try:
    from visualization.naive_bayes_results import create_naive_bayes_results_plots
    from visualization.naive_bayes_data_exploration import (
        create_naive_bayes_data_exploration_plots, 
        print_feature_analysis_summary
    )
    print("✅ visualization modules imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

def main():
    print("🚀 Bitcoin Price Prediction using Naive Bayes")
    print("="*50)

    # Parametry treningu
    SEQUENCE_LENGTH = 60
    PREDICTION_HORIZON = 1
    CLASSIFICATION_TYPE = 'direction'  # 'direction' lub 'range'
    VAR_SMOOTHING = 1e-9
    
    print(f"🔧 Training Parameters:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Prediction Horizon: {PREDICTION_HORIZON}")
    print(f"  Classification Type: {CLASSIFICATION_TYPE}")
    print(f"  Variance Smoothing: {VAR_SMOOTHING}")

    # 1. Ładowanie i eksploracja danych
    print("\n📊 Step 1: Loading and exploring data...")
    try:
        df = load_processed_data()
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Columns: {df.columns.tolist()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Przygotowanie danych
    print("\n🔧 Step 2: Preparing data for Naive Bayes...")
    try:
        feature_columns = get_feature_columns()
        target_column = get_target_column()
        
        # Sprawdź dostępność kolumn
        available_features = [col for col in feature_columns if col in df.columns]
        if target_column not in df.columns:
            print(f"❌ Target column '{target_column}' not found in data")
            return
        
        print(f"Available features: {available_features}")
        print(f"Target column: {target_column}")
        
        # Przygotuj dane
        X_train, X_test, y_train, y_test = prepare_lstm_data(
            df, available_features, target_column, 
            SEQUENCE_LENGTH, PREDICTION_HORIZON
        )
        
        print(f"✅ Data prepared successfully!")
        print(f"  Training set: X={X_train.shape}, y={y_train.shape}")
        print(f"  Test set: X={X_test.shape}, y={y_test.shape}")
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return

    # 3. Budowa modelu Naive Bayes
    print("\n🏗️ Step 3: Building Naive Bayes model...")
    try:
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"Model input shape: {input_shape}")
        
        model = BitcoinNaiveBayesModel(input_shape, classification_type=CLASSIFICATION_TYPE)
        model.create_model(var_smoothing=VAR_SMOOTHING)
        
        print("\n📋 Model Architecture:")
        model.get_model_summary()
        
        # Zapisz architekturę modelu
        model.save_model_architecture()
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return

    # 4. Trenowanie modelu
    print("\n🏃‍♂️ Step 4: Training Naive Bayes model...")
    try:
        print("Training started...")
        
        history = model.train(
            X_train, y_train,
            validation_split=0.2
        )
        
        print("✅ Training completed!")
        
        # Zapisz model
        model.save_model('saved_models/bitcoin_naive_bayes_model_optimized.joblib')
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    # 5. Ewaluacja modelu
    print("\n📊 Step 5: Model evaluation...")
    try:
        print("Making predictions on test set...")
        
        # Przygotuj dane testowe
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        y_test_labels = model._create_labels(y_test, X_test)
        
        # Oblicz metryki klasyfikacyjne
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        test_accuracy = accuracy_score(y_test_labels, y_test_pred)
        test_precision = precision_score(y_test_labels, y_test_pred, average='weighted')
        test_recall = recall_score(y_test_labels, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test_labels, y_test_pred, average='weighted')
        
        test_metrics = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'classification_report': classification_report(y_test_labels, y_test_pred, 
                                                         target_names=model.class_names, 
                                                         output_dict=True)
        }
        
        print(f"\n📈 Test Set Performance:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test_labels, y_test_pred, target_names=model.class_names))
        
        # Macierz pomyłek
        print(f"\n🔍 Confusion Matrix:")
        cm = confusion_matrix(y_test_labels, y_test_pred)
        print(cm)
        
        # Ocena jakości modelu Naive Bayes
        quality_assessment = assess_naive_bayes_quality(test_accuracy, test_f1)
        print(f"\n🎯 Model Quality Assessment:")
        print(f"  Overall Quality: {quality_assessment['overall']}")
        print(f"  Accuracy Quality: {quality_assessment['accuracy_quality']}")
        print(f"  F1-Score Quality: {quality_assessment['f1_quality']}")
        
        # Zapisz metryki do pliku
        os.makedirs('results/metrics', exist_ok=True)
        with open('results/metrics/naive_bayes_results.json', 'w') as f:
            json.dump({
                'test_metrics': test_metrics,
                'quality_assessment': quality_assessment,
                'training_history': history,
                'model_params': {
                    'sequence_length': SEQUENCE_LENGTH,
                    'prediction_horizon': PREDICTION_HORIZON,
                    'classification_type': CLASSIFICATION_TYPE,
                    'var_smoothing': VAR_SMOOTHING,
                    'model_type': 'Naive Bayes'
                }
            }, f, indent=2, default=str)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return

    # 6. Wizualizacje
    if VISUALIZATION_AVAILABLE:
        print("\n📊 Step 6: Creating visualizations...")
        try:
            print("Creating Naive Bayes results plots...")
            create_naive_bayes_results_plots(
                model, y_test_labels, y_test_pred, y_test_proba, test_metrics,
                save_path='results/plots/naive_bayes/naive_bayes_results.png'
            )
            
            print("Creating Naive Bayes data exploration plots...")
            create_naive_bayes_data_exploration_plots(model)
            
            print("Printing feature analysis summary...")
            print_feature_analysis_summary(model)
            
            print("✅ Visualizations created successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create some visualizations: {e}")
    else:
        print("\n⚠️ Step 6: Skipping visualizations (modules not available)")

    # 6b. Zapisz krzywe uczenia
    print("\n📈 Step 6b: Saving training curves...")
    try:
        print("Creating and saving training curves...")
        model.save_training_curves()
        print("✅ Training curves saved successfully!")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save training curves: {e}")

    # 7. Predykcja na następny dzień
    print("\n🔮 Step 7: Next Day Price Direction Prediction...")
    try:
        # Ładowanie scalera do denormalizacji
        scaler = None
        try:
            scaler = joblib.load('data/processed_data/scaler.joblib')
            print("✅ Scaler loaded successfully")
        except:
            print("⚠️ Warning: Could not load scaler")
        
        # Pobierz ostatnie 60 dni danych
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        last_60_days = df_sorted.tail(SEQUENCE_LENGTH)
        
        print(f"Using last {len(last_60_days)} days for prediction:")
        if 'Date' in last_60_days.columns:
            print(f"Date range: {last_60_days['Date'].iloc[0]} to {last_60_days['Date'].iloc[-1]}")
        
        # Przygotuj dane wejściowe
        feature_data = last_60_days[available_features].values
        X_next = feature_data.reshape(1, SEQUENCE_LENGTH, len(available_features))
        
        print(f"Input sequence shape: {X_next.shape}")
        
        # Wykonaj predykcję
        next_day_prediction = model.predict(X_next)[0]
        next_day_probabilities = model.predict_proba(X_next)[0]
        
        print(f"\n🎯 NEXT DAY DIRECTION PREDICTION:")
        print("=" * 50)
        
        predicted_class = model.class_names[next_day_prediction]
        confidence = next_day_probabilities[next_day_prediction]
        
        print(f"🚀 Predicted Direction: {predicted_class}")
        print(f"📊 Confidence: {confidence:.2%}")
        
        print(f"\n📈 Class Probabilities:")
        for i, (class_name, prob) in enumerate(zip(model.class_names, next_day_probabilities)):
            print(f"  {class_name}: {prob:.2%}")
        
        # NOWA SEKCJA: Oszacowanie przewidywanej ceny
        print(f"\n💰 ESTIMATED PRICE PREDICTION:")
        print("=" * 50)
        
        # Pobierz ostatnią znaną cenę
        last_price_normalized = df_sorted[target_column].iloc[-1]
        
        if scaler is not None:
            try:
                # Denormalizuj ostatnią cenę
                dummy_data = np.zeros((1, len(scaler.feature_names_in_)))
                price_idx = list(scaler.feature_names_in_).index(target_column)
                dummy_data[0, price_idx] = last_price_normalized
                last_price_actual = scaler.inverse_transform(dummy_data)[0, price_idx]
                
                # Oszacuj zmianę ceny na podstawie kierunku i pewności
                if predicted_class == 'Wzrost':
                    # Dla wzrostu: używamy pewności do oszacowania siły wzrostu
                    estimated_change_pct = confidence * 0.05  # Maksymalnie 5% wzrostu przy 100% pewności
                    estimated_next_price = last_price_actual * (1 + estimated_change_pct)
                    trend_symbol = "📈"
                    trend_text = "BULLISH"
                else:  # Spadek
                    # Dla spadku: używamy pewności do oszacowania siły spadku
                    estimated_change_pct = -confidence * 0.05  # Maksymalnie 5% spadku przy 100% pewności
                    estimated_next_price = last_price_actual * (1 + estimated_change_pct)
                    trend_symbol = "📉"
                    trend_text = "BEARISH"
                
                estimated_change_dollar = estimated_next_price - last_price_actual
                
                print(f"📊 Current Bitcoin Price: ${last_price_actual:,.2f}")
                print(f"🚀 Estimated Next Day Price: ${estimated_next_price:,.2f}")
                print(f"📈 Estimated Change: ${estimated_change_dollar:+,.2f} ({estimated_change_pct*100:+.2f}%)")
                print(f"{trend_symbol} Trend: {trend_text}")
                
                # Dodatkowa analiza na podstawie pewności
                if confidence > 0.8:
                    strength = "STRONG"
                    reliability = "High reliability"
                elif confidence > 0.6:
                    strength = "MODERATE"
                    reliability = "Medium reliability"
                else:
                    strength = "WEAK"
                    reliability = "Low reliability"
                
                print(f"💪 Signal Strength: {strength} ({reliability})")
                
            except Exception as denorm_error:
                print(f"⚠️ Could not denormalize prices: {denorm_error}")
                print(f"📊 Current Price (normalized): {last_price_normalized:.6f}")
                print(f"🚀 Predicted Direction: {predicted_class} (confidence: {confidence:.2%})")
        else:
            print(f"📊 Current Price (normalized): {last_price_normalized:.6f}")
            print(f"🚀 Predicted Direction: {predicted_class} (confidence: {confidence:.2%})")
            print("⚠️ Cannot estimate actual price without scaler")
        
        # Dodatkowe informacje o pewności predykcji
        print(f"\n🔍 Prediction Details:")
        print(f"  Model Accuracy: {test_accuracy:.4f}")
        print(f"  Model F1-Score: {test_f1:.4f}")
        print(f"  Sequence Length Used: {SEQUENCE_LENGTH} days")
        print(f"  Features Used: {len(available_features)}")
        
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        print(f"  Prediction Confidence: {confidence_level}")
        
        if confidence > 0.8:
            print("  💡 Strategy: High confidence - strong directional signal")
        elif confidence > 0.6:
            print("  💡 Strategy: Medium confidence - proceed with caution")
        else:
            print("  💡 Strategy: Low confidence - consider additional analysis")
            
        print(f"  ⚠️ Disclaimer: This is an estimated prediction based on direction and confidence, not financial advice!")
        
    except Exception as e:
        print(f"❌ Error in next day prediction: {e}")

    print(f"\n🎉 Naive Bayes model training and evaluation completed successfully!")
    print(f"📁 Results saved to:")
    print(f"  - Model: saved_models/bitcoin_naive_bayes_model_optimized.joblib")
    print(f"  - Architecture: saved_models/naive_bayes_architecture.json")
    print(f"  - Metrics: results/metrics/naive_bayes_results.json")
    print(f"  - Plots: results/plots/naive_bayes/")

if __name__ == "__main__":
    main()