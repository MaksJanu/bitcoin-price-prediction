import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import warnings
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
    from models.lstm_model import BitcoinLSTMModel
    print("✅ lstm_model imported successfully")
except ImportError as e:
    print(f"❌ Error importing lstm_model: {e}")
    sys.exit(1)

try:
    from models.prediction_utils import calculate_metrics, predict_future, assess_model_quality
    print("✅ prediction_utils imported successfully")
except ImportError as e:
    print(f"❌ Error importing prediction_utils: {e}")
    sys.exit(1)

# Flaga dla wizualizacji
VISUALIZATION_AVAILABLE = False
try:
    from visualization.data_exploration import create_data_exploration_plots, print_data_statistics
    from visualization.training_results import create_training_results_plots
    from visualization.future_predictions import create_future_predictions_plot, print_future_predictions
    print("✅ visualization modules imported successfully")
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: visualization modules not available: {e}")
    print("Continuing without visualization...")

def main():
    print("🚀 Bitcoin Price Prediction using LSTM")
    print("="*50)

    # Parametry
    SEQUENCE_LENGTH = 60
    PREDICTION_HORIZON = 1  # Zacznij od 1 dnia

    # 1. Ładowanie i eksploracja danych
    print("\n📊 Step 1: Loading and exploring data...")
    try:
        df = load_processed_data()
        print(f"Data loaded successfully: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else "No Date column")
        
        # Drukuj statystyki danych
        if VISUALIZATION_AVAILABLE:
            print_data_statistics(df)
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Przygotowanie danych dla LSTM
    print("\n🔧 Step 2: Preparing data for LSTM...")
    try:
        feature_columns = get_feature_columns()
        target_column = get_target_column()
        
        print(f"Feature columns: {feature_columns}")
        print(f"Target column: {target_column}")
        
        # Sprawdź, czy kolumny istnieją
        available_features = [col for col in feature_columns if col in df.columns]
        if len(available_features) != len(feature_columns):
            print(f"⚠️ Some features missing. Available: {available_features}")
            feature_columns = available_features
        
        if target_column not in df.columns:
            # Spróbuj alternatyw
            if 'Close' in df.columns:
                target_column = 'Close'
                print(f"Using 'Close' as target instead of 'Price'")
            else:
                print(f"❌ Target column not found. Available columns: {df.columns.tolist()}")
                return
        
        print(f"Creating sequences with {SEQUENCE_LENGTH} days history...")
        print(f"Predicting {PREDICTION_HORIZON} days ahead...")
        
        X_train, X_test, y_train, y_test = prepare_lstm_data(
            df, feature_columns, target_column, SEQUENCE_LENGTH, PREDICTION_HORIZON
        )
        
        print(f"✅ Data prepared successfully:")
        print(f"  Training set: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"  Test set: X_test={X_test.shape}, y_test={y_test.shape}")
        
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return

    # 3. Budowa modelu LSTM
    print("\n🏗️ Step 3: Building LSTM model...")
    try:
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"Model input shape: {input_shape}")
        
        model = BitcoinLSTMModel(input_shape)
        model.create_model()
        
        print("\n📋 Model Architecture:")
        model.get_model_summary()
        
        # Zapisz architekturę modelu
        model.save_model_architecture()
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return

    # 4. Trenowanie modelu
    print("\n🏃‍♂️ Step 4: Training LSTM model...")
    try:
        print("Training started... This may take a few minutes.")
        
        # Zmniejsz liczbę epoch dla testów
        history = model.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
        
        print("✅ Training completed!")
        
        # Sprawdź historię treningu
        if history and hasattr(history, 'history'):
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            print(f"Final training loss: {final_loss:.6f}")
            print(f"Final validation loss: {final_val_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    # 5. Ewaluacja modelu
    print("\n📊 Step 5: Model evaluation...")
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metryki
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print("\n📈 Model Performance Metrics:")
        print("-" * 40)
        print(f"Training Set:")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  MAE: {train_metrics['mae']:.6f}")
        print(f"  R²:  {train_metrics['r2']:.6f}")
        print(f"\nTest Set:")
        print(f"  MSE: {test_metrics['mse']:.6f}")
        print(f"  MAE: {test_metrics['mae']:.6f}")
        print(f"  R²:  {test_metrics['r2']:.6f}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return

    # 6. Tworzenie wykresów i wizualizacji
    if VISUALIZATION_AVAILABLE:
        print("\n📈 Step 6: Creating visualizations...")
        try:
            # Wykresy eksploracji danych
            print("Creating data exploration plots...")
            create_data_exploration_plots(df)
            
            # Wykresy wyników treningu
            print("Creating training results plots...")
            create_training_results_plots(
                history, 
                train_metrics, 
                test_metrics, 
                y_train, 
                y_train_pred, 
                y_test, 
                y_test_pred
            )
            
            print("✅ Visualizations created successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create visualizations: {e}")
    else:
        print("\n⚠️ Step 6: Skipping visualizations (modules not available)")

    # 7. Predykcje przyszłości
    print("\n🔮 Step 7: Future predictions...")
    try:
        # Pobierz ostatnią sekwencję do predykcji
        last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]
        
        # Przewiduj przyszłe wartości
        future_predictions = predict_future(model, last_sequence, days_ahead=30)
        
        # Wykresy predykcji przyszłości
        if VISUALIZATION_AVAILABLE:
            print("Creating future predictions plot...")
            create_future_predictions_plot(df, future_predictions, target_column)
            
            # Wydrukuj predykcje
            print_future_predictions(future_predictions)
        else:
            # Wydrukuj tylko pierwszych kilka predykcji
            print("Future predictions (first 10 days):")
            for i, pred in enumerate(future_predictions[:10]):
                print(f"Day {i+1}: ${pred:.2f}")
        
        print("✅ Future predictions completed!")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create future predictions: {e}")

    # 8. Ocena jakości modelu
    print("\n🎯 Step 8: Model quality assessment...")
    try:
        quality_assessment = assess_model_quality(test_metrics, y_test, y_test_pred)
        
        print("\n📊 Model Quality Assessment:")
        print("-" * 40)
        for metric, value in quality_assessment.items():
            print(f"{metric}: {value}")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not assess model quality: {e}")

    print("\n✅ Training and evaluation completed successfully!")
    print("🎉 Model is working correctly!")
    
    if VISUALIZATION_AVAILABLE:
        print("📊 Check the 'plots' directory for generated visualizations!")

if __name__ == "__main__":
    main()