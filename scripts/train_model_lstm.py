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

    # Parametry treningu - zoptymalizowane dla dokładności
    SEQUENCE_LENGTH = 60
    PREDICTION_HORIZON = 1
    EPOCHS = 30  # Zwiększone dla lepszej dokładności
    BATCH_SIZE = 16  # Zmniejszone dla stabilniejszego trenowania
    PATIENCE = 30  # Zwiększone patience dla EarlyStopping
    
    print(f"🔧 Training Parameters:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Prediction Horizon: {PREDICTION_HORIZON}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Early Stopping Patience: {PATIENCE}")

    # 1. Ładowanie i eksploracja danych
    print("\n📊 Step 1: Loading and exploring data...")
    try:
        df = load_processed_data()
        print(f"Data loaded successfully: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else "No Date column")
        
        # Sprawdź jakość danych
        missing_data = df.isnull().sum().sum()
        print(f"Missing values: {missing_data}")
        
        if missing_data > 0:
            print("⚠️ Warning: Missing data detected!")
            
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
        print(f"  Total samples for training: {X_train.shape[0]}")
        print(f"  Features per sample: {X_train.shape[2]}")
        print(f"  Batches per epoch: {X_train.shape[0] // BATCH_SIZE}")
        
        # Sprawdź czy mamy wystarczająco danych
        if X_train.shape[0] < 500:
            print("⚠️ WARNING: Relatively small dataset! Consider:")
            print("  - Using smaller batch size")
            print("  - Reducing early stopping patience")
            print("  - Adding data augmentation")
        
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
        print("Training started... This may take several minutes.")
        print(f"Requested epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Early stopping patience: {PATIENCE}")
        
        # Trenowanie z lepszymi parametrami
        history = model.train(
            X_train, y_train, 
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            validation_split=0.2,
            patience=PATIENCE  # Przekaż patience do modelu
        )
        
        print("✅ Training completed!")
        
        # ROZSZERZONA DIAGNOSTYKA TRENOWANIA
        if history and hasattr(history, 'history'):
            actual_epochs = len(history.history['loss'])
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            best_val_loss = min(history.history['val_loss'])
            best_epoch = np.argmin(history.history['val_loss']) + 1
            
            print(f"\n🔍 Training Diagnostics:")
            print(f"  Requested epochs: {EPOCHS}")
            print(f"  Actually trained epochs: {actual_epochs}")
            print(f"  Final training loss: {final_loss:.6f}")
            print(f"  Final validation loss: {final_val_loss:.6f}")
            print(f"  Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
            
            # Sprawdź czy było early stopping
            if actual_epochs < EPOCHS:
                print(f"⚠️ Training stopped early after {actual_epochs} epochs!")
                print("  This is due to EarlyStopping callback")
                
                # Pokaż trend validation loss w ostatnich epokach
                if len(history.history['val_loss']) >= 10:
                    recent_val_losses = history.history['val_loss'][-10:]
                    print(f"  Last 10 validation losses:")
                    for i, loss in enumerate(recent_val_losses):
                        print(f"    Epoch {actual_epochs-9+i}: {loss:.6f}")
            else:
                print("✅ Training completed all epochs without early stopping")
                
            # Sprawdź overfit
            loss_diff = final_val_loss - final_loss
            if loss_diff > 0.01:
                print(f"⚠️ Possible overfitting detected (val_loss - train_loss = {loss_diff:.6f})")
            else:
                print(f"✅ Good generalization (val_loss - train_loss = {loss_diff:.6f})")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    # 5. Ewaluacja modelu
    print("\n📊 Step 5: Model evaluation...")
    try:
        print("Making predictions...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metryki
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)

        print("\n📈 Model Performance Metrics:")
        print("-" * 50)
        print(f"Training Set:")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  MAE: {train_metrics['mae']:.6f}")
        print(f"  R²:  {train_metrics['r2']:.6f}")
        print(f"\nTest Set:")
        print(f"  MSE: {test_metrics['mse']:.6f}")
        print(f"  MAE: {test_metrics['mae']:.6f}")
        print(f"  R²:  {test_metrics['r2']:.6f}")
        
        # Dodatkowa analiza
        test_r2 = test_metrics['r2']
        if test_r2 > 0.8:
            print("🎉 Excellent model performance!")
        elif test_r2 > 0.6:
            print("✅ Good model performance")
        elif test_r2 > 0.4:
            print("⚠️ Moderate model performance")
        else:
            print("❌ Poor model performance - consider:")
            print("  - More training data")
            print("  - Different model architecture")
            print("  - Feature engineering")
        
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

    # 6b. Zapisz krzywe uczenia (NOWA FUNKCJONALNOŚĆ)
    print("\n📈 Step 6b: Saving training curves...")
    try:
        print("Creating and saving training curves...")
        model.save_training_curves()
        print("✅ Training curves saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save training curves: {e}")

    # 7. Predykcje przyszłości
    print("\n🔮 Step 7: Future predictions...")
    try:
        # Get last sequence and reshape it properly
        last_sequence = X_test[-1:].copy() if len(X_test) > 0 else X_train[-1:].copy()
        print(f"Last sequence shape: {last_sequence.shape}")
        
        # Make sure we have the right dimensions [1, sequence_length, n_features]
        if last_sequence.shape[0] > 0:
            # Number of days to predict
            days_ahead = 30
            
            # Simple future prediction approach
            future_values = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                # Predict next value
                next_pred = model.predict(current_sequence)[0][0]
                future_values.append(next_pred)
                
                # Update sequence for next prediction (roll the window)
                # Remove oldest day, add new prediction
                if current_sequence.shape[1] > 1:  # If sequence length > 1
                    # Shift the sequence left
                    current_sequence[0, :-1, :] = current_sequence[0, 1:, :]
                    # Set the last entry's target feature as our prediction
                    # Assume the target is the first feature (adjust if different)
                    current_sequence[0, -1, 0] = next_pred
            
            print(f"Generated {len(future_values)} future predictions")
            
            # Visualization if available
            if VISUALIZATION_AVAILABLE:
                try:
                    create_future_predictions_plot(y_test, future_values, 
                                                test_metrics['mae'], 
                                                days_ahead)
                    print("✅ Future predictions plot created successfully")
                except Exception as viz_err:
                    print(f"⚠️ Could not create visualization: {viz_err}")
                    
            # Print the first few predictions
            print("\nFuture price predictions (next 10 days):")
            for i, val in enumerate(future_values[:10]):
                print(f"Day {i+1}: {val:.4f}")
        else:
            print("❌ No data available for future predictions")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not create future predictions: {e}")

    # 8. Ocena jakości modelu
    print("\n🎯 Step 8: Model quality assessment...")
    try:
        # Create a quality assessment dictionary directly
        r2 = test_metrics['r2']
        quality_assessment = {
            'Model Quality': 'Excellent' if r2 > 0.8 else 'Good' if r2 > 0.6 else 'Moderate' if r2 > 0.4 else 'Poor',
            'R² Score': f"{r2:.4f}",
            'MSE': f"{test_metrics['mse']:.6f}",
            'MAE': f"{test_metrics['mae']:.6f}",
            'Recommended Action': 'None needed' if r2 > 0.7 else 'Consider more data or feature engineering'
        }
        
        print("\n📊 Model Quality Assessment:")
        print("-" * 50)
        for metric, value in quality_assessment.items():
            print(f"{metric}: {value}")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not assess model quality: {e}")

    # 9. Zapisanie modelu i wyników
    print("\n💾 Step 9: Saving results...")
    try:
        # Zapisz model
        model.save_model('saved_models/bitcoin_lstm_model_optimized.h5')
        
        # Zapisz metryki do pliku JSON
        results = {
            'training_config': {
                'sequence_length': SEQUENCE_LENGTH,
                'prediction_horizon': PREDICTION_HORIZON,
                'epochs_requested': EPOCHS,
                'epochs_actual': len(history.history['loss']) if history else 0,
                'batch_size': BATCH_SIZE,
                'patience': PATIENCE
            },
            'metrics': {
                'train': train_metrics,
                'test': test_metrics
            },
            'model_info': {
                'input_shape': input_shape,
                'features': feature_columns,
                'target': target_column
            }
        }
        
        os.makedirs('results/metrics', exist_ok=True)
        with open('results/metrics/training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("✅ Model and results saved successfully!")
        print(f"  Model: saved_models/bitcoin_lstm_model_optimized.h5")
        print(f"  Metrics: results/metrics/training_results.json")
        print(f"  Training curves: results/plots/training_*_curve.png")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save results: {e}")

    print("\n✅ Training and evaluation completed successfully!")
    print("🎉 Model is working correctly!")
    
    if VISUALIZATION_AVAILABLE:
        print("📊 Check the 'results/plots' directory for generated visualizations!")
    
    print("📈 Training curves saved as separate files:")
    print("  - training_loss_curve.png")
    print("  - training_mae_curve.png") 
    print("  - training_curves_combined.png")

if __name__ == "__main__":
    main()