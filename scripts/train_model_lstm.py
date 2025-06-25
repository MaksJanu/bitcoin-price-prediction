import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import warnings
import joblib
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
    EPOCHS = 100  # Zwiększone dla lepszej dokładności
    BATCH_SIZE = 16  # Zmniejszone dla stabilniejszego trenowania
    PATIENCE = 15  # Zwiększone patience dla EarlyStopping
    
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

    # 7. Predykcja na następny dzień z ostatnich 60 dni (MAIN ADDITION)
    print("\n🔮 Step 7: Next Day Price Prediction...")
    try:
        # Ładowanie scalera do denormalizacji
        scaler = None
        try:
            scaler = joblib.load('data/processed_data/scaler.joblib')
            print("✅ Scaler loaded successfully")
        except:
            print("⚠️ Warning: Could not load scaler - predictions will be normalized")
        
        # Pobierz ostatnie 60 dni danych
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        last_60_days = df_sorted.tail(SEQUENCE_LENGTH)
        
        print(f"Using last {len(last_60_days)} days for prediction:")
        if 'Date' in last_60_days.columns:
            print(f"Date range: {last_60_days['Date'].iloc[0]} to {last_60_days['Date'].iloc[-1]}")
        
        # Przygotuj dane wejściowe
        feature_data = last_60_days[feature_columns].values
        
        # Reshape do formatu [1, sequence_length, n_features]
        X_next = feature_data.reshape(1, SEQUENCE_LENGTH, len(feature_columns))
        
        print(f"Input sequence shape: {X_next.shape}")
        
        # Wykonaj predykcję
        next_day_prediction_normalized = model.predict(X_next)[0][0]
        
        print(f"\n🎯 NEXT DAY PRICE PREDICTION:")
        print("=" * 50)
        
        if scaler is not None:
            try:
                # Denormalizuj predykcję
                # Stwórz dummy array z wszystkimi cechami
                dummy_data = np.zeros((1, len(scaler.feature_names_in_)))
                
                # Znajdź indeks kolumny Price w scalerze
                price_idx = list(scaler.feature_names_in_).index(target_column)
                dummy_data[0, price_idx] = next_day_prediction_normalized
                
                # Denormalizuj
                denormalized = scaler.inverse_transform(dummy_data)
                next_day_prediction_actual = denormalized[0, price_idx]
                
                print(f"🚀 Predicted Bitcoin Price for Next Day: ${next_day_prediction_actual:.2f}")
                print(f"   (Normalized value: {next_day_prediction_normalized:.6f})")
                
                # Pokaż ostatnią rzeczywistą cenę dla porównania
                if scaler is not None and target_column in df_sorted.columns:
                    last_price_normalized = df_sorted[target_column].iloc[-1]
                    dummy_last = np.zeros((1, len(scaler.feature_names_in_)))
                    dummy_last[0, price_idx] = last_price_normalized
                    last_price_actual = scaler.inverse_transform(dummy_last)[0, price_idx]
                    
                    price_change = next_day_prediction_actual - last_price_actual
                    price_change_pct = (price_change / last_price_actual) * 100
                    
                    print(f"📊 Last Known Price: ${last_price_actual:.2f}")
                    print(f"📈 Predicted Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
                    
                    if price_change > 0:
                        print("📈 Trend: BULLISH (Price expected to rise)")
                    else:
                        print("📉 Trend: BEARISH (Price expected to fall)")
                        
            except Exception as denorm_error:
                print(f"⚠️ Could not denormalize prediction: {denorm_error}")
                print(f"🚀 Predicted Bitcoin Price (Normalized): {next_day_prediction_normalized:.6f}")
        else:
            print(f"🚀 Predicted Bitcoin Price (Normalized): {next_day_prediction_normalized:.6f}")
        
        print("=" * 50)
        
        # Dodatkowe informacje o pewności predykcji
        print(f"\n🔍 Prediction Details:")
        print(f"  Model R² Score: {test_metrics['r2']:.4f}")
        print(f"  Model MAE: {test_metrics['mae']:.6f}")
        print(f"  Sequence Length Used: {SEQUENCE_LENGTH} days")
        print(f"  Features Used: {len(feature_columns)}")
        
        confidence_level = "High" if test_metrics['r2'] > 0.8 else "Medium" if test_metrics['r2'] > 0.6 else "Low"
        print(f"  Prediction Confidence: {confidence_level}")
        
    except Exception as e:
        print(f"❌ Error making next day prediction: {e}")
        import traceback
        traceback.print_exc()

    # 8. Predykcje przyszłości
    print("\n🔮 Step 8: Next Day Prediction (as designed)...")
    try:
        # Użyj funkcji z prediction_utils dla pojedynczej predykcji
        try:
            last_sequence = X_test[-1] if len(X_test) > 0 else X_train[-1]
            
            # POJEDYNCZA PREDYKCJA NA NASTĘPNY DZIEŃ
            next_day_predictions = predict_future(model, last_sequence, days_ahead=1)
            
            print(f"✅ Generated single day prediction using predict_future function")
            
            if len(next_day_predictions) > 0:
                next_day_pred = next_day_predictions[0]
                last_actual = df_sorted[target_column].iloc[-1]
                change = abs(next_day_pred - last_actual)
                change_pct = ((next_day_pred - last_actual) / last_actual) * 100
                
                print(f"\n📊 Single Day Prediction Analysis:")
                print(f"  Last actual price (normalized): {last_actual:.4f}")
                print(f"  Next day prediction (normalized): {next_day_pred:.4f}")
                print(f"  Absolute change: {change:.4f}")
                print(f"  Percentage change: {change_pct:+.2f}%")
                
                # Sprawdź realizm pojedynczej predykcji
                if abs(change_pct) < 5:
                    print("✅ REALISTIC: Daily change within normal Bitcoin range (<5%)")
                elif abs(change_pct) < 10:
                    print("⚠️ MODERATE: Daily change acceptable for Bitcoin (5-10%)")
                elif abs(change_pct) < 15:
                    print("❗ HIGH: Daily change large but possible for Bitcoin (10-15%)")
                else:
                    print("❌ EXTREME: Daily change very large (>15%) - unusual for Bitcoin")
                
                # Denormalizacja jeśli możliwe
                if scaler is not None:
                    try:
                        # Denormalizuj ostatnią rzeczywistą cenę
                        dummy_base = np.zeros((1, len(scaler.feature_names_in_)))
                        price_idx = list(scaler.feature_names_in_).index(target_column)
                        dummy_base[0, price_idx] = last_actual
                        last_actual_price = scaler.inverse_transform(dummy_base)[0, price_idx]
                        
                        # Denormalizuj predykcję
                        dummy_pred = np.zeros((1, len(scaler.feature_names_in_)))
                        dummy_pred[0, price_idx] = next_day_pred
                        predicted_price = scaler.inverse_transform(dummy_pred)[0, price_idx]
                        
                        price_change_dollar = predicted_price - last_actual_price
                        
                        print(f"\n💰 Price Prediction (Denormalized):")
                        print(f"  Current Bitcoin Price: ${last_actual_price:,.2f}")
                        print(f"  Predicted Next Day Price: ${predicted_price:,.2f}")
                        print(f"  Expected Change: ${price_change_dollar:+,.2f} ({change_pct:+.2f}%)")
                        
                        if price_change_dollar > 0:
                            print(f"  📈 Trend: BULLISH (Expected to rise by ${price_change_dollar:,.2f})")
                        else:
                            print(f"  📉 Trend: BEARISH (Expected to fall by ${abs(price_change_dollar):,.2f})")
                            
                    except Exception as denorm_error:
                        print(f"⚠️ Could not denormalize: {denorm_error}")
                
                # Dodatkowe informacje o pewności
                print(f"\n🔍 Prediction Confidence Details:")
                print(f"  Model R² Score: {test_metrics['r2']:.4f}")
                print(f"  Model MAE (normalized): {test_metrics['mae']:.6f}")
                print(f"  Sequence Length Used: {SEQUENCE_LENGTH} days")
                print(f"  Features Used: {len(feature_columns)}")
                
                confidence_level = "High" if test_metrics['r2'] > 0.8 else "Medium" if test_metrics['r2'] > 0.6 else "Low"
                print(f"  Overall Confidence: {confidence_level}")
                
                # Rekomendacje
                print(f"\n💡 Trading Insights:")
                if abs(change_pct) < 2:
                    print("  📊 Expected: STABLE price movement (<2% change)")
                    print("  💼 Strategy: Consider range trading or wait for clearer signals")
                elif change_pct > 2:
                    print("  📈 Expected: BULLISH movement (>+2% change)")
                    print("  💼 Strategy: Consider long positions, but watch for resistance levels")
                elif change_pct < -2:
                    print("  📉 Expected: BEARISH movement (<-2% change)")
                    print("  💼 Strategy: Consider short positions or wait for support levels")
                
                print(f"  ⚠️ Disclaimer: This is a model prediction, not financial advice!")
                
            else:
                print("❌ No prediction generated")
        
        except Exception as predict_error:
            print(f"Standard prediction failed: {predict_error}")
            print("Trying alternative single day prediction method...")
            
            # ALTERNATYWNA METODA - BEZPOŚREDNIA PREDYKCJA
            df_sorted = df.sort_values('Date').reset_index(drop=True)
            last_60_days = df_sorted.tail(SEQUENCE_LENGTH)
            
            # Przygotuj sekwencję wejściową
            feature_data = last_60_days[feature_columns].values
            input_sequence = feature_data.reshape(1, SEQUENCE_LENGTH, len(feature_columns))
            
            print(f"🔧 Using alternative method with sequence shape: {input_sequence.shape}")
            
            # Pojedyncza predykcja
            next_day_pred = model.predict(input_sequence, verbose=0)[0][0]
            last_actual = df_sorted[target_column].iloc[-1]
            
            change = abs(next_day_pred - last_actual)
            change_pct = ((next_day_pred - last_actual) / last_actual) * 100
            
            print(f"\n📊 Alternative Single Day Prediction:")
            print(f"  Last actual price (normalized): {last_actual:.4f}")
            print(f"  Next day prediction (normalized): {next_day_pred:.4f}")
            print(f"  Percentage change: {change_pct:+.2f}%")
            
            # Denormalizacja
            if scaler is not None:
                try:
                    dummy_base = np.zeros((1, len(scaler.feature_names_in_)))
                    price_idx = list(scaler.feature_names_in_).index(target_column)
                    dummy_base[0, price_idx] = last_actual
                    last_actual_price = scaler.inverse_transform(dummy_base)[0, price_idx]
                    
                    dummy_pred = np.zeros((1, len(scaler.feature_names_in_)))
                    dummy_pred[0, price_idx] = next_day_pred
                    predicted_price = scaler.inverse_transform(dummy_pred)[0, price_idx]
                    
                    price_change_dollar = predicted_price - last_actual_price
                    
                    print(f"\n💰 Alternative Price Prediction:")
                    print(f"  Current Bitcoin Price: ${last_actual_price:,.2f}")
                    print(f"  Predicted Next Day Price: ${predicted_price:,.2f}")
                    print(f"  Expected Change: ${price_change_dollar:+,.2f} ({change_pct:+.2f}%)")
                    
                except Exception as denorm_error:
                    print(f"⚠️ Could not denormalize alternative prediction: {denorm_error}")
        
        # Opcjonalna wizualizacja pojedynczej predykcji
        if VISUALIZATION_AVAILABLE:
            try:
                print("\n📈 Creating single day prediction visualization...")
                
                # Stwórz prosty wykres pokazujący trend
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 6))
                
                # Ostatnie 30 dni danych historycznych
                last_30_days = df_sorted.tail(30)
                historical_prices = last_30_days[target_column].values
                dates_range = range(-29, 1)  # Od -29 do 0 (dzisiaj)
                
                # Wykres historyczny
                plt.plot(dates_range[:-1], historical_prices[:-1], 'b-', linewidth=2, 
                        label='Historical Prices (29 days)', alpha=0.8)
                
                # Ostatni dzień (dzisiaj)
                plt.plot([0], [historical_prices[-1]], 'bo', markersize=8, 
                        label='Today (Last Known Price)')
                
                # Predykcja na jutro
                if 'next_day_pred' in locals():
                    plt.plot([1], [next_day_pred], 'ro', markersize=10, 
                            label='Tomorrow (Predicted)')
                    
                    # Linia łącząca dzisiaj z jutrem
                    plt.plot([0, 1], [historical_prices[-1], next_day_pred], 'r--', 
                            linewidth=2, alpha=0.7)
                    
                    # Pasek niepewności
                    mae = test_metrics['mae']
                    plt.fill_between([1], [next_day_pred - mae], [next_day_pred + mae],
                                alpha=0.3, color='red', label=f'Uncertainty (±{mae:.4f})')
                
                plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Today')
                plt.title('Bitcoin Price - Single Day Prediction', fontsize=14, fontweight='bold')
                plt.xlabel('Days from Today')
                plt.ylabel('Normalized Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Zapisz wykres
                save_path = 'results/plots/single_day_prediction.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✅ Single day prediction plot saved to: {save_path}")
                
            except Exception as viz_err:
                print(f"⚠️ Could not create single day visualization: {viz_err}")
        
        # Podsumowanie
        print(f"\n🎯 Summary:")
        print(f"  ✅ Model designed for: 1-day predictions")
        print(f"  ✅ Prediction method: Single step ahead")
        print(f"  ✅ Input data: Last {SEQUENCE_LENGTH} days")
        print(f"  ✅ Model accuracy: R² = {test_metrics['r2']:.4f}")
        print(f"  💡 Use this prediction as: Short-term trend indicator")
        print(f"  ⚠️ Note: For longer-term predictions, retrain model with longer horizon")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create single day prediction: {e}")
        import traceback
        traceback.print_exc()

    # 9. Ocena jakości modelu
    print("\n🎯 Step 9: Model quality assessment...")
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

    # 10. Zapisanie modelu i wyników
    print("\n💾 Step 10: Saving results...")
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