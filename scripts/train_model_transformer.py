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
    from models.transformer_model import BitcoinTransformerModel
    print("✅ transformer_model imported successfully")
except ImportError as e:
    print(f"❌ Error importing transformer_model: {e}")
    sys.exit(1)

try:
    from models.prediction_utils import calculate_metrics, predict_future, assess_model_quality
    print("✅ prediction_utils imported successfully")
except ImportError as e:
    print(f"❌ Error importing prediction_utils: {e}")
    sys.exit(1)

# Flaga dla wizualizacji
VISUALIZATION_AVAILABLE = True
try:
    from visualization.training_results import create_training_results_plots
    from visualization.future_predictions import create_future_predictions_plot
    from visualization.data_exploration import create_data_exploration_plots
    print("✅ visualization modules imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

def create_single_day_prediction_plot(df, model, test_metrics, sequence_length, feature_columns, target_column, scaler=None):
    """Tworzy wykres pojedynczej predykcji na następny dzień dla Transformer"""
    try:
        import matplotlib.pyplot as plt
        
        # Sortuj dane i pobierz ostatnie dni
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        last_30_days = df_sorted.tail(30)
        
        # Przygotuj dane dla predykcji
        last_60_days = df_sorted.tail(sequence_length)
        feature_data = last_60_days[feature_columns].values
        X_next = feature_data.reshape(1, sequence_length, len(feature_columns))
        
        # Wykonaj predykcję
        next_day_pred_normalized = model.predict(X_next)[0][0]
        
        # Stwórz wykres
        plt.figure(figsize=(12, 6))
        
        # Ostatnie 30 dni danych historycznych
        historical_prices = last_30_days[target_column].values
        dates_range = range(-29, 1)  # Od -29 do 0 (dzisiaj)
        
        # Wykres historyczny
        plt.plot(dates_range[:-1], historical_prices[:-1], 'b-', linewidth=2, 
                label='Historical Prices (29 days)', alpha=0.8)
        
        # Ostatni dzień (dzisiaj)
        plt.plot([0], [historical_prices[-1]], 'bo', markersize=8, 
                label='Today (Last Known Price)')
        
        # Predykcja na jutro
        plt.plot([1], [next_day_pred_normalized], 'ro', markersize=10, 
                label='Tomorrow (Transformer Predicted)')
        
        # Linia łącząca dzisiaj z jutrem
        plt.plot([0, 1], [historical_prices[-1], next_day_pred_normalized], 'r--', 
                linewidth=2, alpha=0.7)
        
        # Pasek niepewności
        mae = test_metrics['mae']
        plt.fill_between([1], [next_day_pred_normalized - mae], [next_day_pred_normalized + mae],
                        alpha=0.3, color='red', label=f'Uncertainty (±{mae:.4f})')
        
        plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Today')
        plt.title('Bitcoin Price - Transformer Single Day Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Days from Today')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Zapisz wykres
        os.makedirs('results/plots/transformer', exist_ok=True)
        single_day_path = 'results/plots/transformer/single_day_prediction.png'
        plt.savefig(single_day_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Transformer single day prediction saved to: {single_day_path}")
        
        return single_day_path
        
    except Exception as e:
        print(f"⚠️ Could not create single day prediction plot: {e}")
        return None

def main():
    print("🚀 Bitcoin Price Prediction using Stabilized Transformer")
    print("="*60)

    # Parametry treningu - zoptymalizowane dla stabilności
    SEQUENCE_LENGTH = 60
    PREDICTION_HORIZON = 1
    EPOCHS = 100
    BATCH_SIZE = 32
    PATIENCE = 20  # Zwiększone dla stabilności
    
    print(f"🔧 Training Parameters:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Prediction Horizon: {PREDICTION_HORIZON}")
    print(f"  Max Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Early Stopping Patience: {PATIENCE}")
    print(f"  Stabilization: BatchNorm, reduced dropout, gradient clipping")

    # 1. Ładowanie danych (BEZ WYKRESÓW EKSPLORACJI - będą na końcu)
    print("\n📊 Step 1: Loading data...")
    try:
        df = load_processed_data()
        print(f"✅ Data loaded successfully: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Sprawdź jakość danych
        missing_data = df.isnull().sum().sum()
        print(f"Missing values: {missing_data}")
        
        if missing_data > 0:
            print("⚠️ Warning: Missing data detected!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Przygotowanie danych
    print("\n🔧 Step 2: Preparing data for Transformer...")
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

    # 3. Budowa modelu Transformer
    print("\n🏗️ Step 3: Building Stabilized Transformer model...")
    try:
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"Model input shape: {input_shape}")
        
        model = BitcoinTransformerModel(input_shape)
        # Stabilne parametry
        model.create_model(
            head_size=128,      # Zmniejszony z 256
            num_heads=4,        # Pozostało
            ff_dim=2,          # Zmniejszony z 4
            num_transformer_blocks=3,  # Zmniejszony z 4
            mlp_units=[64, 32], # Zmniejszone z [128, 64]
            dropout=0.2,       # Zmniejszony z 0.3
            mlp_dropout=0.3,   # Zmniejszony z 0.4
            learning_rate=0.0005  # Zmniejszony z 0.001
        )
        
        print("\n📋 Model Architecture:")
        model.get_model_summary()
        
        # Zapisz architekturę modelu
        model.save_model_architecture()
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return

    # 4. Trenowanie modelu
    print("\n🏃‍♂️ Step 4: Training Stabilized Transformer model...")
    try:
        print("Training started... This may take several minutes.")
        print(f"Requested epochs: {EPOCHS}")
        print(f"Batch size: {BATCH_SIZE}")
        print("Stabilization features: BatchNorm, reduced complexity, conservative learning")
        
        history = model.train(
            X_train, y_train,
            validation_split=0.2,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patience=PATIENCE
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
            
            # Sprawdź stabilność trenowania
            if len(history.history['val_loss']) >= 10:
                recent_val_losses = history.history['val_loss'][-10:]
                val_loss_std = np.std(recent_val_losses)
                print(f"  Validation loss stability (last 10 epochs std): {val_loss_std:.6f}")
                
                if val_loss_std < 0.001:
                    print("✅ Very stable training (low variance in recent epochs)")
                elif val_loss_std < 0.01:
                    print("✅ Stable training")
                else:
                    print("⚠️ Some instability detected in recent epochs")
            
            # Sprawdź czy było early stopping
            if actual_epochs < EPOCHS:
                print(f"⚠️ Training stopped early after {actual_epochs} epochs!")
                print("  This is due to EarlyStopping callback")
            else:
                print("✅ Training completed all epochs without early stopping")
        
        # Zapisz model
        model.save_model('saved_models/bitcoin_transformer_model_stabilized.h5')
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return

    # 5. Ewaluacja modelu
    print("\n📊 Step 5: Model evaluation...")
    try:
        print("Making predictions on test set...")
        y_test_pred = model.predict(X_test)
        
        # Również zrób predykcje na zbiorze treningowym dla porównania
        print("Making predictions on training set...")
        y_train_pred = model.predict(X_train)
        
        # Oblicz metryki
        test_metrics = calculate_metrics(y_test, y_test_pred.flatten())
        train_metrics = calculate_metrics(y_train, y_train_pred.flatten())
        
        print(f"\n📈 Training Set Performance:")
        print(f"  R² Score: {train_metrics['r2']:.4f}")
        print(f"  MAE: {train_metrics['mae']:.6f}")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  RMSE: {train_metrics['rmse']:.6f}")
        
        print(f"\n📈 Test Set Performance:")
        print(f"  R² Score: {test_metrics['r2']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.6f}")
        print(f"  MSE: {test_metrics['mse']:.6f}")
        print(f"  RMSE: {test_metrics['rmse']:.6f}")
        
        # Sprawdź overfit
        r2_diff = train_metrics['r2'] - test_metrics['r2']
        mae_diff = test_metrics['mae'] - train_metrics['mae']
        
        print(f"\n🔍 Overfitting Analysis:")
        print(f"  R² difference (train - test): {r2_diff:.4f}")
        print(f"  MAE difference (test - train): {mae_diff:.6f}")
        
        if r2_diff > 0.1:
            print("⚠️ Warning: Possible overfitting detected (R² diff > 0.1)")
            print("  Consider:")
            print("  - Increasing dropout rate")
            print("  - Reducing model complexity")
            print("  - More training data")
        elif r2_diff > 0.05:
            print("ℹ️ Mild overfitting detected (R² diff > 0.05)")
            print("  Model performance is acceptable")
        else:
            print("✅ No significant overfitting detected")
        
        # Ocena jakości modelu
        quality_assessment = assess_model_quality(test_metrics['r2'], test_metrics['mae'])
        print(f"\n🎯 Model Quality Assessment:")
        print(f"  Overall Quality: {quality_assessment['overall']}")
        print(f"  R² Quality: {quality_assessment['r2_quality']}")
        print(f"  MAE Quality: {quality_assessment['mae_quality']}")
        
        # Zapisz metryki do pliku
        os.makedirs('results/metrics', exist_ok=True)
        with open('results/metrics/transformer_results.json', 'w') as f:
            json.dump({
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'quality_assessment': quality_assessment,
                'model_params': {
                    'sequence_length': SEQUENCE_LENGTH,
                    'prediction_horizon': PREDICTION_HORIZON,
                    'epochs_requested': EPOCHS,
                    'epochs_actual': len(history.history['loss']) if history else 0,
                    'batch_size': BATCH_SIZE,
                    'model_type': 'Stabilized Transformer',
                    'stabilization_features': [
                        'BatchNormalization',
                        'Reduced dropout rates',
                        'Gradient clipping',
                        'Conservative learning rate',
                        'Reduced model complexity'
                    ]
                }
            }, f, indent=2)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return

    # 6. Zapisz krzywe uczenia (BEZPOŚREDNIO PO TRENOWANIU)
    print("\n📈 Step 6: Saving training curves...")
    try:
        print("Creating and saving training curves...")
        model.save_training_curves(save_dir='results/plots/transformer')
        print("✅ Training curves saved successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not save training curves: {e}")

    # 7. Wizualizacje wyników trenowania (PO TRENOWANIU I EWALUACJI)
    if VISUALIZATION_AVAILABLE:
        print("\n📊 Step 7: Creating post-training visualizations...")
        try:
            print("Creating training results plots...")
            create_training_results_plots(
                history, y_test, y_test_pred.flatten(), test_metrics['r2'],
                save_path='results/plots/transformer/transformer_training_results.png'
            )
            
            # Predykcje przyszłości
            print("Creating future predictions plot...")
            last_sequence = X_test[-1]
            future_pred = predict_future(model, last_sequence, days_ahead=7)
            create_future_predictions_plot(
                y_test, future_pred, test_metrics['mae'], 7,
                save_path='results/plots/transformer/transformer_future_predictions.png'
            )
            
            # Single day prediction plot
            print("Creating single day prediction plot...")
            create_single_day_prediction_plot(
                df, model, test_metrics, SEQUENCE_LENGTH, 
                available_features, target_column
            )
            
            print("✅ All post-training visualizations created successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create some visualizations: {e}")
    else:
        print("\n⚠️ Step 7: Skipping visualizations (modules not available)")

    # 8. Predykcja na następny dzień z REALISTYCZNYMI wartościami
    print("\n🔮 Step 8: Next Day Price Prediction...")
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
        feature_data = last_60_days[available_features].values
        X_next = feature_data.reshape(1, SEQUENCE_LENGTH, len(available_features))
        
        print(f"Input sequence shape: {X_next.shape}")
        
        # Wykonaj predykcję z ograniczeniem do realistycznych wartości
        next_day_prediction_raw = model.predict(X_next)[0][0]
        
        # Ograniczenie zmian do realistycznych wartości dla Bitcoin (max 10% dziennie)
        last_price_normalized = df_sorted[target_column].iloc[-1]
        max_change = 0.10  # 10% maksymalna dzienna zmiana
        
        min_allowed = last_price_normalized * (1 - max_change)
        max_allowed = last_price_normalized * (1 + max_change)
        
        next_day_prediction_normalized = np.clip(next_day_prediction_raw, min_allowed, max_allowed)
        
        if next_day_prediction_raw != next_day_prediction_normalized:
            print(f"⚠️ Prediction adjusted for realism:")
            print(f"  Raw prediction: {next_day_prediction_raw:.6f}")
            print(f"  Adjusted prediction: {next_day_prediction_normalized:.6f}")
            print(f"  Constraint: ±{max_change*100:.0f}% daily change limit")
        
        print(f"\n🎯 NEXT DAY PRICE PREDICTION:")
        print("=" * 50)
        
        if scaler is not None:
            try:
                # Denormalizuj predykcję
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
                    
                    if abs(price_change_pct) <= 2:
                        print("📊 Trend: STABLE (Small change expected)")
                        print("💡 Trading insight: Consider range trading or wait for clearer signals")
                    elif price_change > 0:
                        print("📈 Trend: BULLISH (Price expected to rise)")
                        print("💡 Trading insight: Potential buying opportunity, but watch resistance levels")
                    else:
                        print("📉 Trend: BEARISH (Price expected to fall)")
                        print("💡 Trading insight: Potential selling opportunity, but watch support levels")
                        
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
        print(f"  Features Used: {len(available_features)}")
        print(f"  Model Type: Stabilized Transformer")
        print(f"  Realism Constraint: ±{max_change*100:.0f}% daily change limit")
        
        confidence_level = "High" if test_metrics['r2'] > 0.8 else "Medium" if test_metrics['r2'] > 0.6 else "Low"
        print(f"  Prediction Confidence: {confidence_level}")
        
        if test_metrics['r2'] > 0.8:
            print("  💡 Strategy: High confidence - consider this prediction reliable")
        elif test_metrics['r2'] > 0.6:
            print("  💡 Strategy: Medium confidence - use with caution")
        else:
            print("  💡 Strategy: Low confidence - consider additional analysis")
            
        print(f"  ⚠️ Disclaimer: This is a model prediction, not financial advice!")
        
    except Exception as e:
        print(f"❌ Error in next day prediction: {e}")

    # 9. EKSPLORACJA DANYCH NA KOŃCU (wszystkie wykresy Bitcoin)
    if VISUALIZATION_AVAILABLE:
        print("\n📊 Step 9: Creating data exploration plots...")
        try:
            print("Creating comprehensive data exploration plots...")
            create_data_exploration_plots(df, save_dir='results/plots/transformer')
            print("✅ Data exploration plots created successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create data exploration plots: {e}")
    else:
        print("\n⚠️ Step 9: Skipping data exploration plots (modules not available)")

    print(f"\n🎉 Stabilized Transformer model training and evaluation completed successfully!")
    print(f"📁 Results saved to:")
    print(f"  - Model: saved_models/bitcoin_transformer_model_stabilized.h5")
    print(f"  - Architecture: saved_models/transformer_architecture.json")
    print(f"  - Metrics: results/metrics/transformer_results.json")
    print(f"  - Plots: results/plots/transformer/")
    print(f"📊 Generated plots:")
    print(f"  - training_curves_combined.png (stabilized training curves)")
    print(f"  - transformer_training_results.png (comprehensive results)")
    print(f"  - transformer_future_predictions.png (7-day forecast)")
    print(f"  - single_day_prediction.png (next day prediction)")
    print(f"  - bitcoin_price_over_time.png (price history)")
    print(f"  - correlation_matrix.png (feature correlations)")
    print(f"  - price_distribution.png (price histogram)")
    print(f"  - moving_averages.png (technical indicators)")
    print(f"  - trading_volume.png (volume analysis)")

if __name__ == "__main__":
    main()