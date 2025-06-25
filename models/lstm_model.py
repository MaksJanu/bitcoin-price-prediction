import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError

class BitcoinLSTMModel:
    def __init__(self, input_shape):
        """
        Inicjalizuje model LSTM dla predykcji cen Bitcoin
        
        Args:
            input_shape (tuple): Kształt danych wejściowych (timesteps, features)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_model(self, lstm_units=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
        """
        Tworzy zoptymalizowaną architekturę modelu LSTM
        
        Args:
            lstm_units (list): Lista liczb neuronów w warstwach LSTM
            dropout_rate (float): Współczynnik dropout
            learning_rate (float): Learning rate dla optimizera
        """
        self.model = Sequential()
        
        # Pierwsza warstwa LSTM z BatchNormalization
        self.model.add(LSTM(
            lstm_units[0], 
            return_sequences=True, 
            input_shape=self.input_shape,
            kernel_regularizer=l2(0.001)
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Druga warstwa LSTM
        self.model.add(LSTM(
            lstm_units[1], 
            return_sequences=True,
            kernel_regularizer=l2(0.001)
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Trzecia warstwa LSTM (bez return_sequences)
        self.model.add(LSTM(
            lstm_units[2],
            return_sequences=False,
            kernel_regularizer=l2(0.001)
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Warstwa gęsta wyjściowa
        self.model.add(Dense(1))
        
        # Kompilacja modelu z zoptymalizowanymi parametrami
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[MeanAbsoluteError()]
        )
        
        print("✅ Model LSTM został utworzony pomyślnie!")
        return self.model
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=100, 
              batch_size=16, patience=30, min_lr=1e-6):
        """
        Trenuje model z zoptymalizowanymi callbackami
        
        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            validation_split: Udział danych walidacyjnych
            epochs: Maksymalna liczba epok
            batch_size: Rozmiar batcha
            patience: Patience dla EarlyStopping
            min_lr: Minimalny learning rate
        """
        if self.model is None:
            raise ValueError("Model nie został utworzony. Użyj create_model() najpierw.")
        
        # Stwórz katalog dla checkpointów
        os.makedirs('saved_models/checkpoints', exist_ok=True)
        
        # Zoptymalizowane callbacki
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                filepath='saved_models/checkpoints/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//3,
                min_lr=min_lr,
                verbose=1,
                mode='min'
            )
        ]
        
        print(f"🏃‍♂️ Rozpoczynam trenowanie modelu...")
        print(f"  Validation split: {validation_split}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Callbacks: EarlyStopping (patience={patience}), ModelCheckpoint, ReduceLROnPlateau")
        
        # Trenowanie
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("✅ Trenowanie zakończone!")
        return self.history
    
    def predict(self, X):
        """Wykonuje predykcję"""
        if self.model is None:
            raise ValueError("Model nie został utworzony lub nie został wytrenowany.")
        return self.model.predict(X, verbose=0)
    
    def get_model_summary(self):
        """Wyświetla podsumowanie modelu"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        print("\n" + "="*60)
        print("📋 ARCHITEKTURA MODELU LSTM")
        print("="*60)
        
        self.model.summary()
        
        # Dodatkowe informacje
        total_params = self.model.count_params()
        trainable_params = sum([np.prod(layer.get_weights()[0].shape) + np.prod(layer.get_weights()[1].shape) 
                               for layer in self.model.layers if layer.get_weights()])
        
        print(f"\n📊 Statystyki modelu:")
        print(f"  Całkowita liczba parametrów: {total_params:,}")
        print(f"  Kształt wejścia: {self.input_shape}")
        print(f"  Liczba warstw: {len(self.model.layers)}")
        
        return total_params
    
    def save_model_architecture(self, filepath='saved_models/model_architecture.json'):
        """Zapisuje architekturę modelu do pliku JSON - POPRAWIONA WERSJA"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # Najpierw zbuduj model, aby zainicjalizować warstwy
            dummy_input = np.zeros((1, self.input_shape[0], self.input_shape[1]))
            _ = self.model.predict(dummy_input, verbose=0)
            
            # Informacje o architekturze
            architecture_info = {
                'input_shape': self.input_shape,
                'layers': [],
                'total_params': self.model.count_params(),
                'optimizer': self.model.optimizer.__class__.__name__ if self.model.optimizer else 'Unknown',
                'loss': self.model.loss if hasattr(self.model, 'loss') else 'Unknown',
                'metrics': getattr(self.model, 'metrics_names', [])
            }
            
            # Szczegóły warstw - BEZPIECZNE podejście
            for i, layer in enumerate(self.model.layers):
                layer_info = {
                    'layer_number': i,
                    'name': layer.name,
                    'type': layer.__class__.__name__
                }
                
                # Bezpieczne pobieranie output_shape
                try:
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                        layer_info['output_shape'] = str(output_shape) if output_shape else 'Unknown'
                    else:
                        layer_info['output_shape'] = 'Unknown'
                except Exception:
                    layer_info['output_shape'] = 'Unknown'
                
                # Dodatkowe informacje dla konkretnych typów warstw
                if hasattr(layer, 'units'):
                    layer_info['units'] = int(layer.units)
                if hasattr(layer, 'rate'):
                    layer_info['dropout_rate'] = float(layer.rate)
                if hasattr(layer, 'return_sequences'):
                    layer_info['return_sequences'] = bool(layer.return_sequences)
                if hasattr(layer, 'activation'):
                    activation = layer.activation
                    if hasattr(activation, '__name__'):
                        layer_info['activation'] = activation.__name__
                    else:
                        layer_info['activation'] = str(activation)
                    
                architecture_info['layers'].append(layer_info)
            
            # Zapisz do pliku
            with open(filepath, 'w') as f:
                json.dump(architecture_info, f, indent=2, default=str)
            
            print(f"✅ Architektura modelu zapisana do: {filepath}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save full architecture details: {e}")
            # Zapisz podstawowe informacje
            basic_info = {
                'input_shape': self.input_shape,
                'total_layers': len(self.model.layers),
                'total_params': self.model.count_params(),
                'model_type': 'LSTM Sequential',
                'error': str(e)
            }
            
            with open(filepath, 'w') as f:
                json.dump(basic_info, f, indent=2, default=str)
            
            print(f"✅ Podstawowe informacje o architekturze zapisane do: {filepath}")
    
    def save_model(self, filepath='saved_models/bitcoin_lstm_model.h5'):
        """Zapisuje cały model"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✅ Model zapisany do: {filepath}")
    
    def plot_training_history(self, save_path='results/plots/training_history.png'):
        """Tworzy wykres historii trenowania"""
        if self.history is None:
            print("Brak historii trenowania.")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in self.history.history:
            ax2.plot(self.history.history['mae'], label='Training MAE', linewidth=2)
            ax2.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
            ax2.set_title('Model MAE', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Wykres historii trenowania zapisany do: {save_path}")
    
    def save_training_curves(self, save_dir='results/plots/'):
        """Zapisuje krzywe uczenia - TYLKO JEDEN POŁĄCZONY WYKRES"""
        if self.history is None:
            print("Brak historii trenowania.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Sprawdź dostępne metryki w historii
        available_metrics = list(self.history.history.keys())
        print(f"Dostępne metryki w historii: {available_metrics}")
        
        # Znajdź nazwę MAE metryki - wszystkie możliwe warianty
        mae_key = None
        val_mae_key = None
        
        # Możliwe nazwy MAE w Keras
        possible_mae_names = ['mae', 'mean_absolute_error', 'MAE', 'MeanAbsoluteError']
        
        for possible_name in possible_mae_names:
            if possible_name in available_metrics:
                mae_key = possible_name
                break
        
        for possible_name in possible_mae_names:
            val_name = f'val_{possible_name}'
            if val_name in available_metrics:
                val_mae_key = val_name
                break
        
        print(f"Znalezione klucze MAE: train='{mae_key}', validation='{val_mae_key}'")
        
        # Jeden wykres z obiema krzywymi uczenia
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
        ax1.set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE (używaj znalezionej nazwy metryki)
        if mae_key and val_mae_key:
            ax2.plot(self.history.history[mae_key], label='Training MAE', linewidth=2, color='green')
            ax2.plot(self.history.history[val_mae_key], label='Validation MAE', linewidth=2, color='red')
            ax2.set_title('Model MAE During Training', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('MAE', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            print("✅ MAE metryki znalezione i wyświetlone")
        else:
            # Alternatywnie pokaż wszystkie dostępne metryki jako tekst
            available_text = '\n'.join([f"• {metric}" for metric in available_metrics])
            ax2.text(0.1, 0.5, f'MAE not found in history\n\nAvailable metrics:\n{available_text}', 
                    ha='left', va='center', transform=ax2.transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            ax2.set_title('Model MAE (Not Available)', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            print("⚠️ MAE metryki nie zostały znalezione")
        
        plt.tight_layout()
        combined_path = os.path.join(save_dir, 'training_curves_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Połączone krzywe uczenia zapisane do: {combined_path}")
        
        return combined_path