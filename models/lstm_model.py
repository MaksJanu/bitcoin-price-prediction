import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import os
import json

class BitcoinLSTMModel:
    def __init__(self, input_shape):
        """
        Inicjalizuje model LSTM dla predykcji cen Bitcoin
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
        
        # Trzecia warstwa LSTM
        self.model.add(LSTM(
            lstm_units[2], 
            return_sequences=False,
            kernel_regularizer=l2(0.001)
        ))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Warstwa Dense z regularyzacją
        self.model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(dropout_rate/2))
        
        # Warstwa wyjściowa
        self.model.add(Dense(1))
        
        # Kompilacja z optymalizowanymi parametrami
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='huber',  # Huber loss jest bardziej odporny na outliers
            metrics=['mae', 'mse']
        )
        
        print(f"✅ Model created with architecture:")
        print(f"  LSTM layers: {lstm_units}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Loss function: Huber")
        
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
                min_delta=1e-6  # Minimalna zmiana która liczy się jako poprawa
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,  # Mniejsza patience dla LR reduction
                min_lr=min_lr,
                verbose=1,
                cooldown=5  # Czas oczekiwania po redukcji LR
            ),
            ModelCheckpoint(
                'saved_models/checkpoints/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        print(f"🚀 Starting model training:")
        print(f"  Max epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation split: {validation_split}")
        print(f"  Early stopping patience: {patience}")
        print(f"  LR reduction patience: {patience//2}")
        print(f"  Min learning rate: {min_lr}")
        
        # Trenowanie
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True  # Mieszanie danych w każdej epoce
        )
        
        # Załaduj najlepszy model
        self.model = load_model('saved_models/checkpoints/best_model.h5')
        print("✅ Best model loaded from checkpoint")
        
        return self.history
    
    def predict(self, X):
        """Przewiduje wartości dla danych wejściowych"""
        if self.model is None:
            raise ValueError("Model nie został wytrenowany.")
        return self.model.predict(X, verbose=0)
    
    def get_model_summary(self):
        """Wyświetla podsumowanie modelu"""
        if self.model is None:
            print("Model nie został utworzony.")
        else:
            self.model.summary()
    
    def save_model(self, filepath='saved_models/bitcoin_lstm_model.h5'):
        """Zapisuje model do pliku"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model zapisany do: {filepath}")
    
    def load_model(self, filepath='saved_models/bitcoin_lstm_model.h5'):
        """Ładuje model z pliku"""
        self.model = load_model(filepath)
        print(f"Model załadowany z: {filepath}")
    
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