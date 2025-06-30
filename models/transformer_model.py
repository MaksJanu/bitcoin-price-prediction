import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Add, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import json

class BitcoinTransformerModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def create_transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Multi-head self-attention z batch normalization
        attention_output = MultiHeadAttention(
            key_dim=head_size, 
            num_heads=num_heads, 
            dropout=dropout
        )(inputs, inputs)
        
        # Gradient clipping przez mniejszy dropout
        attention_output = Dropout(dropout * 0.5)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
        
        # Residual connection
        attention_output = Add()([attention_output, inputs])
        
        # Feed Forward Network z batch normalization
        ffn_output = Dense(ff_dim, activation="relu")(attention_output)
        ffn_output = BatchNormalization()(ffn_output)  # Dodana batch normalization
        ffn_output = Dropout(dropout * 0.7)(ffn_output)  # Zmniejszony dropout
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        ffn_output = Dropout(dropout * 0.5)(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
        
        # Residual connection
        return Add()([ffn_output, attention_output])
    
    def create_model(self, head_size=128, num_heads=4, ff_dim=2, num_transformer_blocks=3,
                     mlp_units=[64, 32], dropout=0.2, mlp_dropout=0.3, learning_rate=0.0005):
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Dodaj początkową batch normalization
        x = BatchNormalization()(x)
        
        # Stack transformer blocks (mniej bloków dla stabilności)
        for i in range(num_transformer_blocks):
            x = self.create_transformer_block(
                x, head_size, num_heads, ff_dim * self.input_shape[-1], dropout
            )
        
        # Global average pooling
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        
        # Batch normalization przed MLP
        x = BatchNormalization()(x)
        
        # MLP layers z batch normalization
        for dim in mlp_units:
            x = Dense(dim, activation="relu", kernel_regularizer=l2(0.0005))(x)  # Zmniejszona regularyzacja
            x = BatchNormalization()(x)
            x = Dropout(mlp_dropout)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        self.model = Model(inputs, outputs)
        
        # Kompilacja modelu z mniejszym learning rate
        optimizer = Adam(learning_rate=learning_rate, clipnorm=0.5)  # Zmniejszony gradient clipping
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[MeanAbsoluteError()]
        )
        
        print("✅ Stabilny model Transformer został utworzony pomyślnie!")
        return self.model
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=100, 
              batch_size=32, patience=20, min_lr=1e-7):
        if self.model is None:
            raise ValueError("Model nie został utworzony. Użyj create_model() najpierw.")
        
        # Stwórz katalog dla checkpointów
        os.makedirs('saved_models/checkpoints', exist_ok=True)
        
        # Bardziej konserwatywne callbacki
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min',
                min_delta=1e-6  # Minimalna zmiana
            ),
            ModelCheckpoint(
                filepath='saved_models/checkpoints/best_transformer_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,  # Mniej agresywne zmniejszanie
                patience=patience//4,
                min_lr=min_lr,
                verbose=1,
                mode='min'
            )
        ]
        
        print(f"🏃‍♂️ Rozpoczynam trenowanie stabilnego modelu Transformer...")
        print(f"  Validation split: {validation_split}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Stabilizacja: Batch Normalization, zmniejszone dropout, gradient clipping")
        
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
        if self.model is None:
            raise ValueError("Model nie został utworzony lub nie został wytrenowany.")
        return self.model.predict(X, verbose=0)
    
    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        print("\n" + "="*60)
        print("📋 ARCHITEKTURA STABILNEGO MODELU TRANSFORMER")
        print("="*60)
        
        self.model.summary()
        
        # Dodatkowe informacje
        total_params = self.model.count_params()
        
        print(f"\n📊 Statystyki modelu:")
        print(f"  Całkowita liczba parametrów: {total_params:,}")
        print(f"  Kształt wejścia: {self.input_shape}")
        print(f"  Liczba warstw: {len(self.model.layers)}")
        print(f"  Stabilizacje: BatchNormalization, zmniejszone dropout, gradient clipping")
        
        return total_params
    
    def save_model_architecture(self, filepath='saved_models/transformer_architecture.json'):
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
                'metrics': getattr(self.model, 'metrics_names', []),
                'model_type': 'Stabilized Transformer',
                'stabilization_features': [
                    'BatchNormalization layers',
                    'Reduced dropout rates',
                    'Gradient clipping',
                    'Conservative learning rate',
                    'Reduced model complexity'
                ]
            }
            
            # Szczegóły warstw
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
                if hasattr(layer, 'num_heads'):
                    layer_info['num_heads'] = int(layer.num_heads)
                if hasattr(layer, 'key_dim'):
                    layer_info['key_dim'] = int(layer.key_dim)
                    
                architecture_info['layers'].append(layer_info)
            
            # Zapisz do pliku
            with open(filepath, 'w') as f:
                json.dump(architecture_info, f, indent=2, default=str)
            
            print(f"✅ Architektura stabilnego modelu zapisana do: {filepath}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save full architecture details: {e}")
            # Zapisz podstawowe informacje
            basic_info = {
                'input_shape': self.input_shape,
                'total_layers': len(self.model.layers),
                'total_params': self.model.count_params(),
                'model_type': 'Stabilized Transformer Sequential',
                'error': str(e)
            }
            
            with open(filepath, 'w') as f:
                json.dump(basic_info, f, indent=2, default=str)
            
            print(f"✅ Podstawowe informacje o architekturze zapisane do: {filepath}")
    
    def save_model(self, filepath='saved_models/bitcoin_transformer_model.h5'):
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✅ Stabilny model zapisany do: {filepath}")
    
    def save_training_curves(self, save_dir='results/plots/transformer'):
        if self.history is None:
            print("Brak historii trenowania.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Sprawdź dostępne metryki w historii
        available_metrics = list(self.history.history.keys())
        print(f"Dostępne metryki w historii: {available_metrics}")
        
        # Znajdź nazwę MAE metryki
        mae_key = None
        val_mae_key = None
        
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
        ax1.set_title('Stabilized Transformer Loss During Training', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        if mae_key and val_mae_key:
            ax2.plot(self.history.history[mae_key], label='Training MAE', linewidth=2, color='green')
            ax2.plot(self.history.history[val_mae_key], label='Validation MAE', linewidth=2, color='red')
            ax2.set_title('Stabilized Transformer MAE During Training', fontsize=14, fontweight='bold')
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
            ax2.set_title('Transformer MAE (Not Available)', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            print("⚠️ MAE metryki nie zostały znalezione")
        
        plt.tight_layout()
        combined_path = os.path.join(save_dir, 'training_curves_combined.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Stabilne krzywe uczenia zapisane do: {combined_path}")
        
        return combined_path