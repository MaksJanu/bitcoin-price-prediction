import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import os


class BitcoinLSTMModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        print(f"Model initialized with input shape: {input_shape}")
        
    def create_model(self):
        """Tworzy model LSTM do predykcji ceny bitcoina"""
        print("Creating LSTM model...")
        
        self.model = Sequential([
            # Pierwsza warstwa LSTM z return_sequences=True
            LSTM(64, return_sequences=True, input_shape=self.input_shape, name='LSTM_1'),
            Dropout(0.2, name='Dropout_1'),
            
            # Druga warstwa LSTM z return_sequences=True
            LSTM(64, return_sequences=True, name='LSTM_2'),
            Dropout(0.2, name='Dropout_2'),
            
            # Trzecia warstwa LSTM bez return_sequences
            LSTM(32, return_sequences=False, name='LSTM_3'),
            Dropout(0.2, name='Dropout_3'),
            
            # Warstwy Dense
            Dense(25, activation='relu', name='Dense_1'),
            Dropout(0.1, name='Dropout_4'),
            Dense(1, name='Output')
        ])
        
        # Kompiluj model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Model created and compiled successfully!")
        return self.model
    
    def train(self, X_train, y_train, validation_split=0.2, epochs=200, batch_size=32):
        """Trenuje model LSTM"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        print(f"Starting training with:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation split: {validation_split}")
        
        # Ensure directory exists
        os.makedirs('saved_models', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'saved_models/bitcoin_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        print("Starting model training...")
        
        # Trenuj model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        print("Training completed!")
        return self.history
    
    def predict(self, X):
        """Wykonuje predykcję"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        print(f"Making predictions for {X.shape[0]} samples...")
        return self.model.predict(X, verbose=0)
    
    def save_model_architecture(self, filepath='results/plots/lstm_model_architecture.png'):
        """Zapisuje diagram architektury modelu"""
        if self.model is None:
            raise ValueError("Model not created yet.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            plot_model(self.model, to_file=filepath, 
                       show_shapes=True, show_layer_names=True, dpi=300)
            print(f"Model architecture saved to: {filepath}")
        except Exception as e:
            print(f"Could not save model architecture: {e}")
    
    def get_model_summary(self):
        """Zwraca podsumowanie modelu"""
        if self.model is None:
            raise ValueError("Model not created yet.")
        return self.model.summary()
    
    def load_model(self, filepath='saved_models/bitcoin_lstm_model.h5'):
        """Ładuje zapisany model"""
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from: {filepath}")
            return self.model
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")