import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns

class BitcoinNaiveBayesModel:
    def __init__(self, input_shape, classification_type='direction'):
        """
        Inicjalizuje model Naive Bayes dla przewidywania Bitcoina
        
        Args:
            input_shape (tuple): Kształt danych wejściowych (timesteps, features)
            classification_type (str): 'direction' (wzrost/spadek) lub 'range' (zakresy cenowe)
        """
        self.input_shape = input_shape
        self.classification_type = classification_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
        self.training_history = {}
        self.prepared_features_for_correlation = None  # Dodane do przechowywania cech
        
    def _safe_division(self, numerator, denominator, default_value=0.0):
        """Bezpieczne dzielenie z obsługą dzielenia przez zero"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result = np.where(np.isfinite(result), result, default_value)
            return result
    
    def _clean_features(self, features):
        """Czyści cechy z wartości problematycznych"""
        # Zamień inf i -inf na NaN
        features = np.where(np.isinf(features), np.nan, features)
        
        # Zamień NaN na 0
        features = np.where(np.isnan(features), 0.0, features)
        
        # Ogranicz wartości do rozsądnego zakresu
        features = np.clip(features, -1e10, 1e10)
        
        return features
        
    def _prepare_features(self, X_sequences, store_for_correlation=False):
        """Przekształca sekwencje 3D na cechy 2D dla Naive Bayes"""
        # Spłaszcz sekwencje do cech statystycznych
        n_samples, timesteps, n_features = X_sequences.shape
        
        features = []
        feature_names = []
        
        # Nazwy oryginalnych cech (zakładając standardowe kolumny)
        original_feature_names = ['Open', 'High', 'Low', 'Volume', 'MA_7', 'MA_30', 'RSI'][:n_features]
        
        for sample in X_sequences:
            sample_features = []
            
            # Czyść dane wejściowe
            sample = self._clean_features(sample)
            
            # Cechy statystyczne dla każdej kolumny
            for feature_idx in range(n_features):
                feature_data = sample[:, feature_idx]
                
                # Sprawdź czy dane są poprawne
                if len(feature_data) == 0 or np.all(feature_data == 0):
                    # Jeśli wszystkie wartości to 0, użyj wartości domyślnych
                    sample_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    # Statystyki podstawowe z bezpiecznym obliczaniem
                    mean_val = np.mean(feature_data) if len(feature_data) > 0 else 0.0
                    std_val = np.std(feature_data) if len(feature_data) > 0 else 0.0
                    min_val = np.min(feature_data) if len(feature_data) > 0 else 0.0
                    max_val = np.max(feature_data) if len(feature_data) > 0 else 0.0
                    median_val = np.median(feature_data) if len(feature_data) > 0 else 0.0
                    
                    sample_features.extend([mean_val, std_val, min_val, max_val, median_val])
                
                if len(feature_names) < n_features * 5:
                    # Użyj czytelnych nazw cech
                    if feature_idx < len(original_feature_names):
                        base_name = original_feature_names[feature_idx]
                    else:
                        base_name = f"Feature_{feature_idx}"
                    
                    feature_names.extend([
                        f"{base_name}_mean",
                        f"{base_name}_std", 
                        f"{base_name}_min",
                        f"{base_name}_max",
                        f"{base_name}_median"
                    ])
            
            # Dodatkowe cechy sekwencyjne z bezpiecznym obliczaniem
            price_column = sample[:, -1]  # Ostatnia kolumna to cena
            
            if len(price_column) > 1:
                price_diffs = np.diff(price_column)
                price_change_mean = np.mean(price_diffs) if len(price_diffs) > 0 else 0.0
                price_change_std = np.std(price_diffs) if len(price_diffs) > 0 else 0.0
                
                # Bezpieczne obliczanie procentowej zmiany
                first_price = price_column[0]
                last_price = price_column[-1]
                
                if first_price != 0:
                    total_change_pct = (last_price - first_price) / first_price
                else:
                    total_change_pct = 0.0
            else:
                price_change_mean = 0.0
                price_change_std = 0.0
                total_change_pct = 0.0
            
            sample_features.extend([
                price_change_mean,
                price_change_std,
                total_change_pct
            ])
            
            if len(feature_names) == n_features * 5:
                feature_names.extend([
                    "Price_change_mean",
                    "Price_change_std", 
                    "Total_price_change_pct"
                ])
            
            features.append(sample_features)
        
        self.feature_names = feature_names
        features_array = np.array(features)
        
        # Finalne czyszczenie cech
        features_array = self._clean_features(features_array)
        
        # Zapisz cechy do analizy korelacji jeśli potrzeba
        if store_for_correlation:
            self.prepared_features_for_correlation = features_array
        
        return features_array
    
    def get_feature_correlation_matrix(self):
        """Zwraca macierz korelacji przetwo rzonych cech"""
        if self.prepared_features_for_correlation is None:
            print("Brak przygotowanych cech. Uruchom najpierw trenowanie.")
            return None
        
        # Stwórz DataFrame z przetworzonymi cechami
        df_features = pd.DataFrame(
            self.prepared_features_for_correlation, 
            columns=self.feature_names
        )
        
        return df_features.corr()
    
    def _create_labels(self, y_true, X_sequences):
        """Tworzy etykiety klasyfikacyjne z wartości rzeczywistych"""
        if self.classification_type == 'direction':
            # Przewiduj kierunek zmiany (wzrost/spadek)
            last_prices = X_sequences[:, -1, -1]  # ostatnia cena w sekwencji
            
            # Bezpieczne obliczanie zmian cen
            price_changes = self._safe_division(y_true - last_prices, last_prices, 0.0)
            
            # Binarna klasyfikacja: 0 = spadek, 1 = wzrost
            labels = (price_changes > 0).astype(int)
            self.class_names = ['Spadek', 'Wzrost']
            
        elif self.classification_type == 'range':
            # Przewiduj zakres zmiany cenowej
            last_prices = X_sequences[:, -1, -1]
            price_changes = self._safe_division(y_true - last_prices, last_prices, 0.0) * 100
            
            # Klasyfikacja wieloklasowa na podstawie procentowej zmiany
            labels = np.zeros(len(price_changes))
            labels[price_changes < -2] = 0  # Duży spadek
            labels[(price_changes >= -2) & (price_changes < -0.5)] = 1  # Mały spadek
            labels[(price_changes >= -0.5) & (price_changes <= 0.5)] = 2  # Stagnacja
            labels[(price_changes > 0.5) & (price_changes <= 2)] = 3  # Mały wzrost
            labels[price_changes > 2] = 4  # Duży wzrost
            
            self.class_names = ['Duży spadek (<-2%)', 'Mały spadek (-2% do -0.5%)', 
                              'Stagnacja (-0.5% do 0.5%)', 'Mały wzrost (0.5% do 2%)', 
                              'Duży wzrost (>2%)']
        
        return labels.astype(int)
    
    def create_model(self, var_smoothing=1e-9):
        """
        Tworzy model Gaussian Naive Bayes
        
        Args:
            var_smoothing (float): Wygładzanie wariancji
        """
        self.model = GaussianNB(var_smoothing=var_smoothing)
        print("✅ Model Naive Bayes został utworzony pomyślnie!")
        return self.model
    
    def train(self, X_train, y_train, validation_split=0.2):
        """
        Trenuje model Naive Bayes
        
        Args:
            X_train: Dane treningowe (sekwencje)
            y_train: Etykiety treningowe (ceny)
            validation_split: Udział danych walidacyjnych
        """
        if self.model is None:
            raise ValueError("Model nie został utworzony. Użyj create_model() najpierw.")
        
        print("🔧 Przygotowywanie cech...")
        # Zapisz cechy do analizy korelacji
        X_features = self._prepare_features(X_train, store_for_correlation=True)
        y_labels = self._create_labels(y_train, X_train)
        
        print(f"📊 Kształt cech: {X_features.shape}")
        print(f"📊 Sprawdzanie jakości danych:")
        print(f"  - Wartości inf: {np.isinf(X_features).sum()}")
        print(f"  - Wartości NaN: {np.isnan(X_features).sum()}")
        print(f"  - Zakres wartości: {np.min(X_features):.6f} do {np.max(X_features):.6f}")
        
        # Dodatkowe czyszczenie przed treningiem
        X_features = self._clean_features(X_features)
        
        # Podział na trenowanie i walidację
        split_idx = int(len(X_features) * (1 - validation_split))
        X_train_split = X_features[:split_idx]
        X_val_split = X_features[split_idx:]
        y_train_split = y_labels[:split_idx]
        y_val_split = y_labels[split_idx:]
        
        print(f"📊 Rozkład klas treningowych: {np.bincount(y_train_split)}")
        
        # Normalizacja cech z dodatkowymi zabezpieczeniami
        print("🔧 Normalizacja cech...")
        try:
            X_train_scaled = self.scaler.fit_transform(X_train_split)
            X_val_scaled = self.scaler.transform(X_val_split)
            
            # Sprawdź wyniki normalizacji
            X_train_scaled = self._clean_features(X_train_scaled)
            X_val_scaled = self._clean_features(X_val_scaled)
            
            print(f"📊 Po normalizacji - zakres: {np.min(X_train_scaled):.6f} do {np.max(X_train_scaled):.6f}")
            
        except Exception as e:
            print(f"⚠️ Problem z normalizacją: {e}")
            print("Używam danych bez normalizacji...")
            X_train_scaled = X_train_split
            X_val_scaled = X_val_split
        
        print("🏃‍♂️ Rozpoczynam trenowanie modelu...")
        
        try:
            # Trenowanie
            self.model.fit(X_train_scaled, y_train_split)
            
            # Ewaluacja na zbiorze walidacyjnym
            train_accuracy = self.model.score(X_train_scaled, y_train_split)
            val_accuracy = self.model.score(X_val_scaled, y_val_split)
            
            # Cross-validation z obsługą błędów
            try:
                cv_scores = cross_val_score(self.model, X_train_scaled, y_train_split, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                print(f"⚠️ Błąd podczas cross-validation: {e}")
                cv_scores = np.array([train_accuracy])
                cv_mean = train_accuracy
                cv_std = 0.0
            
            # Zapisz historię trenowania
            self.training_history = {
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'class_distribution': np.bincount(y_train_split).tolist()
            }
            
            print(f"✅ Trenowanie zakończone!")
            print(f"  Dokładność treningowa: {train_accuracy:.4f}")
            print(f"  Dokładność walidacyjna: {val_accuracy:.4f}")
            print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
        except Exception as e:
            print(f"❌ Błąd podczas trenowania: {e}")
            # Zapisz podstawową historię trenowania
            self.training_history = {
                'train_accuracy': 0.0,
                'val_accuracy': 0.0,
                'cv_scores': [0.0],
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'class_distribution': np.bincount(y_train_split).tolist(),
                'error': str(e)
            }
            raise
        
        return self.training_history
    
    def predict(self, X):
        """Wykonuje predykcję"""
        if self.model is None:
            raise ValueError("Model nie został utworzony lub nie został wytrenowany.")
        
        X_features = self._prepare_features(X)
        X_features = self._clean_features(X_features)
        
        try:
            X_scaled = self.scaler.transform(X_features)
            X_scaled = self._clean_features(X_scaled)
        except:
            X_scaled = X_features
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Wykonuje predykcję z prawdopodobieństwami"""
        if self.model is None:
            raise ValueError("Model nie został utworzony lub nie został wytrenowany.")
        
        X_features = self._prepare_features(X)
        X_features = self._clean_features(X_features)
        
        try:
            X_scaled = self.scaler.transform(X_features)
            X_scaled = self._clean_features(X_scaled)
        except:
            X_scaled = X_features
        
        return self.model.predict_proba(X_scaled)
    
    def get_model_summary(self):
        """Wyświetla podsumowanie modelu"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        print("\n" + "="*60)
        print("📋 ARCHITEKTURA MODELU NAIVE BAYES")
        print("="*60)
        
        print(f"Model type: Gaussian Naive Bayes")
        print(f"Classification type: {self.classification_type}")
        print(f"Number of features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
        print(f"Number of classes: {len(self.class_names) if self.class_names else 'Unknown'}")
        
        if self.class_names:
            print(f"Classes: {self.class_names}")
        
        if hasattr(self.model, 'class_count_'):
            print(f"Class counts: {self.model.class_count_}")
        
        print("="*60)
    
    def save_model_architecture(self, filepath='saved_models/naive_bayes_architecture.json'):
        """Zapisuje architekturę modelu do pliku JSON"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        architecture_info = {
            'model_type': 'Gaussian Naive Bayes',
            'classification_type': self.classification_type,
            'input_shape': self.input_shape,
            'number_of_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'number_of_classes': len(self.class_names) if self.class_names else None,
            'class_names': self.class_names,
            'var_smoothing': self.model.var_smoothing if hasattr(self.model, 'var_smoothing') else None
        }
        
        if hasattr(self.model, 'class_count_'):
            architecture_info['class_counts'] = self.model.class_count_.tolist()
        
        with open(filepath, 'w') as f:
            json.dump(architecture_info, f, indent=2, default=str)
        
        print(f"✅ Architektura modelu zapisana do: {filepath}")
    
    def save_model(self, filepath='saved_models/bitcoin_naive_bayes_model.joblib'):
        """Zapisuje cały model"""
        if self.model is None:
            raise ValueError("Model nie został utworzony.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'input_shape': self.input_shape,
            'classification_type': self.classification_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model zapisany do: {filepath}")
    
    def load_model(self, filepath='saved_models/bitcoin_naive_bayes_model.joblib'):
        """Ładuje zapisany model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.input_shape = model_data['input_shape']
        self.classification_type = model_data['classification_type']
        self.feature_names = model_data['feature_names']
        self.class_names = model_data['class_names']
        self.training_history = model_data['training_history']
        
        print(f"✅ Model załadowany z: {filepath}")
    
    def save_training_curves(self, save_dir='results/plots/naive_bayes'):
        """Zapisuje wykresy trenowania"""
        if not self.training_history:
            print("Brak historii trenowania.")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Wykres 1: Dokładność
        categories = ['Training', 'Validation', 'CV Mean']
        accuracies = [
            self.training_history['train_accuracy'],
            self.training_history['val_accuracy'],
            self.training_history['cv_mean']
        ]
        
        bars = ax1.bar(categories, accuracies, color=['blue', 'orange', 'green'], alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Dodaj wartości na słupkach
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Wykres 2: Cross-validation scores
        cv_scores = self.training_history['cv_scores']
        if len(cv_scores) > 1:
            ax2.boxplot(cv_scores)
            ax2.set_title('Cross-Validation Scores Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Accuracy')
            ax2.set_xticklabels(['CV Scores'])
        else:
            ax2.bar(['CV Score'], cv_scores)
            ax2.set_title('Cross-Validation Score', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Accuracy')
        
        # Wykres 3: Rozkład klas
        class_dist = self.training_history['class_distribution']
        if self.class_names and len(class_dist) == len(self.class_names):
            ax3.pie(class_dist, labels=self.class_names, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Class Distribution in Training Data', fontsize=14, fontweight='bold')
        else:
            ax3.bar(range(len(class_dist)), class_dist)
            ax3.set_title('Class Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Count')
        
        # Wykres 4: Feature importance (top 10)
        if hasattr(self.model, 'feature_log_prob_') and self.feature_names:
            try:
                # Oblicz "ważność" cech jako wariancję prawdopodobieństw między klasami
                feature_importance = np.var(self.model.feature_log_prob_, axis=0)
                
                # Zabezpieczenie przed wartościami problematycznymi
                feature_importance = self._clean_features(feature_importance.reshape(-1)).flatten()
                
                top_features_idx = np.argsort(feature_importance)[-10:]
                
                ax4.barh(range(10), feature_importance[top_features_idx])
                ax4.set_yticks(range(10))
                ax4.set_yticklabels([self.feature_names[i] for i in top_features_idx])
                ax4.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Feature Importance (Log Prob Variance)')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Feature importance\nerror: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=10)
                ax4.set_title('Feature Importance (Error)', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Feature Importance (N/A)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        combined_path = os.path.join(save_dir, 'naive_bayes_training_curves.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Wykresy trenowania zapisane do: {combined_path}")
        return combined_path