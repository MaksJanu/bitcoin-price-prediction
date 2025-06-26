import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_model_results():
    """Ładuje wyniki obu modeli z DOKŁADNĄ strukturą plików"""
    results = {}
    
    # LSTM results - POPRAWIONA STRUKTURA
    print("🔍 Loading LSTM results...")
    try:
        with open('results/metrics/training_results.json', 'r') as f:
            lstm_results = json.load(f)
            print(f"✅ LSTM file loaded")
            print(f"Raw LSTM structure: {list(lstm_results.keys())}")
            
            # LSTM ma strukturę: {"metrics": {"test": {"mse": ..., "mae": ..., "r2": ...}}}
            if 'metrics' in lstm_results and 'test' in lstm_results['metrics']:
                test_data = lstm_results['metrics']['test']
                print(f"Found 'metrics.test' key with: {list(test_data.keys())}")
                
                # Sprawdź czy wszystkie potrzebne metryki są dostępne
                required_metrics = ['mse', 'mae', 'r2']
                missing_metrics = [m for m in required_metrics if m not in test_data]
                
                if missing_metrics:
                    print(f"⚠️ Missing metrics in LSTM: {missing_metrics}")
                
                # Dodaj brakujące RMSE jeśli nie ma
                if 'rmse' not in test_data and 'mse' in test_data:
                    test_data['rmse'] = (test_data['mse'] ** 0.5)
                    print("✅ Calculated RMSE from MSE")
                
                results['LSTM'] = {
                    'mse': test_data.get('mse', 0),
                    'mae': test_data.get('mae', 0),
                    'rmse': test_data.get('rmse', (test_data.get('mse', 0) ** 0.5)),
                    'r2': test_data.get('r2', 0)
                }
                print("✅ LSTM results loaded from 'metrics.test' structure")
                print(f"Final LSTM metrics: {results['LSTM']}")
                
            else:
                print("❌ Expected structure 'metrics.test' not found in LSTM file")
                print("Available keys:", list(lstm_results.keys()))
                if 'metrics' in lstm_results:
                    print("Available in metrics:", list(lstm_results['metrics'].keys()))
                results['LSTM'] = None
                
    except FileNotFoundError:
        print("❌ LSTM results file not found: 'results/metrics/training_results.json'")
        results['LSTM'] = None
    except Exception as e:
        print(f"❌ Error loading LSTM results: {e}")
        import traceback
        traceback.print_exc()
        results['LSTM'] = None
    
    # Transformer results - BEZ ZMIAN (działa poprawnie)
    print("\n🔍 Loading Transformer results...")
    try:
        with open('results/metrics/transformer_results.json', 'r') as f:
            transformer_results = json.load(f)
            print(f"✅ Transformer file loaded")
            print(f"Raw Transformer structure: {list(transformer_results.keys())}")
            
            # Transformer ma strukturę: {"test_metrics": {"mse": ..., "mae": ..., etc}}
            if 'test_metrics' in transformer_results:
                results['Transformer'] = transformer_results['test_metrics']
                print("✅ Transformer results loaded from 'test_metrics' key")
                print(f"Final Transformer metrics: {results['Transformer']}")
            else:
                print("❌ Expected structure 'test_metrics' not found in Transformer file")
                print("Available keys:", list(transformer_results.keys()))
                results['Transformer'] = None
                
    except FileNotFoundError:
        print("❌ Transformer results file not found: 'results/metrics/transformer_results.json'")
        results['Transformer'] = None
    except Exception as e:
        print(f"❌ Error loading Transformer results: {e}")
        results['Transformer'] = None
    
    return results

def create_comparison_plots(results):
    """Tworzy wykresy porównawcze - PROSTE ZESTAWIENIE BEZ WYRÓŻNIANIA"""
    os.makedirs('results/plots/comparison', exist_ok=True)
    
    # Sprawdź dostępność danych
    if not results['LSTM'] and not results['Transformer']:
        print("❌ Cannot create comparison plots - both models missing results")
        return
    
    if not results['LSTM']:
        print("⚠️ Cannot create full comparison - LSTM results missing")
        return
        
    if not results['Transformer']:
        print("⚠️ Cannot create full comparison - Transformer results missing")
        return
    
    print("✅ Both models have results - creating comparison plots")
    
    # Metryki do porównania
    metrics = ['r2', 'mae', 'mse', 'rmse']
    models = ['LSTM', 'Transformer']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c']  # Niebieski dla LSTM, czerwony dla Transformer
    
    for i, metric in enumerate(metrics):
        try:
            values = []
            labels = []
            colors_used = []
            
            for j, model in enumerate(models):
                if results[model] and metric in results[model] and results[model][metric] != 0:
                    values.append(results[model][metric])
                    labels.append(model)
                    colors_used.append(colors[j])
                    print(f"Added {model} {metric}: {results[model][metric]}")
            
            if not values:
                axes[i].text(0.5, 0.5, f'No valid data for {metric.upper()}', 
                            ha='center', va='center', transform=axes[i].transAxes,
                            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[i].set_title(f'{metric.upper()} - No Data', fontsize=14)
                continue
            
            print(f"Plotting {metric}: {labels} = {values}")
            
            # PROSTY WYKRES BEZ WYRÓŻNIANIA
            bars = axes[i].bar(labels, values, color=colors_used, alpha=0.7, edgecolor='black', linewidth=1)
            axes[i].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Dodaj wartości na słupkach
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                            f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # USUŃ WYRÓŻNIANIE ZWYCIĘZCY - po prostu pokaż oba słupki równo
            
        except Exception as e:
            print(f"❌ Error plotting {metric}: {e}")
            axes[i].text(0.5, 0.5, f'Error displaying {metric}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{metric.upper()} - Error', fontsize=14)
    
    plt.tight_layout()
    save_path = 'results/plots/comparison/models_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✅ Model comparison saved to: {save_path}")

def print_comparison_table(results):
    """Wyświetla PROSTE ZESTAWIENIE bez wyróżniania zwycięzców"""
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON TABLE")
    print("="*70)
    
    # Sprawdź dostępność danych
    available_models = [model for model, data in results.items() if data]
    if not available_models:
        print("❌ No model results available for comparison")
        return
    
    print(f"Available models: {available_models}")
    
    # Pokaż szczegóły każdego modelu
    for model, data in results.items():
        if data:
            print(f"\n{model} metrics:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n{model}: No data available")
    
    # Jeśli mamy oba modele, pokaż PROSTE PORÓWNANIE
    if results['LSTM'] and results['Transformer']:
        lstm_metrics = results['LSTM']
        transformer_metrics = results['Transformer']
        
        # Stwórz DataFrame z wynikami
        comparison_data = {}
        for metric in ['r2', 'mae', 'mse', 'rmse']:
            comparison_data[metric] = {
                'LSTM': lstm_metrics.get(metric, 'N/A'),
                'Transformer': transformer_metrics.get(metric, 'N/A')
            }
        
        df = pd.DataFrame(comparison_data).T
        print("\n📊 DETAILED METRICS COMPARISON:")
        print(df.to_string(col_space=15))
        
        # PROSTE PODSUMOWANIE BEZ WYRÓŻNIANIA ZWYCIĘZCÓW
        print("\n" + "="*70)
        print("📋 SUMMARY")
        print("="*70)
        
        # Pokaż różnice między modelami
        print(f"\n📈 Model Performance Summary:")
        
        for metric in ['r2', 'mae', 'mse', 'rmse']:
            lstm_val = lstm_metrics.get(metric)
            transformer_val = transformer_metrics.get(metric)
            
            if lstm_val is not None and transformer_val is not None and lstm_val != 0 and transformer_val != 0:
                diff = abs(lstm_val - transformer_val)
                if metric == 'r2':
                    pct_diff = (diff / max(lstm_val, transformer_val)) * 100
                    print(f"  {metric.upper()}: LSTM={lstm_val:.6f}, Transformer={transformer_val:.6f}")
                    print(f"    Difference: {diff:.6f} ({pct_diff:.2f}%)")
                else:
                    pct_diff = (diff / min(lstm_val, transformer_val)) * 100 if min(lstm_val, transformer_val) > 0 else 0
                    print(f"  {metric.upper()}: LSTM={lstm_val:.6f}, Transformer={transformer_val:.6f}")
                    print(f"    Difference: {diff:.6f} ({pct_diff:.2f}%)")
            else:
                print(f"  {metric.upper()}: Cannot compare (missing or zero data)")
        
        # Ogólne podsumowanie
        print(f"\n💡 General Assessment:")
        best_r2 = max(lstm_metrics.get('r2', 0), transformer_metrics.get('r2', 0))
        best_mae = min(lstm_metrics.get('mae', float('inf')), transformer_metrics.get('mae', float('inf')))
        
        if best_r2 >= 0.8:
            print(f"   ✅ Both models show excellent accuracy (best R² = {best_r2:.4f})")
        elif best_r2 >= 0.6:
            print(f"   ✅ Both models show good accuracy (best R² = {best_r2:.4f})")
        else:
            print(f"   ⚠️ Models need improvement (best R² = {best_r2:.4f})")
        
        if best_mae <= 0.05:
            print(f"   ✅ Both models show excellent precision (best MAE = {best_mae:.6f})")
        elif best_mae <= 0.1:
            print(f"   ✅ Both models show good precision (best MAE = {best_mae:.6f})")
        else:
            print(f"   ⚠️ Models need precision improvements (best MAE = {best_mae:.6f})")
        
    else:
        print("\n⚠️ Cannot perform detailed comparison - missing data from one or both models")

def main():
    print("🔍 Bitcoin Prediction Models Comparison - SIMPLE OVERVIEW")
    print("="*70)
    
    # Sprawdź czy pliki istnieją
    print("🔍 Checking for result files...")
    lstm_file = 'results/metrics/training_results.json'
    transformer_file = 'results/metrics/transformer_results.json'
    
    print(f"LSTM file exists: {os.path.exists(lstm_file)}")
    print(f"Transformer file exists: {os.path.exists(transformer_file)}")
    
    if os.path.exists(lstm_file):
        print(f"LSTM file size: {os.path.getsize(lstm_file)} bytes")
    if os.path.exists(transformer_file):
        print(f"Transformer file size: {os.path.getsize(transformer_file)} bytes")
    
    # Załaduj wyniki z poprawioną strukturą
    results = load_model_results()
    
    # Debug: pokaż końcowe załadowane dane
    print("\n🔍 Final loaded results:")
    for model, data in results.items():
        if data:
            print(f"  {model}: ✅ Loaded successfully")
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {model}: ❌ Failed to load")
    
    # Wyświetl PROSTE PORÓWNANIE
    print_comparison_table(results)
    
    # Stwórz PROSTE WYKRESY PORÓWNAWCZE
    if any(results.values()):
        print("\n📊 Creating simple comparison plots...")
        create_comparison_plots(results)
    else:
        print("\n❌ No data available for plotting")
    
    print("\n✅ Comparison completed!")

if __name__ == "__main__":
    main()