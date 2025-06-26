import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_model_results():
    """Ładuje wyniki obu modeli"""
    results = {}
    
    # LSTM results
    try:
        with open('results/metrics/training_results.json', 'r') as f:
            lstm_results = json.load(f)
            results['LSTM'] = lstm_results['test_metrics']
    except FileNotFoundError:
        print("⚠️ LSTM results not found")
        results['LSTM'] = None
    
    # Transformer results
    try:
        with open('results/metrics/transformer_results.json', 'r') as f:
            transformer_results = json.load(f)
            results['Transformer'] = transformer_results['test_metrics']
    except FileNotFoundError:
        print("⚠️ Transformer results not found")
        results['Transformer'] = None
    
    return results

def create_comparison_plots(results):
    """Tworzy wykresy porównawcze"""
    os.makedirs('results/plots/comparison', exist_ok=True)
    
    if not results['LSTM'] or not results['Transformer']:
        print("Cannot create comparison - missing results")
        return
    
    # Metryki do porównania
    metrics = ['r2', 'mae', 'mse', 'rmse']
    models = ['LSTM', 'Transformer']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, color=['blue', 'red'], alpha=0.7)
        axes[i].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.upper())
        axes[i].grid(True, alpha=0.3)
        
        # Dodaj wartości na słupkach
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/plots/comparison/models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Model comparison saved to: results/plots/comparison/models_comparison.png")

def print_comparison_table(results):
    """Wyświetla tabelę porównawczą"""
    if not results['LSTM'] or not results['Transformer']:
        print("Cannot create comparison table - missing results")
        return
    
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON TABLE")
    print("="*60)
    
    df = pd.DataFrame(results).T
    df = df.round(6)
    
    print(df.to_string())
    
    # Określ lepszy model
    print("\n🏆 WINNER ANALYSIS:")
    
    # R² - wyższy jest lepszy
    if results['LSTM']['r2'] > results['Transformer']['r2']:
        print(f"  R² Score: LSTM wins ({results['LSTM']['r2']:.4f} vs {results['Transformer']['r2']:.4f})")
        r2_winner = 'LSTM'
    else:
        print(f"  R² Score: Transformer wins ({results['Transformer']['r2']:.4f} vs {results['LSTM']['r2']:.4f})")
        r2_winner = 'Transformer'
    
    # MAE - niższy jest lepszy
    if results['LSTM']['mae'] < results['Transformer']['mae']:
        print(f"  MAE: LSTM wins ({results['LSTM']['mae']:.6f} vs {results['Transformer']['mae']:.6f})")
        mae_winner = 'LSTM'
    else:
        print(f"  MAE: Transformer wins ({results['Transformer']['mae']:.6f} vs {results['LSTM']['mae']:.6f})")
        mae_winner = 'Transformer'
    
    # Overall winner
    if r2_winner == mae_winner:
        print(f"\n🥇 OVERALL WINNER: {r2_winner}")
    else:
        print(f"\n🤝 MIXED RESULTS: LSTM better at {'R²' if r2_winner == 'LSTM' else 'MAE'}, Transformer better at {'MAE' if mae_winner == 'Transformer' else 'R²'}")

def main():
    print("🔍 Bitcoin Prediction Models Comparison")
    print("="*50)
    
    # Załaduj wyniki
    results = load_model_results()
    
    # Wyświetl tabelę porównawczą
    print_comparison_table(results)
    
    # Stwórz wykresy porównawcze
    create_comparison_plots(results)
    
    print("\n✅ Comparison completed!")

if __name__ == "__main__":
    main()