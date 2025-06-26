import matplotlib.pyplot as plt
import numpy as np
import os

def create_transformer_future_predictions_plot(y_test, future_predictions, test_mae, prediction_horizon,
                                             save_path='results/plots/transformer/transformer_future_predictions.png'):
    """Tworzy wykres predykcji przyszłości dla modelu Transformer"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))

    # Ostatnie 30 dni rzeczywistych danych + predykcje
    n_historical = 30
    historical_prices = y_test[-n_historical:]
    days = np.arange(-n_historical, prediction_horizon)

    plt.plot(days[:n_historical], historical_prices, 'b-', linewidth=2, 
             label='Historical Prices', alpha=0.8)
    plt.plot(days[n_historical-1:], 
             np.concatenate([[historical_prices[-1]], future_predictions]), 
             'r--', linewidth=2, label=f'Transformer Predictions ({prediction_horizon} days)', alpha=0.8)
    
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7, label='Today')
    plt.fill_between(days[n_historical-1:], 
                     np.concatenate([[historical_prices[-1]], future_predictions]) - test_mae,
                     np.concatenate([[historical_prices[-1]], future_predictions]) + test_mae,
                     alpha=0.3, color='red', label=f'Uncertainty (±{test_mae:.4f})')

    plt.title('Bitcoin Price Prediction - Transformer Model (Historical vs Future)', fontsize=14, fontweight='bold')
    plt.xlabel('Days from Today')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Transformer future predictions saved to: {save_path}")

    return save_path

def create_attention_heatmap(attention_weights, save_path='results/plots/transformer/attention_heatmap.png'):
    """Tworzy heatmapę wag attention (jeśli dostępne)"""
    if attention_weights is None:
        print("Attention weights not available")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('Transformer Attention Weights Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Input Position')
    plt.ylabel('Output Position')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Attention heatmap saved to: {save_path}")

    return save_path