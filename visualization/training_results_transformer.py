import matplotlib.pyplot as plt
import numpy as np
import os

def create_transformer_training_results_plots(history, y_test, y_test_pred, test_r2, 
                                            save_path='results/plots/transformer/transformer_training_results.png'):
    """Tworzy wykresy wyników treningu dla modelu Transformer"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(15, 12))

    # Loss podczas treningu
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Transformer Loss During Training', fontsize=14, fontweight='bold')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # MAE podczas treningu
    plt.subplot(2, 3, 2)
    # Sprawdź dostępne nazwy metryk MAE
    mae_keys = ['mae', 'mean_absolute_error']
    mae_key = None
    for key in mae_keys:
        if key in history.history:
            mae_key = key
            break
    
    if mae_key:
        plt.plot(history.history[mae_key], label='Training MAE', linewidth=2)
        plt.plot(history.history[f'val_{mae_key}'], label='Validation MAE', linewidth=2)
        plt.title('Transformer MAE During Training', fontsize=14, fontweight='bold')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'MAE metrics not available', ha='center', va='center')
        plt.title('MAE Not Available', fontsize=14, fontweight='bold')

    # Predykcje vs rzeczywiste wartości
    plt.subplot(2, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Transformer Predictions vs Actual (Test Set)\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Timeline predykcji
    plt.subplot(2, 3, 4)
    n_show = min(100, len(y_test))
    plt.plot(range(n_show), y_test[-n_show:], label='Actual', linewidth=2, alpha=0.8)
    plt.plot(range(n_show), y_test_pred[-n_show:], label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f'Transformer Price Prediction Timeline (Last {n_show} days)', fontsize=14, fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Błędy predykcji
    plt.subplot(2, 3, 5)
    errors = y_test - y_test_pred
    plt.hist(errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.title('Transformer Prediction Errors Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Błędy w czasie
    plt.subplot(2, 3, 6)
    plt.plot(range(len(errors)), errors, alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Transformer Prediction Errors Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Sample')
    plt.ylabel('Prediction Error')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Transformer training results saved to: {save_path}")

    return save_path