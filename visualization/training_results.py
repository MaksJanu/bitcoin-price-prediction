import matplotlib.pyplot as plt
import os

def create_training_results_plots(history, y_test, y_test_pred, test_r2, 
                                save_path='results/plots/lstm/lstm_training_results.png'):
    """Tworzy wykresy wyników treningu"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(15, 12))

    # Loss i MAE podczas treningu
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model MAE During Training', fontsize=14, fontweight='bold')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Predykcje vs rzeczywiste wartości
    plt.subplot(2, 3, 3)
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Predictions vs Actual (Test Set)\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Timeline predykcji
    plt.subplot(2, 3, 4)
    n_show = min(100, len(y_test))
    plt.plot(range(n_show), y_test[-n_show:], label='Actual', linewidth=2, alpha=0.8)
    plt.plot(range(n_show), y_test_pred[-n_show:], label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f'Price Prediction Timeline (Last {n_show} days)', fontsize=14, fontweight='bold')
    plt.xlabel('Days')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Błędy predykcji
    plt.subplot(2, 3, 5)
    errors = y_test - y_test_pred.flatten()
    plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Prediction Errors Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Błędy w czasie
    plt.subplot(2, 3, 6)
    plt.plot(range(len(errors)), errors, alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Test Sample')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path