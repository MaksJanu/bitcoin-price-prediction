import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def _clean_data(data):
    """Czyści dane z wartości problematycznych"""
    data = np.array(data)
    data = np.where(np.isinf(data), 0, data)
    data = np.where(np.isnan(data), 0, data)
    return data

def create_naive_bayes_results_plots(model, y_true, y_pred, y_proba, test_metrics, 
                                   save_path='results/plots/naive_bayes/naive_bayes_results.png'):
    """
    Tworzy wykresy wyników dla modelu Naive Bayes
    
    Args:
        model: Trained Naive Bayes model
        y_true: True labels
        y_pred: Predicted labels  
        y_proba: Prediction probabilities
        test_metrics: Dictionary with test metrics
        save_path: Path to save the plot
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Czyść dane wejściowe
    y_true = _clean_data(y_true)
    y_pred = _clean_data(y_pred)
    y_proba = _clean_data(y_proba)
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    try:
        # 1. Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.class_names, yticklabels=model.class_names)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 2. Classification Metrics Bar Chart
        plt.subplot(2, 3, 2)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            test_metrics.get('accuracy', 0),
            test_metrics.get('precision', 0), 
            test_metrics.get('recall', 0),
            test_metrics.get('f1_score', 0)
        ]
        
        # Czyść metryki
        metrics_values = _clean_data(metrics_values)
        
        bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Classification Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Prediction Probabilities Distribution
        plt.subplot(2, 3, 3)
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            max_proba = np.max(y_proba, axis=1)
        else:
            max_proba = np.array([0.5] * len(y_true))  # Wartość domyślna
        
        max_proba = _clean_data(max_proba)
        
        plt.hist(max_proba, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Maximum Probability')
        plt.ylabel('Frequency')
        
        mean_proba = np.mean(max_proba) if len(max_proba) > 0 else 0.5
        plt.axvline(mean_proba, color='red', linestyle='--', 
                    label=f'Mean: {mean_proba:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Class Distribution
        plt.subplot(2, 3, 4)
        unique, counts = np.unique(y_true, return_counts=True)
        
        if len(unique) > 0 and hasattr(model, 'class_names'):
            class_names_true = [model.class_names[i] if i < len(model.class_names) else f'Class_{i}' for i in unique]
            plt.pie(counts, labels=class_names_true, autopct='%1.1f%%', startangle=90)
            plt.title('True Class Distribution', fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No class data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('True Class Distribution (N/A)', fontsize=14, fontweight='bold')
        
        # 5. Prediction vs True Class Comparison
        plt.subplot(2, 3, 5)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        if hasattr(model, 'class_names') and len(model.class_names) > 0:
            x = np.arange(len(model.class_names))
            width = 0.35
            
            # Get counts for all classes (fill with 0 if class not present)
            true_counts = np.zeros(len(model.class_names))
            pred_counts = np.zeros(len(model.class_names))
            
            for i, count in zip(unique, counts):
                if i < len(true_counts):
                    true_counts[i] = count
            for i, count in zip(unique_pred, counts_pred):
                if i < len(pred_counts):
                    pred_counts[i] = count
            
            plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
            plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
            
            plt.title('True vs Predicted Class Counts', fontsize=14, fontweight='bold')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(x, model.class_names)
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Class comparison\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Class Comparison (N/A)', fontsize=14, fontweight='bold')
        
        # 6. Model Performance Summary
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Bezpieczne pobieranie metryk
        accuracy = test_metrics.get('accuracy', 0)
        precision = test_metrics.get('precision', 0)
        recall = test_metrics.get('recall', 0)
        f1_score = test_metrics.get('f1_score', 0)
        
        # Czyść metryki
        accuracy = float(_clean_data([accuracy])[0])
        precision = float(_clean_data([precision])[0])
        recall = float(_clean_data([recall])[0])
        f1_score = float(_clean_data([f1_score])[0])
        
        class_names_str = ', '.join(model.class_names) if hasattr(model, 'class_names') else 'Unknown'
        num_classes = len(model.class_names) if hasattr(model, 'class_names') else 0
        
        correct_predictions = np.sum(y_true == y_pred) if len(y_true) > 0 else 0
        avg_confidence = np.mean(max_proba) if len(max_proba) > 0 else 0
        
        summary_text = f"""
        Naive Bayes Model Performance Summary
        
        Overall Accuracy: {accuracy:.4f}
        Precision (weighted): {precision:.4f}
        Recall (weighted): {recall:.4f}
        F1-Score (weighted): {f1_score:.4f}
        
        Number of Classes: {num_classes}
        Classes: {class_names_str}
        
        Total Test Samples: {len(y_true)}
        Correct Predictions: {correct_predictions}
        
        Average Confidence: {avg_confidence:.4f}
        """
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")
        # Stwórz podstawowy wykres z informacją o błędzie
        plt.clf()
        plt.text(0.5, 0.5, f'Error creating plots:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Visualization Error', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Naive Bayes results plots saved to: {save_path}")
    return save_path


def create_class_wise_performance_plot(model, y_true, y_pred, test_metrics,
                                     save_path='results/plots/naive_bayes/class_performance.png'):
    """
    Tworzy wykres wydajności dla każdej klasy osobno
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Extract per-class metrics from classification report
        class_report = test_metrics.get('classification_report', {})
        
        if not class_report or not hasattr(model, 'class_names'):
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, 'Classification report not available', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Per-Class Performance (N/A)', fontsize=16, fontweight='bold')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return save_path
        
        classes = model.class_names
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for cls in classes:
            if cls in class_report:
                precision_scores.append(class_report[cls].get('precision', 0))
                recall_scores.append(class_report[cls].get('recall', 0))
                f1_scores.append(class_report[cls].get('f1-score', 0))
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        # Czyść dane
        precision_scores = _clean_data(precision_scores)
        recall_scores = _clean_data(recall_scores)
        f1_scores = _clean_data(f1_scores)
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
            plt.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
            plt.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=9)
        
    except Exception as e:
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, f'Error creating class performance plot:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Per-Class Performance (Error)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Class-wise performance plot saved to: {save_path}")
    return save_path


def create_prediction_confidence_analysis(y_proba, y_true, y_pred, model,
                                        save_path='results/plots/naive_bayes/confidence_analysis.png'):
    """
    Tworzy analizę pewności predykcji
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Czyść dane wejściowe
        y_true = _clean_data(y_true)
        y_pred = _clean_data(y_pred)
        y_proba = _clean_data(y_proba)
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            max_proba = np.max(y_proba, axis=1)
        else:
            max_proba = np.array([0.5] * len(y_true))
        
        max_proba = _clean_data(max_proba)
        correct_predictions = (y_true == y_pred)
        
        plt.figure(figsize=(15, 10))
        
        # 1. Confidence vs Accuracy
        plt.subplot(2, 2, 1)
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        
        accuracy_by_confidence = []
        counts_by_confidence = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_proba >= confidence_bins[i]) & (max_proba < confidence_bins[i+1])
            if np.sum(mask) > 0:
                accuracy = np.mean(correct_predictions[mask])
                count = np.sum(mask)
            else:
                accuracy = 0
                count = 0
            accuracy_by_confidence.append(accuracy)
            counts_by_confidence.append(count)
        
        plt.bar(bin_centers, accuracy_by_confidence, width=0.08, alpha=0.7, color='skyblue')
        plt.title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Level')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # 2. Sample Count by Confidence
        plt.subplot(2, 2, 2)
        plt.bar(bin_centers, counts_by_confidence, width=0.08, alpha=0.7, color='lightgreen')
        plt.title('Sample Count by Confidence Level', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Level')
        plt.ylabel('Number of Samples')
        plt.grid(True, alpha=0.3)
        
        # 3. Confidence Distribution by Correctness
        plt.subplot(2, 2, 3)
        if len(correct_predictions) > 0:
            correct_conf = max_proba[correct_predictions]
            incorrect_conf = max_proba[~correct_predictions]
            
            plt.hist([correct_conf, incorrect_conf], bins=20, alpha=0.7, 
                     label=['Correct', 'Incorrect'], color=['green', 'red'])
        else:
            plt.hist(max_proba, bins=20, alpha=0.7, color='blue')
        
        plt.title('Confidence Distribution by Prediction Correctness', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence Level')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Per-Class Confidence
        plt.subplot(2, 2, 4)
        if hasattr(model, 'class_names') and len(model.class_names) > 0:
            class_confidences = []
            class_labels = []
            
            for class_idx, class_name in enumerate(model.class_names):
                class_mask = y_pred == class_idx
                if np.sum(class_mask) > 0:
                    class_conf = max_proba[class_mask]
                    class_confidences.extend(class_conf)
                    class_labels.extend([class_name] * len(class_conf))
            
            if class_confidences:
                df_conf = pd.DataFrame({
                    'Confidence': class_confidences,
                    'Class': class_labels
                })
                
                sns.boxplot(data=df_conf, x='Class', y='Confidence')
                plt.title('Confidence Distribution by Predicted Class', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No confidence data\navailable', 
                        ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'Class names\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
    except Exception as e:
        plt.clf()
        plt.text(0.5, 0.5, f'Error creating confidence analysis:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Confidence Analysis (Error)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Confidence analysis plot saved to: {save_path}")
    return save_path