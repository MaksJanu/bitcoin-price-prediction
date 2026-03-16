# Bitcoin Price Prediction

This project predicts Bitcoin price movement for the next day using three approaches:
- LSTM sequence regression,
- Transformer sequence regression,
- Naive Bayes direction classification (Up/Down).

The goal is to compare deep learning and classical probabilistic modeling in terms of:
- next-day price prediction quality,
- direction prediction quality,
- training stability,
- interpretability through visual analytics.

## Project Workflow

1. Load raw historical BTC data.
2. Clean and standardize numerical values.
3. Engineer features (including MA_7, MA_30, RSI, date-based features, and volatility features).
4. Scale data and create time windows with sequence length 60.
5. Train models:
   - LSTM for price regression,
   - Stabilized Transformer for price regression,
   - Naive Bayes for direction classification.
6. Evaluate on held-out test data.
7. Save metrics as JSON and generate comparison plots.
8. Produce next-day forecasts with trend signals.

## Repository Structure

- data/
  - raw/ - source data
  - processed_data/ - processed data and saved scaler
  - preprocessing.py - data preparation and transformations
- models/
  - data_utils.py - scaling and sequence preparation utilities
  - lstm_model.py - LSTM architecture and training
  - transformer_model.py - Stabilized Transformer architecture and training
  - naive_bayes_model.py - GaussianNB direction classifier
  - prediction_utils.py - metrics and evaluation utilities
- scripts/
  - train_model_lstm.py
  - train_model_transformer.py
  - train_model_naive_bayes.py
- results/
  - metrics/ - training/test metrics in JSON format
  - plots/ - visualizations and model comparisons
- saved_models/ - trained models and checkpoints
- visualization/ - scripts for plotting and forecast reporting

## Libraries Used

### Data and Numerical Computing
- numpy - numerical operations and arrays.
- pandas - tabular data handling and feature preparation.

### Machine Learning and Deep Learning
- tensorflow / keras - building, training, and saving LSTM/Transformer models.
- scikit-learn - preprocessing, metrics, cross-validation, and GaussianNB.

### Visualization
- matplotlib - trend plots, training curves, and forecast charts.
- seaborn - correlation heatmaps and statistical plots.

### Artifact Persistence and Utilities
- joblib - saving/loading scalers and classical ML models.
- json - storing run configurations and metrics.
- os, sys, warnings, re - system and helper operations.

## Models and Settings

### LSTM (price regression)
- Sequence length: 60
- Prediction horizon: 1 day
- Epochs: 100
- Batch size: 16
- Input features: Open, High, Low, Volume, MA_7, MA_30, RSI

### Stabilized Transformer (price regression)
- Sequence length: 60
- Prediction horizon: 1 day
- Epochs: 100
- Batch size: 32
- Stabilization features:
  - BatchNormalization
  - reduced dropout
  - gradient clipping
  - conservative learning rate
  - reduced model complexity

### Naive Bayes (direction classification)
- Model type: GaussianNB
- Task: direction classification (Down / Up)
- Sequence length: 60
- Prediction horizon: 1 day
- var_smoothing: 1e-09

## Results and Evaluation

Metrics are taken from:
- results/metrics/training_results.json
- results/metrics/transformer_results.json
- results/metrics/naive_bayes_results.json

### 1) LSTM - Regression (test)
- MSE: 0.0029
- MAE: 0.0446
- RMSE: 0.0540
- R2: 0.9006

Conclusion: LSTM gives the best regression quality in the current experiment set.

### 2) Stabilized Transformer - Regression (test)
- MSE: 0.0072
- MAE: 0.0716
- RMSE: 0.0850
- R2: 0.7534

Conclusion: Transformer training is stable, but test accuracy is lower than LSTM in this run.

### 3) Naive Bayes - Direction Classification (test)
- Accuracy: 0.7841
- Precision (weighted): 0.7479
- Recall (weighted): 0.7841
- F1-score (weighted): 0.7149

Class Down:
- Precision: 0.6000
- Recall: 0.0857
- F1: 0.1500

Class Up:
- Precision: 0.7902
- Recall: 0.9837
- F1: 0.8764

Conclusion: the model strongly detects upward moves but misses many downward moves (low recall for Down), which suggests class imbalance effects.

## Selected Plot Visualizations

### LSTM vs Transformer comparison
Model metrics comparison (R2, MAE, MSE, RMSE):

![Model Comparison](results/plots/comparison/models_comparison.png)

LSTM training curves:

![LSTM Training Curves](results/plots/lstm/training_curves_combined.png)

Transformer training curves:

![Transformer Training Curves](results/plots/transformer/training_curves_combined.png)

### Time-series analysis of BTC
Price over time:

![BTC Price Over Time](results/plots/lstm/bitcoin_price_over_time.png)

Price with moving averages:

![BTC Moving Averages](results/plots/lstm/moving_averages.png)

Price distribution:

![BTC Price Distribution](results/plots/lstm/price_distribution.png)

Feature correlation matrix:

![BTC Correlation Matrix](results/plots/lstm/correlation_matrix.png)

### Forecast example
Single-day forecast visualization with uncertainty:

![Single Day Prediction](results/plots/lstm/single_day_prediction.png)

### Naive Bayes evaluation plots
Main classification dashboard (confusion matrix, metrics, confidence distribution):

![Naive Bayes Results](results/plots/naive_bayes/naive_bayes_results.png)

Top feature variance analysis:

![Naive Bayes Feature Variance](results/plots/naive_bayes/naive_bayes_feature_variance_analysis.png)

Naive Bayes feature correlation matrix:

![Naive Bayes Correlation Matrix](results/plots/naive_bayes/naive_bayes_features_correlation_matrix.png)

## Example Next-Day Predictions

Prediction scripts report:
- predicted next-day price,
- absolute and percentage change,
- trend signal (BULLISH/BEARISH),
- signal strength and interpretation notes.

Example outputs in this project include both bearish and bullish scenarios, which helps compare model behavior across different market conditions.

## How to Run

1. Prepare data:
   - run data/preprocessing.py
2. Train a model:
   - scripts/train_model_lstm.py
   - scripts/train_model_transformer.py
   - scripts/train_model_naive_bayes.py
3. Generate visualizations:
   - run scripts from the visualization/ folder

## Summary

- Best regression performance in current metrics: LSTM.
- Transformer offers stable training curves and a robust alternative baseline.
- Naive Bayes provides solid overall direction metrics but needs improvement for Down-class recall.
- The project includes complete artifacts: JSON metrics, saved models, and an extensive plot set for analysis.