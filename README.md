
# Forecasting Cyber Threats Using Machine Learning

## Data Collection and Preprocessing

Data sources include:

- **Hackmageddon**: Aggregated records of cyber-attacks.
- **Social Media and Internet Usage**: Monthly estimates based on growth trends in major platforms and user statistics.
- **Economic Indicators**: GDP growth rates by country, integrated to account for economic factors influencing cyber threats.
- **War and Conflict Events**: Synthetic data generated using GANs, capturing the influence of geopolitical events on cyber-attack trends.

The dataset is structured with **monthly granularity**, allowing for consistent and robust time-series forecasting.

## Feature Selection and Engineering

- **Univariate Forecasting**: Focuses on individual attack types, optimizing predictions for each specific cyber threat.
- **Multivariate Forecasting**: Employs **BorutaPy** for feature selection to capture interactions among features, enhancing predictions for multiple attack types simultaneously.

## Model Architecture and Training

The core forecasting model uses **LSTM** due to its ability to capture long-term dependencies in time-series data:

- **LSTM-Dropout Architecture**: Incorporates dropout layers to reduce overfitting.
- **Monte Carlo Dropout**: Provides uncertainty estimates in forecasts by generating multiple predictions with random neuron deactivation.
- **Hyperparameter Tuning**: Optimized via **Random Search** for dropout rates, activation functions, batch sizes, and learning rates.

## Evaluation Metrics and Results

- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Primary metric, chosen for its accuracy in evaluating time-series predictions.
- **MAE and RMSE**: Complementary metrics to assess model accuracy.
- **Confidence Intervals**: Generated using Monte Carlo dropout, indicating prediction reliability.

## SHAP Model Explainability

**SHAP values** were used for feature interpretability, revealing the impact of each feature on model predictions. This analysis provides transparency, showing which factors (e.g., social media growth, economic indicators) most significantly influence cyber threats.

## Conclusion

This project advances cyber threat forecasting by integrating comprehensive data sources and leveraging an **LSTM-based model** for long-term prediction. The approach, which includes feature importance analysis with SHAP and uncertainty estimation through Monte Carlo dropout, provides actionable insights to support proactive cybersecurity strategies.

