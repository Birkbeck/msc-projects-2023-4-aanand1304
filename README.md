# Cyber Threat Forecasting

## Overview

This project, titled **Forecasting Cyber Threats Using Machine Learning**, addresses the need for proactive cybersecurity by building advanced machine learning models to predict long-term cyber threat trends. The project integrates data from diverse sources, including historical cyber-attack records, social media metrics, internet usage, and socioeconomic indicators, to create a comprehensive predictive framework. By leveraging **Long Short-Term Memory (LSTM)** networks with **Monte Carlo dropout** for uncertainty estimation and **BorutaPy** for feature selection, the model provides an interpretable, long-term threat forecasting tool, empowering organizations to anticipate and mitigate emerging cyber risks.

## Table of Contents
1. Overview
2. Project Structure
3. Setup Instructions
4. Data Collection and Preprocessing
5. Feature Selection and Engineering
6. Model Architecture and Training
7. Evaluation Metrics and Results
8. SHAP Model Explainability
9. Conclusion

## Project Structure

- **Data_Preparation**: Contains scripts for data cleaning, feature extraction, and restructuring. Includes data on specific cyber events, holidays, internet usage, and social media.
- **Dataset**: The main dataset file, `FinalDataset.csv`, consolidates cyber-attack incidents with auxiliary data sources for analysis.
- **Model**: Notebooks for training both univariate and multivariate forecasting models, optimizing hyperparameters, and implementing Monte Carlo dropout for uncertainty estimation.
- **SHAP_Plot**: Visualizations using SHAP values to interpret the importance of various features influencing model predictions.
- **Validation_plot_univariate_multivariate**: Validation plots for assessing model performance across different forecasting approaches.
- **hyperparameters_univariate_multivariate**: Configurations and settings for tuning both univariate and multivariate models.
- **prediction_plot_univariate_multivariate**: Prediction plots showcasing forecasted trends, enabling users to visually assess model accuracy.

## Data Collection and Preprocessing

**Data sources include:**

- **Hackmageddon**: Aggregated records of cyber-attacks.
- **Social Media and Internet Usage**: Monthly estimates based on growth trends in major platforms.
- **Economic Indicators**: GDP growth rates by country, capturing economic factors influencing cyber threats.
- **War and Conflict Events**: Synthetic data generated using GANs, representing the influence of geopolitical events on cyber-attack trends.

The dataset is structured with **monthly granularity**, allowing for consistent and robust time-series forecasting.

## Feature Selection and Engineering

- **Univariate Forecasting**: Focuses on individual attack types, optimizing predictions for each specific cyber threat.
- **Multivariate Forecasting**: Employs **BorutaPy** for feature selection to capture interactions among features, enhancing predictions for multiple attack types simultaneously.

## Model Architecture and Training

The core forecasting model uses **LSTM**, chosen for its ability to capture long-term dependencies in time-series data:

- **LSTM-Dropout Architecture**: Integrates dropout layers to reduce overfitting.
- **Monte Carlo Dropout**: Provides uncertainty estimates by generating multiple predictions with random neuron deactivation.
- **Hyperparameter Tuning**: Optimized via **Random Search** for dropout rates, activation functions, batch sizes, and learning rates.

## Evaluation Metrics and Results

- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Primary metric, chosen for its accuracy in evaluating time-series predictions.
- **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**: Complementary metrics to assess model accuracy.
- **Confidence Intervals**: Generated using Monte Carlo dropout, providing a measure of prediction reliability.

## SHAP Model Explainability

For interpretability, **SHAP (Shapley Additive Explanations)** values are calculated to explain each feature's impact on predictions. By providing insights into feature importance, SHAP analysis reveals which factors (e.g., social media activity, economic indicators, holidays) significantly affect cyber threats. This transparency enhances trust in the model and aids decision-makers in understanding critical factors.

## Conclusion

This project advances cyber threat forecasting by integrating diverse data sources and leveraging an **LSTM-based model** for long-term prediction. The inclusion of **SHAP** for model interpretability and **Monte Carlo dropout** for uncertainty estimation provides actionable insights that support a proactive cybersecurity approach. This framework enables organizations to better allocate resources, anticipate risks, and mitigate future cyber threats effectively.


