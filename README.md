**Forecasting Cyber Threats Using Machine Learning**

**Overview**
Provide a brief introduction to the project here, including the main objectives, the problem it addresses, and a quick overview of the methodologies and models used.

**Table of Contents**
Project Structure
Setup Instructions
Data Preparation
Modeling
SHAP Analysis
Validation
Hyperparameters
Prediction and Visualization
Results
Conclusion

**Project Structure**

**1. Data_Preparation**
Contains scripts and files for data cleaning, feature extraction, and preparation for model training. Key files include:

ExploratoryDataAnalysis: Scripts for initial data exploration, visualizations, and data summaries.
Twitter, hackmageddon, monthly_Mentiondata, other features: Feature extraction methods or additional data sources.
python_holidy.py: Adds holiday-related data, potentially to adjust for seasonal or holiday effects.

**2. Dataset**
FinalDataset.csv: The main dataset used for model training and evaluation. Includes details on features, target variable, and any pre-processing steps required.

**3. Model**
Notebooks and scripts related to model training and hyperparameter optimization:

modelfinding: Scripts for model selection.
multivariate_Forecast.ipynb: Notebook for multivariate forecasting.
univariate_Forecast.ipynb: Notebook for univariate forecasting.
multivariate_hp_optimisation.ipynb: Hyperparameter optimization for multivariate models.
univariate_hp_optimisation.ipynb: Hyperparameter optimization for univariate models.

**4. SHAP_Plot**
Contains SHAP (SHapley Additive exPlanations) plots to visualize feature importance for various model predictions:

Advance.png, Advare.png, Backdoor-ALL.png, etc.: SHAP plots by category (e.g., Malware, Phishing), showing feature impact on predictions.

**5. Validation_plot_univariate_multivariate**
Validation plots for model performance evaluation:

multiVariateplot: Validation results for multivariate model predictions.
univariateplot: Validation results for univariate model predictions.

**6. hyperparameters_univariate_multivariate**
Documents hyperparameters for each model:

multivariate: Hyperparameter configurations for multivariate models.
univariate: Hyperparameter configurations for univariate models.

**7. prediction_plot_univariate_multivariate**
Prediction visualizations:

multivariate_forecast_plot: Forecast plots for multivariate models.
univariate_forecast_plot: Forecast plots for univariate models.
other plot: Additional prediction plots.


# Cyber Threat Forecasting

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

