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

