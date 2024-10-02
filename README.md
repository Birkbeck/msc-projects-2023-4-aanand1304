Cyber Threat Forecasting Using Machine Learning

This project aims to forecast cyber threats using both univariate and multivariate machine learning approaches. The dataset incorporates cyber-attack data, public holidays, social media metrics, and other relevant features to predict future attack trends.

Project Structure

1. Data Preparation
hackmageddon: Includes data and scripts from Hackmageddon, an online resource for cyber-attack statistics.
Hackmageddon_sourcefromweb.xlsx: Source data retrieved from the Hackmageddon website.
Hackmageddon.csv: Preprocessed dataset extracted from Hackmageddon.
Nol_daily.csv & Nol_monthly.csv: Daily and monthly statistics datasets.
Nol_daily.py & Nol_monthly.py: Python scripts for processing daily and monthly statistics into the required formats.
monthly_Mentiondata: Scripts for handling mentions of cyber threats from various sources.
A_NoM.py: Script for processing mentions data.
API_Key.json: Contains necessary API keys for fetching external data.
PH_July2011_April2024.csv: Preprocessed public holiday data for various countries from July 2011 to April 2024.
Attacks_NoM_Jan2023_Apr2024.csv: Forecasted attack mentions data for the period from January 2023 to April 2024.
python_holidy.py & holiday.py: Scripts for collecting and processing holiday data for different countries.
2. Dataset
FinalDataset.csv: The main dataset used for training and forecasting cyber threats.
3. Model
modelfinding: Explores different models used in the project.
multivariate_Forecast.ipynb: Jupyter notebook for multivariate forecasting.
multivariate_hp_optimisation.ipynb: Hyperparameter optimization for multivariate models.
univariate_Forecast.ipynb: Jupyter notebook for univariate forecasting.
univariate_hp_optimisation.ipynb: Hyperparameter optimization for univariate models.
4. Plotting
prediction_plot_univariate_multivariate: Contains scripts for generating forecast plots.
multivariate_forecast_plot: Generates plots for multivariate forecast results.
univariate_forecast_plot: Generates plots for univariate forecast results.
other_plot: Contains additional relevant plots.
Validation_plot_univariate_multivariate: Scripts for generating validation plots for forecast models.
multivariateplot: Generates validation plots for multivariate models.
univariateplot: Generates validation plots for univariate models.
SHAP_Plot: Scripts for generating SHAP plots to explain model predictions and show feature importance.
5. Hyperparameters
hyperparameters_univariate_multivariate_optimization: Contains the best SMAPE values saved in JSON format for both univariate and multivariate models.
