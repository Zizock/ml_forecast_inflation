# Inflation Forecasting Project

**Current status:** four models under models/, complete.

**Working:** integrate into a main runner script

## Overview

This project extends the IMF Working Paper by Liu et al. (2024) on machine-learning based inflation forecasting. In their original study, the authors implement four classical supervised learning models, engineered for time series forecasting, to predict inflation in Japan. Japan provides a particularly informative testing ground, having experienced nearly three decades of near-zero inflation followed by an unusual inflation surge in recent years. This environment offers a natural setting to evaluate whether machine learning methods outperform traditional econometric time series models under conditions of high uncertainty.

In the original paper, the authors evaluate LASSO, Elastic Net, Random Forest, and XGBoost models. Among these approaches, LASSO is found to deliver the strongest forecasting performance.

I extend their analysis in two directions. First, I implement a more extensively tuned XGBoost model. Second, I introduce two deep learning approaches: a standard LSTM model and a custom implemented LSTNet architecture. For comparability, I also implement LASSO, the best-performing model in the original study, as well as a classical ARIMA model and a Bayesian VAR, which serve as econometric benchmark models.

## Data Structure

The data are expected to be provided in CSV format. Raw inputs are sourced from several different files and merged during preprocessing. After running the `data_processing.py` script, a single processed CSV file is generated with the following structure:

- **`Date`**: Date column
- **`X1–X24`**: Time series variables (see the list below for detailed descriptions)

### Monthly series
- **X1**: Inflation (base index)
- **X2**: USD–JPY exchange rate (raw series)
- **X3**: Chained PPI (YoY % change)
- **X4**: PPI index (base index)
- **X5**: SPPI (YoY % change)
- **X6**: Import price index (base index)
- **X7**: Import energy price index (base index)
- **X8**: Nikkei 225 index (raw series)
- **X9**: Unemployment rate (%)
- **X10**: Household income (YoY % change)
- **X11**: Consumption activity index (base index)
- **X12**: Shadow interest rate (%)
- **X13**: Loan size (raw series)
- **X14**: Industrial production index (base index)
- **X15**: Monetary base (raw series)

### Monthly series
- **X16**: Tourist arrivals (raw series)
- **X17**: ESP inflation expectations (%)

### Quarterly series
- **X18**: Tankan output price change (net percentage responses)
- **X19**: Tankan input price change (net percentage responses)
- **X20**: Real GDP (raw series)
- **X21**: BOJ output gap (%)
- **X22**: Household disposable income (YoY % change)
- **X23**: Government consumption (raw series)
- **X24**: Government investment (raw series)

### Converting quarterly date to monthly frequency

For quarterly series, I convert them to monthly frequency using only information available within each month in order to avoid information leakage.

### Feature transformation
For features that are in percentages, no transformation is needed. For features that are in levels or raw indices, I calculate YoY percentage changes.

### Feature engineering
For supervised learning models, I construct lagged and differenced features for each variable. In current version, the maximum lag length is set to three months.

## Setup and Workflow
Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
Currently, each model is executed independently. The `run_main.py` script will be updated laterfuture to run all models in a single trial.