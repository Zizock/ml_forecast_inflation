# ml_forecast_inflation
Working project: ML based inflation forecasting

Current status: four models under models/, complete.

Working: integrate into a main runner script

This is a ML forecasting project extending IMF working paper Liu et al., 2024.

This project uses 20+ macroeconomic features (and engineered features in the supervised models) to predict inflation.

The project currently includes four models: Lasso, XGBoost, LSTM, and LSTNet.

In Liu et al. (2024), four supervised learning models are implemented, and Lasso is found to perform best among them. I extend their analysis by implementing a more heavily tuned XGboost and two representative deep learning methods, both of which outperform LASSO for some forecasting horizons. My implementation of XGBoost also performs substantially better than the version reported in their paper.

This difference may also (probably) partly attribute to the fact that the feature sets used in their study and in my analysis do not fully overlap.

