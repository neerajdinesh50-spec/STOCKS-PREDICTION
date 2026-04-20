import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Stock Market Price Prediction using Machine Learning\n",
                "\n",
                "> **Objective:** Predict the next day's closing price for a stock (e.g., AAPL) using both traditional Machine Learning (Linear Regression, Random Forest) and Deep Learning (LSTM).\n",
                "\n",
                "## 1. Project Overview\n",
                "- **Problem Statement:** Stock prices are volatile and influenced by many factors. Can history provide insights into the future?\n",
                "- **Real-World Relevance:** Algorithmic trading, personal finance. \n",
                "- **Limitations:** No model predicts the future perfectly; news, sentiment, and macroeconomics matter more than just history.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Tech Stack Setup\n",
                "We use `yfinance` to fetch data, `pandas`/`numpy` for processing, `matplotlib`/`seaborn` for EDA, and `scikit-learn`/`tensorflow` for modeling."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import yfinance as yf\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.preprocessing import MinMaxScaler\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
                "from tensorflow.keras.models import Sequential\n",
                "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
                "\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "plt.style.use('seaborn-v0_8-darkgrid')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Collection\n",
                "Let's fetch historical data for Apple (AAPL)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ticker = 'AAPL'\n",
                "period = '5y'\n",
                "df = yf.download(ticker, period=period, auto_adjust=True)\n",
                "if isinstance(df.columns, pd.MultiIndex):\n",
                "    df.columns = df.columns.get_level_values(0)\n",
                "\n",
                "print(df.tail())\n",
                "print(\"\\nTotal Records:\", len(df))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Exploratory Data Analysis (EDA)\n",
                "Let's visualize the price trend and add Moving Averages (50-day and 100-day)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df['MA50'] = df['Close'].rolling(50).mean()\n",
                "df['MA100'] = df['Close'].rolling(100).mean()\n",
                "\n",
                "plt.figure(figsize=(14,5))\n",
                "plt.plot(df.index, df['Close'], label='Close', alpha=0.8)\n",
                "plt.plot(df.index, df['MA50'], label='50-Day MA')\n",
                "plt.plot(df.index, df['MA100'], label='100-Day MA')\n",
                "plt.title(f\"{ticker} Close Price & Moving Averages\")\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Data Preprocessing\n",
                "We need to explicitly sequence the data for LSTM (using past 60 days to predict the next day)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature = df['Close'].dropna().values.reshape(-1, 1)\n",
                "scaler = MinMaxScaler(feature_range=(0,1))\n",
                "scaled_data = scaler.fit_transform(feature)\n",
                "\n",
                "split_idx = int(len(scaled_data) * 0.8)\n",
                "train_data = scaled_data[:split_idx]\n",
                "test_data = scaled_data[split_idx:]\n",
                "\n",
                "def create_sequences(data, lookback=60):\n",
                "    X, y = [], []\n",
                "    for i in range(lookback, len(data)):\n",
                "        X.append(data[i-lookback:i, 0])\n",
                "        y.append(data[i, 0])\n",
                "    return np.array(X)[..., np.newaxis], np.array(y)\n",
                "\n",
                "X_train, y_train = create_sequences(train_data)\n",
                "\n",
                "# Prepend enough test context for LSTM testing sequences\n",
                "test_input = scaled_data[split_idx - 60:]\n",
                "X_test, y_test = create_sequences(test_input)\n",
                "\n",
                "print(\"Training sequences shape:\", X_train.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model 1: Deep Learning (LSTM)\n",
                "LSTM networks have memory properties suitable for time-series."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lstm_model = Sequential([\n",
                "    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),\n",
                "    Dropout(0.2),\n",
                "    LSTM(64, return_sequences=False),\n",
                "    Dropout(0.2),\n",
                "    Dense(1)\n",
                "])\n",
                "lstm_model.compile(optimizer='adam', loss='mean_squared_error')\n",
                "history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Model 2: Traditional ML (Random Forest)\n",
                "Let's compare baseline traditional predictions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_tr_rf = train_data[:-1, 0].reshape(-1, 1)\n",
                "y_tr_rf = train_data[1:, 0]\n",
                "X_te_rf = test_data[:-1, 0].reshape(-1, 1)\n",
                "y_te_rf = test_data[1:, 0]\n",
                "\n",
                "rf_model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
                "rf_model.fit(X_tr_rf, y_tr_rf)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Evaluation & Comparison\n",
                "Measuring performance using RMSE and MAE on the True (USD) scale."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "lstm_preds = scaler.inverse_transform(lstm_model.predict(X_test))\n",
                "lstm_actual = scaler.inverse_transform(y_test.reshape(-1, 1))\n",
                "\n",
                "rf_preds = scaler.inverse_transform(rf_model.predict(X_te_rf).reshape(-1,1))\n",
                "rf_actual = scaler.inverse_transform(y_te_rf.reshape(-1, 1))\n",
                "\n",
                "print(\"LSTM RMSE:\", np.sqrt(mean_squared_error(lstm_actual, lstm_preds)))\n",
                "print(\"LSTM MAE:\", mean_absolute_error(lstm_actual, lstm_preds))\n",
                "print(\"RF RMSE:\", np.sqrt(mean_squared_error(rf_actual, rf_preds)))\n",
                "print(\"RF MAE:\", mean_absolute_error(rf_actual, rf_preds))\n",
                "\n",
                "plt.figure(figsize=(14,5))\n",
                "plt.plot(lstm_actual, label='Actual Price', color='grey')\n",
                "plt.plot(lstm_preds, label='LSTM Predicted', color='green')\n",
                "plt.title('LSTM Actual vs Predicted')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Next Day Prediction"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "last_60_days = df['Close'][-60:].values.reshape(-1, 1)\n",
                "last_60_scaled = scaler.transform(last_60_days)\n",
                "X_next = last_60_scaled.reshape(1, 60, 1)\n",
                "pred_next_scaled = lstm_model.predict(X_next)\n",
                "pred_price = scaler.inverse_transform(pred_next_scaled)[0][0]\n",
                "print(\"Predicted Next Day Price: $\", round(pred_price, 2))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

os.makedirs(r"C:\Users\Admin\Documents\AIML\stock_prediction\notebooks", exist_ok=True)
with open(r"C:\Users\Admin\Documents\AIML\stock_prediction\notebooks\stock_prediction.ipynb", "w") as f:
    json.dump(notebook, f, indent=4)
