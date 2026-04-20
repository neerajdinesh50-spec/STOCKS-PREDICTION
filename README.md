# Stock Market Price Prediction using Machine Learning

> ⚠️ **Disclaimer:** This project is for **educational purposes only**. Stock markets are inherently unpredictable. Do NOT use these predictions for real trading or investment decisions.

---

## 📌 Problem Statement

Predicting stock market prices is one of the most challenging problems in finance. Stock prices are influenced by countless factors — earnings, geopolitics, news sentiment, and market psychology — many of which are unquantifiable.

**Objective:** Build ML models that learn historical price patterns and predict the **next day's closing price** for a given stock (e.g., AAPL).

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `yfinance` | Download real-time & historical stock OHLCV data |
| `pandas` | Data manipulation & time-series handling |
| `numpy` | Numerical operations & array processing |
| `matplotlib` / `seaborn` | Visualization: trend plots, moving averages |
| `scikit-learn` | Linear Regression, Random Forest, MinMaxScaler |
| `tensorflow` / `keras` | Build & train the LSTM deep learning model |
| `streamlit` | Interactive web dashboard |
| `joblib` | Saving/loading sklearn models |

---

## 📁 Project Structure

```
stock_prediction/
├── app.py                    # Streamlit interactive dashboard
├── requirements.txt
├── README.md
├── data/                     # Downloaded CSVs + saved plots
├── models/                   # Trained model artifacts
│   ├── lstm_model.keras
│   ├── linear_regression.pkl
│   └── random_forest.pkl
├── notebooks/
│   └── stock_prediction.ipynb
└── src/
    ├── data_loader.py        # yfinance data download
    ├── preprocessing.py      # Scaling, sequences, train/test split
    ├── eda.py                # EDA visualization
    ├── train.py              # Train all models (CLI)
    ├── evaluate.py           # Compare model performance
    ├── predict.py            # Predict next-day price (CLI)
    └── models/
        ├── linear_model.py   # Linear Regression + Random Forest
        └── lstm_model.py     # LSTM (TensorFlow/Keras)
```

---

## ⚙️ Installation

```bash
# Clone or navigate to the project directory
cd stock_prediction

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Option A — Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
Opens an interactive browser UI. Enter a ticker (e.g., `AAPL`) and click **Fetch & Train**.

### Option B — Command Line Pipeline
```bash
# Step 1: Train all models
python src/train.py

# Step 2: View evaluation plots
python src/evaluate.py

# Step 3: Predict next-day price
python src/predict.py
```

---

## 🧠 Models Explained

### 1. Linear Regression
- Uses **yesterday's close price** to predict **today's close price**
- Very fast but assumes a linear relationship
- Serves as a **baseline** for comparison

### 2. Random Forest
- Ensemble of 100 decision trees
- Handles non-linear patterns better than Linear Regression
- Robust to noise and outliers

### 3. LSTM (Long Short-Term Memory)
- Recurrent neural network designed for **sequential data**
- Architecture: `LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(1)`
- Uses the last **60 days** of prices as input to predict the next day
- Training: Adam optimizer, EarlyStopping, ReduceLROnPlateau

---

## 📊 Sample Output

```
==============================
  Model Comparison Summary
==============================
  Model                   RMSE       MAE
  -----------------------------------------
  Linear Regression       $4.21      $3.10
  Random Forest           $3.87      $2.95
  LSTM                    $2.53      $1.84
==============================

  AAPL — Next-Day Prediction
==============================
  Last Close Price : $189.30
  Predicted Price  : $191.75
  Change           : +$2.45 (+1.29%) 📈 UP
```

---

## 🚀 Bonus Features Included

- [x] **Streamlit Dashboard** — Interactive UI with live training
- [x] **Multiple ticker support** — Change any stock in sidebar
- [x] **Moving average analysis** — 50-day & 100-day MA
- [x] **Volume visualization**
- [ ] Sentiment analysis (Twitter/News) — planned
- [ ] Hyperparameter tuning (Optuna)

---

## ⚠️ Limitations

- Models are trained only on **price history** — no fundamentals or news
- Stock markets are **non-stationary** (patterns change over time)
- High RMSE during volatile periods (earnings, macro events)
- LSTM predictions may lag real price movements

---

## 📄 License

MIT License — Free for educational and personal use.
