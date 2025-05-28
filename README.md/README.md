# 📊 Crypto Sentiment Forecasting Dashboard

A robust, real-time dashboard that predicts cryptocurrency price trends using machine learning, time series forecasting, and market sentiment derived from social media. The project integrates multiple models and techniques to deliver actionable insights for Bitcoin (BTC) and Ethereum (ETH).

---

## 🧠 Project Overview

This end-to-end project uses:

- Historical price data
- Social sentiment data (tweets)
- Forecasting models (ARIMA, SARIMA, Prophet, RF, XGBoost, LSTM, RNN)
- NLP-based sentiment scoring

### 🎯 Goal:

To classify and forecast trend direction (High, Low, Stable) for BTC and ETH using both numerical and sentiment-driven features — visualized through a dynamic dashboard.

---

## ⚙️ Tech Stack

| Layer               | Technology                                                    |
|---------------------|---------------------------------------------------------------|
| Frontend            | Dash (built on React.js)                                      |
| Backend             | Flask (used implicitly via Dash)                              |
| Data Processing & EDA| Pandas, NumPy, Matplotlib, Seaborn                           |
| ML Models           | scikit-learn (Random Forest, XGBoost), TensorFlow/Keras (LSTM, RNN) |
| NLP / Sentiment     | VADER (NLTK), TextBlob                                        |
| Forecasting         | statsmodels (ARIMA, SARIMA, Holt-Winters), Prophet            |
| Data Sources        | Yahoo Finance, Twitter Dataset                                |
| Deployment          | Dash Web App (locally or via server)                          |

> **Note:** Dash abstracts the frontend–backend layers internally — React.js is used for rendering, while Flask powers the backend routing.

---

## 🔁 Workflow Overview

### 📥 Data Acquisition & Preprocessing  
**Notebook:** `notebooks/Data_Acquisition_and_Preprocessing.ipynb`  
**Source:** Yahoo Finance API  
**Assets:** BTC-USD and ETH-USD  

**Steps:**  
1. Downloaded data using yfinance  
2. Cleaned missing values and anomalies  
3. Engineered features:  
   - Daily_Return  
   - Volatility (Rolling Std)  
   - MA_7, MA_30 (Moving Averages)  
4. Normalization done using Min-Max Scaling  

**Files Saved:**  
- `bitcoin_data.xlsx` – raw data  
- `bitcoin_normalized.xlsx` – for ML models  
- `bitcoin_unnormalized.xlsx` – for time series models and dashboard  
- Similar files created for Ethereum  

**Why Normalized & Unnormalized?**  
Normalized data improves ML model training performance. However, unnormalized data is required for interpretable forecasts in models like ARIMA, SARIMA, and Prophet.

---

## 📈 Time Series Analysis

### 🪙 Bitcoin  
**Notebook:** `notebooks/Time_Series_Analysis_of_Bitcoin.ipynb`  

**Models Used:**  
- Statistical: ARIMA, SARIMA, Holt-Winters  
- Machine Learning: Random Forest ✅, XGBoost  
- Deep Learning: LSTM, RNN, Prophet  

**RMSE Results:**  
- Random Forest: 0.0540 ✅ (Selected)  
- XGBoost: 0.0586  
- LSTM: 0.1588  
- RNN: 0.0895  

**Training/Test Split:** 89/11 except RNN (For RNN we used 50/50 split)  

**Key Insight:** RNN showed promising alignment with RF despite smaller training set.

---

### 🪙 Ethereum  
**Notebook:** `notebooks/Time_Series_Analysis_of_Ethereum.ipynb`  

**Models Used:**  
- ARIMA, SARIMA, Holt-Winters  
- Random Forest ✅ (Selected), XGBoost  
- LSTM, RNN, Prophet  

**RMSE Results:**  
- Random Forest: 0.0976 ✅ (Selected)  
- RNN: 0.0423  
- XGBoost: 0.1230  
- LSTM: 0.1505  

**Observation:** RNN performance was consistent across BTC & ETH using only 50% training data — reinforcing model robustness.

---

## 💬 Market Sentiment Analysis with VADER

**Notebook:** `notebooks/Market_Sentiment_Analysis_for_BTC_and_ETH.ipynb`  
**Data:** `data/Tweets_Crypto_2013-2021.csv`  
**Tool Used:** VADER from NLTK  

**Steps:**  
1. Preprocessed tweets: filled nulls, formatted dates  
2. Applied SentimentIntensityAnalyzer to calculate: compound, pos, neg, neu scores  
3. Aggregated daily compound scores → merged with BTC/ETH data  
4. Classified sentiment into:  
   - Very High (> 0.6)  
   - High (0.2 to 0.6)  
   - Neutral (-0.2 to 0.2)  
   - Low (-0.6 to -0.2)  
   - Very Low (< -0.6)  

**Correlation Findings:**  
- BTC Sentiment ↔ Daily Return: 0.0087  
- ETH Sentiment ↔ Daily Return: 0.0080  

**Insight:** While weak, sentiment signals still affect micro-trends.

**Final Datasets Used for Dashboard:**  
- `btc_with_sentiment.csv`  
- `eth_with_sentiment.csv`  

**Visualizations Included:**  
- Daily Return by Sentiment  
- Volatility by Sentiment  
- Volume by Sentiment  
- Price Up/Down Counts  

---

## 📊 Live Dashboard — `dashboard/app.py`

- Models update automatically based on user-selected date ranges and sentiment filters.  
- Enables forward-looking trend classification every time the user interacts.

**Tech Stack:**  
- Frontend: React.js (via Dash abstraction)  
- Backend: Flask (implicit via Dash)  
- Visualization: Plotly + Dash Core Components  

**Interaction:**  
- Dropdown for crypto selection (BTC/ETH)  
- Date and Sentiment sliders  
- Real-time forecasting output  
- Trend classification: 📈 High / 📉 Low / ⚖️ Stable  
- Visual advice like: “Sentiment Improving – Possible Bullish Trend”  

**Models Used:**  
- Pre-trained Random Forest Regressors  
  - `/models/rf_btc_model.pkl`  
  - `/models/rf_eth_model.pkl`  

---

🟣 Power BI Dashboard
This Power BI dashboard offers advanced visual insights for BTC and ETH, including price action, sentiment scores, and volatility patterns.

📂 Power BI dashboard — contains:

dashboard_btc.png

dashboard_eth.png

🗂️ Power BI file
Download the Power BI file used to generate the above visuals:

📄 Bitcoin and Ethereum.pbix

## 🚀 Future Scope

- Integration with live Twitter API for real-time sentiment updates  
- Enhanced explainability with SHAP/feature importance visualizations  
- Model retraining with live feedback loop  

---

## 📢 Call-to-Action

If you'd like to contribute, report issues, or suggest features — feel free to fork the repo and raise a pull request!

---

## 📌 Author

**Chaitanya Moudgil**  
Email: chaitanya.moudgil5112@gmail.com  
GitHub: [https://github.com/Chaitanya2026/](https://github.com/Chaitanya2026/)  
LinkedIn: [www.linkedin.com/in/chaitanya-moudgil-da](https://www.linkedin.com/in/chaitanya-moudgil-da)

