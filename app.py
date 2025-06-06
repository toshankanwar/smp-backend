import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import schedule
import time
import threading
import datetime
from flask import Flask, jsonify
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)
# Global variables
model_15m = None
scaler_15m = None
last_trained_time = None
last_price = None
last_sentiment = None


# **Fetch real-time Bitcoin data**
def fetch_bitcoin_data():
    print("📥 Fetching real-time Bitcoin data...")

    btc_15m = yf.download("BTC-USD", interval="15m", period="60d")

    if btc_15m.empty:
        print("⚠️ Error fetching BTC data")
        return None

    btc_15m = btc_15m.reset_index()

    # Convert timezone to IST (India)
    btc_15m["Datetime"] = btc_15m["Datetime"].dt.tz_convert("Asia/Kolkata")

    # Keep only DateTime and Close Price
    btc_15m = btc_15m[["Datetime", "Close"]]

    return btc_15m


# **Fetch News and Perform Sentiment Analysis**
def fetch_news_and_sentiment():
    print("📥 Fetching Bitcoin news...")

    url = "https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&language=en&apiKey=d7f70c4bdbf140bb8a72949a9e541209"
    response = requests.get(url)
    news_data = response.json()

    if "articles" not in news_data:
        print("⚠️ Error fetching news")
        return 0  # Default sentiment

    # Extract headlines
    headlines = [article["title"] for article in news_data["articles"][:10]]

    # Compute sentiment
    sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
    avg_sentiment = np.mean(sentiments) if sentiments else 0

    print(f"📰 Sentiment Score: {avg_sentiment:.2f}")
    return avg_sentiment


# **Train LSTM Model**
def train_model():
    global model_15m, scaler_15m, last_trained_time, last_price, last_sentiment

    btc_15m = fetch_bitcoin_data()
    if btc_15m is None:
        return

    latest_price = btc_15m["Close"].iloc[-1]  # Get the latest closing price
    latest_sentiment = fetch_news_and_sentiment()

    # Check if new data is available
    if last_price is not None and last_sentiment is not None:
        if float(last_price) == float(latest_price) and float(last_sentiment) == float(latest_sentiment):
            print("⚠️ No new data, skipping retraining.")
            return  

    # Update tracking variables
    last_price = latest_price
    last_sentiment = latest_sentiment
    last_trained_time = datetime.datetime.now()

    print("🔄 Training LSTM Model with latest data...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_15m = scaler.fit_transform(btc_15m["Close"].values.reshape(-1, 1))

    def create_sequences(data, time_steps=60):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i+time_steps])
            y.append(data[i+time_steps])
        return np.array(X), np.array(y)

    X_train_15m, y_train_15m = create_sequences(scaled_15m)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_15m.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train_15m, y_train_15m, epochs=5, batch_size=32)

    print("✅ Training completed!")

    model_15m = model
    scaler_15m = scaler


# **Make Predictions**
def make_prediction(model, data):
    last_60 = data["Close"].values[-60:].reshape(-1, 1)
    last_60_scaled = scaler_15m.transform(last_60)
    last_60_scaled = np.reshape(last_60_scaled, (1, 60, 1))

    predicted_price = model.predict(last_60_scaled)
    predicted_price = scaler_15m.inverse_transform(predicted_price)

    return round(float(predicted_price[0][0]), 2)


# **API Endpoint for Predictions**
@app.route("/predict", methods=["GET"])
def predict():
    btc_15m = fetch_bitcoin_data()
    if btc_15m is None:
        return jsonify({"error": "Failed to fetch Bitcoin data"}), 500

    sentiment = fetch_news_and_sentiment()

    if model_15m is None:
        return jsonify({"error": "Model is not trained yet"}), 500

    pred_15m = make_prediction(model_15m, btc_15m)

    response_data = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment_score": round(sentiment, 2),
        "prediction_15m": pred_15m
    }

    # Print API response data to terminal for debugging
    print("\n📢 API Response Sent:\n", json.dumps(response_data, indent=4))

    return jsonify(response_data)


# **Schedule tasks**
def schedule_jobs():
    schedule.every(15).minutes.do(train_model)

    while True:
        schedule.run_pending()
        time.sleep(60)


# **Start scheduler in a separate thread**
threading.Thread(target=schedule_jobs, daemon=True).start()

# **Run Flask App**
if __name__ == "__main__":
    train_model()  # Initial training
    app.run(host="0.0.0.0", port=5000, debug=True)
