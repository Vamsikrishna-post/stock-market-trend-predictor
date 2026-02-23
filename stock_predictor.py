import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def load_and_preprocess_data(filename="stock_data.csv", lookback=10):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Please run download_data.py first.")
        return None, None, None, None
    
    # Load data skipping multi-index headers if present
    df = pd.read_csv(filename, skiprows=[1, 2])
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Use 'Close' price
    prices = df['Close'].values
    
    # Create simple features: use last 'lookback' days to predict current day
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i-lookback:i])
        y.append(prices[i])
        
    X, y = np.array(X), np.array(y)
    return X, y, prices, df['Date'].values

def run_prediction():
    lookback = 10
    X, y, original_prices, dates = load_and_preprocess_data("stock_data.csv", lookback)
    
    if X is None:
        return

    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Simple Linear Regression
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    print("Making predictions...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Predict future trend (next 30 days)
    future_input = X[-1].tolist()
    future_preds = []
    for _ in range(30):
        next_pred = model.predict([future_input])[0]
        future_preds.append(next_pred)
        # Update input window: remove oldest, add newest prediction
        future_input = future_input[1:] + [next_pred]
    
    # Plotting
    plt.figure(figsize=(16, 9))
    plt.style.use('dark_background')
    
    # Colors
    actual_color = '#00ff9d'
    pred_color = '#00d4ff'
    future_color = '#ff007c'
    
    # Plot Actual
    plt.plot(dates, original_prices, label='Actual Price', color=actual_color, linewidth=1.5, alpha=0.7)
    
    # Plot Test Predictions
    test_dates = dates[lookback + split:]
    plt.plot(test_dates, test_pred, label='Test Prediction', color=pred_color, linewidth=2, linestyle='--')
    
    # Plot Future Trend
    last_date = dates[-1]
    future_dates = pd.date_range(start=last_date, periods=31)[1:]
    plt.plot(future_dates, future_preds, label='Future Trend (30 Days)', color=future_color, linewidth=2.5)
    
    # Aesthetic touches
    plt.title('Stock Price Trend Predictor (Linear Regression)', fontsize=22, fontweight='bold', pad=20, color='white')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Price USD ($)', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12, loc='upper left', frameon=True, shadow=True)
    
    # Adding a subtle fill for the future region
    plt.axvspan(last_date, future_dates[-1], color=future_color, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('prediction_trend.png')
    print("Visualisation saved as 'prediction_trend.png'")
    # Show status
    mse = mean_squared_error(y_test, test_pred)
    print(f"Model Performance (MSE): {mse:.4f}")

if __name__ == "__main__":
    run_prediction()
