import numpy as np
import pandas as pd
import os  
from load_model import load_trained_model
from trading_env import TradingEnv

def load_latest_data(directory, symbols):
    """ Load the latest stock data from CSV files for prediction. """
    stock_data = {}
    for symbol in symbols:
        file_path = os.path.join(directory, f'{symbol}_daily.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col='date')
            # Get the last 30 days of data, ensuring it remains a DataFrame
            latest_data = data.iloc[-30:]
            stock_data[symbol] = latest_data[['4. close']]  # Keep it as a DataFrame
        else:
            print(f"Data file for {symbol} not found.")
    return stock_data


def make_prediction(model, env):
    """ Use the model to make a single prediction based on the latest data. """
    state = env.reset() 
    action_probs = model.predict(state.reshape(1, -1))
    action = np.argmax(action_probs, axis=1)
    print(f"Predicted action probabilities: {action_probs}")
    print(f"Chosen action: {action}")

if __name__ == "__main__":
    model_path = 'model/dqn_trading_model.h5'
    model = load_trained_model(model_path)
    
    data_directory = 'data'
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    stock_data = load_latest_data(data_directory, symbols)
    
    env = TradingEnv(stock_data)  
    
    make_prediction(model, env)
