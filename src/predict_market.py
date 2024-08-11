import numpy as np
import pandas as pd
import os
from load_model import load_trained_model
from predictive_trading_env import PredictiveTradingEnv

def load_latest_data(directory, symbols):
    """ Load the latest stock data from CSV files for prediction. """
    stock_data = {}
    for symbol in symbols:
        file_path = os.path.join(directory, f'{symbol}_daily.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            if '4. close' not in data.columns:
                print(f"Missing '4. close' column in data for {symbol}")
                continue
            latest_data = data[['4. close']].iloc[-3000:]  # Last 3000 days of data
            stock_data[symbol] = latest_data
        else:
            print(f"Data file for {symbol} not found.")
    return stock_data

def make_prediction(model, env):
    """ Use the model to make a single prediction based on the latest data. """
    state = env.reset()
    action_probs = model.predict(np.array([state]))
    action_index = np.argmax(action_probs, axis=1)[0]
    num_actions = env.action_space.nvec[0]  
    action = np.unravel_index(action_index, (num_actions,) * env.num_stocks)
    action_mappings = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    action_readable = [action_mappings[a] for a in action]
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    action_dict = dict(zip(stock_symbols, action_readable))
    print(f"Chosen actions: {action_dict}")

if __name__ == "__main__":
    model_path = 'model/dqn_trading_model.h5'
    model = load_trained_model(model_path)
    
    data_directory = 'data'
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    stock_data = load_latest_data(data_directory, symbols)
    
    env = PredictiveTradingEnv(stock_data)
    make_prediction(model, env)
