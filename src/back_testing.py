import numpy as np
import pandas as pd
from load_model import load_trained_model
from predictive_trading_env import PredictiveTradingEnv

def backtest(model, data, symbols):
    env = PredictiveTradingEnv(data)
    transactions = []
    state = env.reset()
    done = False
    while not done:
        action_probs = model.predict(state.reshape(1, -1))
        action_index = np.argmax(action_probs, axis=1)[0]
        actions = np.unravel_index(action_index, env.action_space.nvec)

        next_state, reward, done, _ = env.step(actions)
        date = data[list(data.keys())[0]]['date'].iloc[env.current_step]

        for i, symbol in enumerate(symbols):
            action = actions[i]
            stock_price = data[symbol]['4. close'].iloc[env.current_step] if env.current_step < len(data[symbol]) else None
            if action == 1:
                transactions.append((date, symbol, 'Buy', stock_price))
            elif action == 2:
                transactions.append((date, symbol, 'Sell', stock_price))
            else:
                transactions.append((date, symbol, 'Hold', stock_price))
        state = next_state
    transaction_df = pd.DataFrame(transactions, columns=['Date', 'Symbol', 'Action', 'Price'])
    # Save the DataFrame to a CSV file
    transaction_df.to_csv('trading_log.csv', index=False)
    print(transaction_df)

if __name__ == "__main__":
    api_key = 'RNT4J074CZ9YNAFL'
    data = {symbol: pd.read_csv(f'data/{symbol}_daily.csv') for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']}
    # Sort the data to have the most recent date at the top for each symbol
    for symbol in data:
        data[symbol] = data[symbol].sort_values(by='date', ascending=False).reset_index(drop=True)

    model = load_trained_model('model/dqn_trading_model.h5')
    backtest(model, data, ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'])
