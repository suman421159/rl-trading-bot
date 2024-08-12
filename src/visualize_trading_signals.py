import pandas as pd
import matplotlib.pyplot as plt

def load_data(stock_symbol):
    try:
        stock_data = pd.read_csv(f'data/{stock_symbol}_daily.csv')
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        stock_data.set_index('date', inplace=True)
        # Filter for the last 2 years
        start_date = pd.Timestamp.now() - pd.DateOffset(years=2)
        stock_data = stock_data[stock_data.index >= start_date]
        return stock_data
    except FileNotFoundError:
        print(f"Data file for {stock_symbol} not found.")
        return None

def load_trading_log():
    try:
        trading_log = pd.read_csv('output/trading_log.csv')
        trading_log['Date'] = pd.to_datetime(trading_log['Date'])
        # Filter for the last 2 years
        start_date = pd.Timestamp.now() - pd.DateOffset(years=2)
        trading_log = trading_log[trading_log['Date'] >= start_date]
        return trading_log
    except FileNotFoundError:
        print("Trading log file not found.")
        return None

def plot_signals(stock_symbol):
    stock_data = load_data(stock_symbol)
    trading_log = load_trading_log()

    if stock_data is not None and trading_log is not None:
        symbol_log = trading_log[trading_log['Symbol'] == stock_symbol]
        
        plt.figure(figsize=(14, 7))
        plt.plot(stock_data.index, stock_data['4. close'], label='Close Price', color='black', alpha=0.7)

        # Plot Buy signals
        buy_signals = symbol_log[symbol_log['Action'] == 'Buy']
        plt.scatter(buy_signals['Date'], buy_signals['Price'], color='green', label='Buy Signal', marker='^', alpha=1, s=100)

        # Plot Sell signals
        sell_signals = symbol_log[symbol_log['Action'] == 'Sell']
        plt.scatter(sell_signals['Date'], sell_signals['Price'], color='red', label='Sell Signal', marker='v', alpha=1, s=100)

        # Plot Hold signals
        hold_signals = symbol_log[symbol_log['Action'] == 'Hold']
        plt.scatter(hold_signals['Date'], hold_signals['Price'], color='gray', label='Hold Signal', marker='o', alpha=0.5, s=50)

        plt.title(f"Trading Signals Overlay on Stock Price - {stock_symbol}")
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    for symbol in symbols:
        plot_signals(symbol)
