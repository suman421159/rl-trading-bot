from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os
import time

def fetch_and_save_data(symbol, api_key, force_update=False):
    directory = 'data'
    filepath = f'{directory}/{symbol}_daily.csv'

    if not force_update and os.path.exists(filepath):
        print(f"Data for {symbol} already exists. Loading from file.")
        return pd.read_csv(filepath)

    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        data['pct_change'] = data['4. close'].pct_change().fillna(0)
        data['moving_avg_50'] = data['4. close'].rolling(window=50).mean()
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

    if not os.path.exists(directory):
        os.makedirs(directory)
    data.to_csv(filepath)
    print(f"Data for {symbol} saved successfully.")
    return data[['4. close', 'pct_change', 'moving_avg_50']].dropna()

def get_all_data(api_key, force_update=False):
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    all_data = {}
    for symbol in symbols:
        data = fetch_and_save_data(symbol, api_key, force_update=force_update)
        all_data[symbol] = data
        time.sleep(12)  # Respect API limits
    return all_data

if __name__ == "__main__":
    api_key = 'RNT4J074CZ9YNAFL'
    get_all_data(api_key, force_update=False)
