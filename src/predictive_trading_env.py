import gym
import numpy as np
import pandas as pd

class PredictiveTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(PredictiveTradingEnv, self).__init__()
        self.stock_data = stock_data
        self.num_stocks = len(stock_data)
        self.current_step = 0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.MultiDiscrete([3] * self.num_stocks)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.num_stocks * 2,), dtype=np.float32)
        # Initialize or reset portfolio
        self.portfolio = {symbol: {'held': False, 'buy_price': 0, 'quantity': 0} for symbol in stock_data.keys()}

    def reset(self):
        self.current_step = 0
        initial_state = []
        for symbol, data in self.stock_data.items():
            if isinstance(data, pd.DataFrame) and '4. close' in data.columns:
                price = data['4. close'].iloc[0] if len(data) > 0 else 0
                held_status = 1 if self.portfolio[symbol]['held'] else 0
                initial_state.extend([price, held_status])
            else:
                initial_state.extend([0, 0])
        return np.array(initial_state)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= min(len(data) for data in self.stock_data.values())  # End if the shortest dataset ends
        next_state = []
        reward = 0
        for symbol in self.stock_data:
            if self.current_step < len(self.stock_data[symbol]):
                price = self.stock_data[symbol]['4. close'].iloc[self.current_step]
                held_status = 1 if self.portfolio[symbol]['held'] else 0
                next_state.extend([price, held_status])
                reward += self.calculate_reward(action, price, symbol, held_status)
            else:
                next_state.extend([0, 0])  
        return np.array(next_state), reward, done, {}

    def calculate_reward(self, action, price, symbol, held_status):
        reward = 0
        symbols_list = list(self.stock_data.keys())  
        index = symbols_list.index(symbol)  
        if action[index] == 1 and not held_status:
            self.portfolio[symbol]['held'] = True
            self.portfolio[symbol]['buy_price'] = price
            self.portfolio[symbol]['quantity'] = 100
        elif action[index] == 2 and held_status:
            reward += (price - self.portfolio[symbol]['buy_price']) * self.portfolio[symbol]['quantity']
            self.portfolio[symbol]['held'] = False
        return reward

    def render(self, mode='human'):
        portfolio_status = {symbol: "held" if self.portfolio[symbol]['held'] else "not held" for symbol in self.stock_data}
        print(f"Step: {self.current_step}, Portfolio: {portfolio_status}, Prices: {[self.stock_data[symbol]['4. close'].iloc[self.current_step] for symbol in self.stock_data if self.current_step < len(self.stock_data[symbol])]}")

    def close(self):
        pass
