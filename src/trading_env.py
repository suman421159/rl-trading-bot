import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(TradingEnv, self).__init__()
        self.stock_data = stock_data
        self.num_stocks = len(stock_data)
        self.current_step = 0
        
        # Actions: 0 = hold, 1 = buy, 2 = sell for each stock
        self.action_space = gym.spaces.MultiDiscrete([3] * self.num_stocks)
        
        # State includes each stock's normalized price and holding status
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_stocks * 2,), dtype=np.float32)
        
        # Portfolio to track holding status and buying prices
        self.portfolio = {symbol: {'held': False, 'buy_price': 0} for symbol in stock_data.keys()}

    def reset(self):
        self.current_step = 0
        for symbol in self.portfolio:
            self.portfolio[symbol] = {'held': False, 'buy_price': 0}
        return self._get_observation()

    def _get_observation(self):
        prices = []
        holdings = []
        for symbol in self.stock_data:
            data = self.stock_data[symbol]
            # Check if data is a DataFrame and contains '4. close'
            if not data.empty and '4. close' in data.columns:
                price = data['4. close'].iloc[self.current_step] if self.current_step < len(data) else data['4. close'].iloc[-1]
            else:
                price = 0  # Default price if data is not available or malformed
            holdings.append(1 if self.portfolio[symbol]['held'] else 0)
            prices.append(price)

        return np.array(prices + holdings, dtype=np.float32)


    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(next(iter(self.stock_data.values()))):
            done = True
            next_state = np.zeros(self.num_stocks * 2)
        else:
            done = False
            next_state = self._get_observation()

        reward = self.calculate_reward(action)
        return next_state, reward, done, {}

    def calculate_reward(self, action):
        reward = 0
        for i, symbol in enumerate(self.stock_data):
            current_price = self.stock_data[symbol]['4. close'].iloc[self.current_step]
            if action[i] == 1 and not self.portfolio[symbol]['held']:  # Buy
                self.portfolio[symbol]['held'] = True
                self.portfolio[symbol]['buy_price'] = current_price
            elif action[i] == 2 and self.portfolio[symbol]['held']:  # Sell
                reward += current_price - self.portfolio[symbol]['buy_price']
                self.portfolio[symbol]['held'] = False
        return reward

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print("Portfolio:", self.portfolio)
            print("Stock Prices:", np.array([self.stock_data[symbol]['4. close'].iloc[self.current_step] for symbol in self.stock_data]))

    def close(self):
        pass
