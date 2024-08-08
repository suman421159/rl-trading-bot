import numpy as np
import tensorflow as tf
from fetch_data import get_all_data
from trading_env import TradingEnv
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import time
import os

# Suppressing TensorFlow logging to have a cleaner look while training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the DQN model
def create_model(state_size, action_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_dim=state_size), 
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Initialize environment and model
api_key = 'RNT4J074CZ9YNAFL'
stock_data = get_all_data(api_key)
env = TradingEnv(stock_data)
state_size = env.observation_space.shape[0]
action_size = int(np.prod(env.action_space.nvec))

model = create_model(state_size, action_size)
target_model = create_model(state_size, action_size)
target_model.set_weights(model.get_weights())

# Experience replay buffers
memory = deque(maxlen=2000)

# Hyperparameters
gamma = 0.95  
epsilon = 1.0  
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 16  

# Training loop
start_time = time.time()
total_episodes = 10  
steps_per_episode = 100  

for e in range(total_episodes):
    state = env.reset()
    print("State shape before reshaping:", state.shape)
    state = np.reshape(state, [1, state_size]) 
    episode_start_time = time.time()
    for time_step in range(steps_per_episode):
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  
        else:
            action_values = model.predict(state, verbose=0)
            action_index = np.argmax(action_values[0])
            action = np.unravel_index(action_index, env.action_space.nvec)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        if done:
            break

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for s, a, r, s_next, d in minibatch:
                target = r
                if not d:
                    target = (r + gamma * np.amax(target_model.predict(s_next, verbose=0)[0]))
                target_f = model.predict(s, verbose=0)
                action_index = np.ravel_multi_index(a, env.action_space.nvec)
                target_f[0][action_index] = target
                model.train_on_batch(s, target_f)
                
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    if e % 10 == 0:
        target_model.set_weights(model.get_weights())

    episode_duration = time.time() - episode_start_time
    elapsed_time = time.time() - start_time
    episodes_done = e + 1
    episodes_left = total_episodes - episodes_done
    estimated_total_time = (elapsed_time / episodes_done) * total_episodes
    estimated_time_left = estimated_total_time - elapsed_time

    print(f"Episode {e+1}/{total_episodes}, Epsilon: {epsilon:.2}, Total reward: {reward:.2f}, Done: {done}, Duration: {episode_duration:.2f}s")
    print(f"Estimated time left: {estimated_time_left:.2f} seconds, Episodes left: {episodes_left}")

print("Training completed")

# Save the model
model.save('dqn_trading_model.h5')
print("Model saved as 'dqn_trading_model.h5'")
