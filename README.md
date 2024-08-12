# RL Trading Agent

## Introduction
This project employs a Reinforcement Learning (RL) approach using a Deep Q-Network (DQN) to create an automated trading agent. It is designed to trade high-profile stocks, known as the "Magnificent Seven," including Apple, Microsoft, Google, Amazon, NVIDIA, Meta, and Tesla, aiming to maximize returns and minimize risks.

## Getting Started

### Prerequisites
Ensure you have the necessary libraries installed:
- numpy
- pandas
- tensorflow
- gym
- alpha_vantage

### Setup
ALL OF THE CODE ARE IN SRC FOLDER 

To train the model with historical data:
RUN 
python train_model.py

FOR BACK TESTING OF THE MODEL
RUN
python backtest.py

To predict trading actions based on the latest available data:
RUN
python predict_market.py
