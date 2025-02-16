# Deep-Reinforcement-Learning using OpenAI Gym & Tensorflow
import gym
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

environment = gym.make("CartPole-v1", render_mode="human")

episodes = 10
for episode in range(1, episodes+1):
    state = environment.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        observation, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        score += reward
        

    print(f"Episode {episode}, Score: {score}")

