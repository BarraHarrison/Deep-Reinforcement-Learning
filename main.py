# Deep-Reinforcement-Learning using OpenAI Gym & Tensorflow
import gym
import random
import numpy as np

import tensorflow as tf

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

environment = gym.make("CartPole-v1")
states = environment.observation_space.shape[0]
actions = environment.action_space.n

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(1, states)))
model.add(tf.keras.layers.Dense(24, activation="relu"))
model.add(tf.keras.layers.Dense(24, activation="relu"))
model.add(tf.keras.layers.Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

agent.compile(Adam(lr=0.001, metrics=["mae"]))
agent.fit(environment, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(environment, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

# episodes = 10
# for episode in range(1, episodes+1):
#     state = environment.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0, 1])
#         observation, reward, terminated, truncated, info = environment.step(action)
#         done = terminated or truncated
#         score += reward
        

#     print(f"Episode {episode}, Score: {score}")

