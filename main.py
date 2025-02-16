# Deep-Reinforcement-Learning using OpenAI Gym & Tensorflow
import gym
import random

environment = gym.make("Cartpole-v1", render_mode="human")

episodes = 10
for episode in range(1, episodes+1):
    state = environment.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        _, = environment.step(action)