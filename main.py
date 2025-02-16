# Deep-Reinforcement-Learning using OpenAI Gym & Tensorflow
import gym
import random

environment = gym.make("CartPole-v1", render_mode="human", new_step_api=True)

episodes = 10
for episode in range(1, episodes+1):
    state = environment.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        _, reward, done, _ = environment.step(action)
        score += reward
        environment.render()

    print(f"Episode {episode}, Score: {score}")

environment.close()