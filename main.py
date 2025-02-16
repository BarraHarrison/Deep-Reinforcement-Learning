# Deep-Reinforcement-Learning using TF-Agent
import numpy as np
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import policy_saver
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics

env_name = "CartPole-v1"
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

fc_layer_params = (24, 24)
q_net = q_network.QNetwork(
    input_tensor_spec=train_env.observation_spec(),
    action_spec=train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step
)
agent.initialize()


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=50000
)
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2
).prefetch(3)
iterator = iter(dataset)



collect_driver = dynamic_step_driver.DynamicStepDriver(
    env=train_env,
    policy=agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=1
)

num_iterations = 50000
agent.train = common.function(agent.train)
for _ in range(num_iterations):
    collect_driver.run()
    experience, _ = next(iterator)
    agent.train(experience)

num_eval_episodes = 10
avg_return = 0.0
for _ in range(num_eval_episodes):
    time_step = eval_env.reset()
    episode_return = 0.0
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        episode_return += time_step.reward
    avg_return += episode_return / num_eval_episodes

print(f"Average Return over {num_eval_episodes} episodes", avg_return.numpy()[0])
