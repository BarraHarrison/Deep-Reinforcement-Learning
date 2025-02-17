# Deep-Reinforcement-Learning with TF-Agents 🤖

This repository demonstrates how to train a **Deep Reinforcement Learning (DRL) agent** on the classic **CartPole-v1** task, illustrating the fundamental concepts of value-based methods such as **DQN**. We use **TensorFlow** alongside **TF-Agents** to streamline the setup and training process.

## Project Purpose

- **Learning the Basics of DRL**  
  Explore how an agent can learn to balance a pole on a cart through trial-and-error interactions (Q-learning approach).

- **Hands-On TensorFlow**  
  Gain experience building neural networks (`tf.keras`) and managing RL workflows with TF-Agents.

## Why Switch from OpenAI Gym + Keras-RL to TF-Agents?

- **Tighter Integration**  
  TF-Agents is developed by the TensorFlow team, minimizing compatibility issues and offering end-to-end reinforcement learning components (networks, replay buffers, training loops, etc.).

- **Active Maintenance**  
  TF-Agents aligns with the latest TensorFlow releases, reducing version conflicts often encountered with standalone libraries such as keras-rl2.

- **More Control**  
  A modular design lets you customize each part of the RL pipeline more easily than some “all-in-one” solutions.

## Difficulties Encountered with TensorFlow 😅

- **Version Conflicts**  
  Having multiple or conflicting TF packages in the same environment (`tensorflow`, `tensorflow-macos`, `tensorflow_keras`, and standalone `keras`) caused import and attribute errors.

- **Apple Silicon vs. Intel**  
  On Macs with M1/M2 chips, one must install `tensorflow-macos` and optionally `tensorflow-metal` for GPU acceleration, whereas Intel-based Macs can use standard `tensorflow`.

- **Ensuring `tf.keras` Compatibility**  
  Some RL libraries like keras-rl2 require `tensorflow.keras`, which can break if the environment isn’t cleanly configured.

## Future Improvements 📈

- Try alternative RL algorithms such as PPO, SAC, or A2C.  
- Experiment with different neural network architectures or hyperparameters.  
- Integrate visualization tools (e.g., TensorBoard) to track training metrics in real time.



