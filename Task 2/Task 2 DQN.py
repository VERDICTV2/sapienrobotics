import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from score_logger import ScoreLogger
import matplotlib.pyplot as plt

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

# Track performance metrics
rewards_history = []
moving_average_rewards = []
average_rewards_per_episode = []  # New list to track average rewards
stability_list = []
MOVING_AVERAGE_WINDOW = 100  # Window size for moving average

class DQNSolver:
    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

def update_metrics(step, run):
    """ Updates rewards, moving averages, and stability measures """
    rewards_history.append(step)
    average_reward = np.mean(rewards_history)  # Calculate average reward
    average_rewards_per_episode.append(average_reward)  # Track average rewards

    # Calculate moving average over the last N episodes
    if len(rewards_history) >= MOVING_AVERAGE_WINDOW:
        moving_average = np.mean(rewards_history[-MOVING_AVERAGE_WINDOW:])
        moving_average_rewards.append(moving_average)

        # Calculate stability (standard deviation) over the last N episodes
        stability = np.std(rewards_history[-MOVING_AVERAGE_WINDOW:])
        stability_list.append(stability)
        print(f"Run: {run}, Moving Avg Reward: {moving_average:.2f}, Stability: {stability:.2f}, Avg Reward: {average_reward:.2f}")

def plot_results():
    """ Plots the learning curve (rewards per episode), moving average, and stability """
    plt.figure(figsize=(18, 6))

    # Plot raw rewards and moving average
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history, label="Reward per Episode")
    if moving_average_rewards:
        plt.plot(range(MOVING_AVERAGE_WINDOW, len(rewards_history) + 1), moving_average_rewards, label=f"Moving Avg (window={MOVING_AVERAGE_WINDOW})", color="orange")
    plt.title("Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()

    # Plot stability (standard deviation of last N rewards)
    plt.subplot(1, 3, 2)
    if stability_list:
        plt.plot(range(MOVING_AVERAGE_WINDOW, len(rewards_history) + 1), stability_list, label="Stability (Std. Dev)", color="green")
    plt.title("Stability over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Stability (Standard Deviation)")
    plt.legend()

    # Plot average rewards per episode
    plt.subplot(1, 3, 3)
    if average_rewards_per_episode:
        plt.plot(range(1, len(average_rewards_per_episode) + 1), average_rewards_per_episode, label="Average Reward per Episode", color="red")
    plt.title("Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.tight_layout()
    plt.show()

def cartpole():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0

    while True:
        run += 1
        state, _ = env.reset()  # Unpack the tuple returned by env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminated, truncated, info = env.step(action)  # Unpack the new return values
            done = terminated or truncated  # Combine terminated and truncated into a single 'done' flag
            reward = reward if not done else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next
            if done:
                print(f"Run: {run}, Exploration Rate: {dqn_solver.exploration_rate:.4f}, Score: {step}")
                score_logger.add_score(step, run)

                # Update metrics after each episode
                update_metrics(step, run)
                break
            dqn_solver.experience_replay()

        # Plot results after every 10 episodes
        if run % 10 == 0:
            plot_results()

    env.close()

if __name__ == "__main__":
    cartpole()
