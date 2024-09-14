import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

ENV_NAME = "CartPole-v1"

GAMMA = 0.99
LEARNING_RATE = 0.0003
EPSILON_CLIP = 0.2 
BATCH_SIZE = 64
EPOCHS = 10
ENTROPY_BETA = 0.01 
EPISODES = 1000
UPDATE_EVERY = 2000

rewards_history = []
moving_average_rewards = []
stability_list = []
MOVING_AVERAGE_WINDOW = 100 


class PPOAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = Adam(learning_rate=LEARNING_RATE)
        self.critic_optimizer = Adam(learning_rate=LEARNING_RATE)
        
        self.memory = []
    
    def build_actor(self):
        """Build the actor model (policy network)"""
        state_input = Input(shape=(self.observation_space,))
        dense_1 = Dense(64, activation='relu')(state_input)
        dense_2 = Dense(64, activation='relu')(dense_1)
        output = Dense(self.action_space, activation='softmax')(dense_2)
        return Model(inputs=state_input, outputs=output)
    
    def build_critic(self):
        """Build the critic model (value function estimator)"""
        state_input = Input(shape=(self.observation_space,))
        dense_1 = Dense(64, activation='relu')(state_input)
        dense_2 = Dense(64, activation='relu')(dense_1)
        output = Dense(1, activation='linear')(dense_2)
        return Model(inputs=state_input, outputs=output)
    
    def remember(self, state, action, reward, next_state, done, prob, value):
        """Store experience for batch update"""
        self.memory.append((state, action, reward, next_state, done, prob, value))
    
    def act(self, state):
        """Choose action according to the current policy (actor network)"""
        state = np.reshape(state, [1, self.observation_space])
        probs = self.actor.predict(state)[0]
        action = np.random.choice(self.action_space, p=probs)
        return action, probs[action]
    
    def ppo_loss(self, old_probs, actions, advantages):
        """Compute PPO loss with clipping"""
        def loss(y_true, y_pred):
            new_probs = tf.reduce_sum(y_true * y_pred, axis=-1)
            old_probs_clipped = tf.clip_by_value(new_probs / old_probs, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP)
            surrogate_loss = advantages * new_probs
            clipped_loss = advantages * old_probs_clipped
            return -tf.reduce_mean(tf.minimum(surrogate_loss, clipped_loss))
        return loss
    
    def compute_advantage(self, rewards, values, next_values, dones):
        """Calculate the advantage estimate used in PPO"""
        advantages = []
        gae = 0
        for reward, value, next_value, done in zip(reversed(rewards), reversed(values), reversed(next_values), reversed(dones)):
            delta = reward + GAMMA * next_value * (1 - done) - value
            gae = delta + GAMMA * 0.95 * gae 
            advantages.append(gae)
        return list(reversed(advantages))
    
    def experience_replay(self):
        """Train the actor and critic using the collected experiences"""
        if len(self.memory) == 0:
            print("No experiences to replay. Skipping update.")
            return

        states, actions, rewards, next_states, dones, old_probs, values = zip(*self.memory)
        
        states = np.array(states).reshape(-1, self.observation_space)
        next_states = np.array(next_states).reshape(-1, self.observation_space)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        values = np.array(values).flatten()
        rewards = np.array(rewards)
        dones = np.array(dones)

        try:
            next_values = self.critic.predict(next_states, verbose=0).flatten()
            advantages = self.compute_advantage(rewards, values, next_values, dones)
            advantages = np.array(advantages)
            returns = advantages + values

            actions_one_hot = np.eye(self.action_space)[actions]
            with tf.GradientTape() as tape:
                new_probs = self.actor(states)
                ppo_loss_value = self.ppo_loss(old_probs, actions_one_hot, advantages)(actions_one_hot, new_probs)
            grads = tape.gradient(ppo_loss_value, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

            with tf.GradientTape() as tape:
                values_pred = self.critic(states)
                critic_loss_value = tf.reduce_mean(tf.square(returns - values_pred))
            grads = tape.gradient(critic_loss_value, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        except Exception as e:
            print(f"Error during experience replay: {e}")
            print(f"States shape: {states.shape}")
            print(f"Next states shape: {next_states.shape}")
            print(f"Actions shape: {actions.shape}")
            print(f"Old probs shape: {old_probs.shape}")
            print(f"Values shape: {values.shape}")
            print(f"Rewards shape: {rewards.shape}")
            print(f"Dones shape: {dones.shape}")

        self.memory = []

def update_metrics(step, run):
    """ Updates rewards, moving averages, and stability measures """
    rewards_history.append(step)

    if len(rewards_history) >= MOVING_AVERAGE_WINDOW:
        moving_average = np.mean(rewards_history[-MOVING_AVERAGE_WINDOW:])
        moving_average_rewards.append(moving_average)

        stability = np.std(rewards_history[-MOVING_AVERAGE_WINDOW:])
        stability_list.append(stability)
        print(f"Run: {run}, Moving Avg Reward: {moving_average:.2f}, Stability: {stability:.2f}")


def plot_results():
    """ Plots the learning curve (rewards per episode), moving average, stability, and average reward """
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(rewards_history, label="Reward per Episode")
    plt.plot(range(MOVING_AVERAGE_WINDOW, len(rewards_history) + 1), moving_average_rewards, label=f"Moving Avg (window={MOVING_AVERAGE_WINDOW})", color="orange")
    plt.title("Learning Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(MOVING_AVERAGE_WINDOW, len(rewards_history) + 1), stability_list, label="Stability (Std. Dev)", color="green")
    plt.title("Stability over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Stability (Standard Deviation)")
    plt.legend()

    plt.subplot(2, 2, 3)
    cumulative_rewards = np.cumsum(rewards_history)
    average_rewards = cumulative_rewards / np.arange(1, len(rewards_history) + 1)
    plt.plot(average_rewards, label="Average Reward", color="purple")
    plt.title("Average Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()

    plt.subplot(2, 2, 4)
    window = MOVING_AVERAGE_WINDOW
    means = moving_average_rewards
    stds = stability_list
    x = range(window, len(rewards_history) + 1)
    plt.plot(x, means, label="Moving Average", color="blue")
    plt.fill_between(x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2, color="blue")
    plt.title("Learning Curve with Confidence Interval")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def cartpole():
    env = gym.make(ENV_NAME, render_mode="human") 
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = PPOAgent(observation_space, action_space)
    
    run = 0
    step_count = 0

    plt.ion() 

    while run < EPISODES:
        run += 1
        state, _ = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        episode_reward = 0
        
        while True:
            step += 1
            step_count += 1
            
            action, prob = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1, observation_space])
            value = agent.critic.predict(state, verbose=0)[0]
            agent.remember(state[0], action, reward, next_state[0], done, prob, value)

            state = next_state
            episode_reward += reward
            
            if done:
                print(f"Run: {run}, Score: {episode_reward}")
                update_metrics(episode_reward, run)
                break
            if step_count % UPDATE_EVERY == 0:
                agent.experience_replay()

        if run % 50 == 0:
            plot_results()
            print(f"Displayed graphs at episode {run}. Close the plot window or press any key to continue training.")
            plt.pause(1) 
            
    plt.show() 

if __name__ == "__main__":
    cartpole()
