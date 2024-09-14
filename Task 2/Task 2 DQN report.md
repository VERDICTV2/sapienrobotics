# Deep Q-Network (DQN) Implementation for CartPole

## 1. Implementation Details

### 1.1 Environment
- **Task**: CartPole-v1 from OpenAI Gym
- **Objective**: Balance a pole on a cart by moving the cart left or right

### 1.2 DQN Architecture
- **Neural Network Structure**:
  - Input Layer: State space dimension (4 for CartPole)
  - Hidden Layer 1: 24 neurons with ReLU activation
  - Hidden Layer 2: 24 neurons with ReLU activation
  - Output Layer: Action space dimension (2 for CartPole) with linear activation

### 1.3 Hyperparameters
- Discount Factor (GAMMA): 0.95
- Learning Rate: 0.001
- Memory Size: 1,000,000
- Batch Size: 20
- Exploration:
  - Initial Rate: 1.0 (100% random actions)
  - Minimum Rate: 0.01
  - Decay Rate: 0.995 per episode

### 1.4 Key Components
1. **DQNSolver Class**:
   - Initializes the neural network
   - Implements experience replay
   - Handles action selection (epsilon-greedy policy)

2. **Memory**: Uses a deque for efficient experience storage and sampling

3. **Training Loop**:
   - Runs episodes until manually stopped
   - Updates the network after each action using experience replay

4. **Metrics Tracking**:
   - Rewards per episode
   - Moving average rewards (window size: 100 episodes)
   - Stability (standard deviation of rewards)
   - Average rewards per episode

## 2. Results

The implementation includes visual representations of the learning process:

1. **Learning Curve**: 
   - Shows raw rewards per episode
   - Displays moving average rewards
  
2. **Average Reward per Episode**:
   - Tracks the overall learning progress

![Task 2 DQN](https://github.com/user-attachments/assets/cc2f0aad-97ac-446c-b96a-4a26e3ed8809)

These plots are updated every 50 episodes, providing real-time insights into the agent's performance.

## 3. Discussion

### 3.1 Strengths

1. **Adaptive Exploration**: The epsilon-greedy strategy with decay allows for balanced exploration and exploitation.

2. **Experience Replay**: Enhances learning stability and efficiency by breaking the correlation between consecutive samples.

3. **Comprehensive Metrics**: The inclusion of moving averages and stability measures provides a nuanced view of the agent's performance.

4. **Visualization**: Real-time plotting aids in understanding the learning dynamics.

5. **Flexibility**: The code structure allows for easy modification of hyperparameters and network architecture.

### 3.2 Weaknesses

1. **Fixed Network Architecture**: The current implementation uses a fixed network size. Implementing a more flexible architecture could improve adaptability to different environments.

2. **Lack of Target Network**: Implementing a separate target network could enhance stability, as per more recent DQN variants.

3. **Limited Hyperparameter Tuning**: The current setup doesn't include systematic hyperparameter optimization, which could potentially improve performance.

4. **No Early Stopping**: The training continues indefinitely. Implementing a stopping criterion based on performance could be beneficial.

5. **Basic Reward Structure**: The reward function is simple (-reward for terminal states). A more sophisticated reward shaping could potentially speed up learning.

6. **Lack of Prioritized Experience Replay**: Implementing prioritized replay could improve learning efficiency by focusing on more important transitions.

## 4. Conclusion

This DQN implementation demonstrates effective learning in the CartPole environment, showcasing key concepts of deep reinforcement learning. The comprehensive metrics and visualizations provide valuable insights into the learning process. While there's room for improvement, particularly in terms of advanced DQN techniques and hyperparameter optimization, the current implementation serves as a solid foundation for further experimentation and learning in the field of reinforcement learning.
