import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class ScoreLogger:
    def __init__(self, env_name):
        self.scores = deque(maxlen=100)
        self.env_name = env_name
        self.average_scores = []
        self.episodes = []

    def add_score(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        avg_score = np.mean(self.scores)
        self.average_scores.append(avg_score)
        print(f"Episode {episode}: Score = {score}, Average Score = {avg_score:.2f}")

        if episode % 100 == 0:
            self.plot_scores()

    def plot_scores(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episodes, self.scores, label='Score')
        plt.plot(self.episodes, self.average_scores, label='Average Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title(f'{self.env_name} Training Progress')
        plt.legend()
        plt.savefig(f'{self.env_name}_scores.png')
        plt.close()