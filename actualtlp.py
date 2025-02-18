import networkx as nx
import pandas as pd
import numpy as np
import gym
from gym import spaces
import random
from stable_baselines3 import DQN
import matplotlib.pyplot as plt


# Load temporal dataset (Example: timestamped edges)
df = pd.read_csv("email-Eu-core-temporal.txt", sep=" ", names=["node1", "node2", "timestamp"])

# Sort by time to maintain temporal order
df = df.sort_values("timestamp")

# Convert into time-based snapshots
time_splits = np.array_split(df, 10)  # Create 10-time slices

# Create dynamic graphs for each time slice
graphs = [nx.from_pandas_edgelist(t, "node1", "node2") for t in time_splits]


class TemporalLinkPredictionEnv(gym.Env):
    def __init__(self, graphs):
        super(TemporalLinkPredictionEnv, self).__init__()

        self.graphs = graphs
        self.current_t = 0
        self.current_graph = graphs[self.current_t]

        # Define action & observation space
        self.node_list = list(self.current_graph.nodes)
        self.action_space = spaces.Discrete(len(self.node_list) ** 2)  # Every pair of nodes
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.node_list), len(self.node_list)), dtype=np.float32)

    def reset(self):
        """Reset environment to the initial state"""
        self.current_t = 0
        self.current_graph = self.graphs[self.current_t]
        return self.get_observation()

    def step(self, action):
        """Take a step in the environment"""
        i, j = divmod(action, len(self.node_list))  # Decode action (node pair)
        node1, node2 = self.node_list[i], self.node_list[j]

        reward = 0
        next_graph = self.graphs[self.current_t + 1] if self.current_t + 1 < len(self.graphs) else self.current_graph

        if next_graph.has_edge(node1, node2):
            reward = 1  # Reward for correct link prediction
        else:
            reward = -1  # Penalty for incorrect prediction

        self.current_t = min(self.current_t + 1, len(self.graphs) - 1)
        done = self.current_t == len(self.graphs) - 1
        return self.get_observation(), reward, done, {}

    def get_observation(self):
        """Get adjacency matrix as observation"""
        return nx.to_numpy_array(self.current_graph)

    def render(self, mode="human"):
        pass

# Create environment
env = TemporalLinkPredictionEnv(graphs)

# Train RL agent
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Save trained model
model.save("temporal_link_prediction_dqn")


# Load trained model
model = DQN.load("temporal_link_prediction_dqn")

# Test the model
obs = env.reset()
total_rewards = 0
for _ in range(len(graphs) - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_rewards += reward
    if done:
        break

print("Total Reward (Prediction Accuracy Score):", total_rewards)

pred_graph = nx.Graph()
for _ in range(100):  # Predict 100 times
    action, _ = model.predict(obs)
    i, j = divmod(action, len(env.node_list))
    node1, node2 = env.node_list[i], env.node_list[j]
    pred_graph.add_edge(node1, node2)

plt.figure(figsize=(10, 5))
nx.draw(pred_graph, with_labels=True, node_color="lightblue", edge_color="red")
plt.title("Predicted Links in Next Time Step")
plt.show()
