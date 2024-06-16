import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

# Define the base directory containing the Results folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)  # Append the path to system
print("Path:", BASE_DIR)

# List of experiment folders
experiment_folders = [
    r'1M_lr0_01\exp1_2024-06-06_19-52-37',
    r'1M_lr0_001\exp1_2024-06-06_19-41-49',
    r'1M_lr0_0001\exp1_2024-06-06_20-03-08',
    r'1M_lr0_00001\exp1_2024-06-06_20-14-49'
]

# Define the window size for moving average
rewards_window_size = 100  # You can adjust this based on your needs

plt.figure(figsize=(10, 5))

# Specify colors for each plot
colors = ['darkgreen', 'orange', 'blue', 'red']

# Legend labels
legend_labels = [
    "Learning Rate = 0.01",
    "Learning Rate = 0.001",
    "Learning Rate = 0.0001",
    "Learning Rate = 0.00001"
]

# Store the rewards lengths to find the minimum length
rewards_lengths = []

# Load rewards from each experiment and store the lengths
all_rewards = []
for folder in experiment_folders:
    rewards_path = os.path.join(BASE_DIR, folder, 'rewards_log.npy')
    print("rewards_path:", rewards_path)

    if os.path.exists(rewards_path):
        # Load the rewards from the .npy file
        rewards = np.load(rewards_path)
        print("Rewards:", rewards)
        rewards_lengths.append(len(rewards))
        all_rewards.append(rewards)

# Find the minimum length of rewards arrays
min_length = min(rewards_lengths)
print("Minimum rewards length:", min_length)

# Iterate through each experiment folder, truncate rewards, and plot the average rewards
for rewards, color, label in zip(all_rewards, colors, legend_labels):
    truncated_rewards = rewards[:min_length]
    
    # Calculate the moving average
    rewards_moving_avg = pd.Series(truncated_rewards).rolling(window=rewards_window_size, min_periods=1).mean()
    
    # Plot the moving average only with specified color and legend label
    plt.plot(rewards_moving_avg, label=label, color=color)

# Customize the plot
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Average Rewards per Episode")
plt.legend()

# Save the plot
plot_save_path = os.path.join(BASE_DIR, "average_rewards_plot.png")
plt.savefig(plot_save_path)
print(f"Plot saved successfully at {plot_save_path}")
plt.show()  # Optionally, display the plot
