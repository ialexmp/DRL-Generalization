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
    r'1M_lr0_01\exp1_2024-06-07_10-53-04',
    r'1M_lr0_001\exp1_2024-06-07_10-39-06',
    r'1M_lr0_0001\exp1_2024-06-07_10-19-44',
    r'1M_lr0_00001\exp1_2024-06-07_10-06-00'
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

# Iterate through each experiment folder and calculate the average rewards
for folder, color, label in zip(experiment_folders, colors, legend_labels):
    rewards_path = os.path.join(BASE_DIR, folder, 'rewards_log.npy')
    print("rewards_path:", rewards_path)

    if os.path.exists(rewards_path):
        # Load the rewards from the .npy file
        rewards = np.load(rewards_path)
        print("Rewards:", rewards)
        
        # Calculate the moving average
        rewards_moving_avg = pd.Series(rewards).rolling(window=rewards_window_size, min_periods=1).mean()
        
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
