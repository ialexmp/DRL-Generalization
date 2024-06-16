import sys
import gymnasium as gym
import numpy as np
import torch
import yaml
import os

from trainer_CartPole import Trainer
from Network_CartPole import ActorNetwork,CriticNetwork

def read_config_file(file_name):
    current_file_path = os.path.abspath(__file__)

    # Get the parent directory of the current file
    parent_dir = os.path.dirname(current_file_path)

    # Construct the path to the YAML file relative to the parent directory
    experiments_path = os.path.join(parent_dir, "Experiments_Config", file_name)

    try:
        # Read experiment configurations from YAML file
        with open(experiments_path, 'r') as file:
            experiments = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{experiments_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file '{experiments_path}': {e}")
        return None
    
    return experiments

def run_experiment(exp_name, experiment_id, config, device):
    
    # Create environment
    env_name = 'CartPole-v1'
    
    render = False
    if render:
        env = gym.make(env_name, render_mode='human')
    else: 
        env = gym.make(env_name)
    
    # Get state and action space sizes
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize actor and critic networks
    actor = ActorNetwork(input_dims=observation_size, n_actions=action_size).to(device)
    critic = CriticNetwork(input_dims=observation_size).to(device)
    
    # Initialize trainer
    trainer = Trainer(
        exp_name,
        experiment_id,
        env,
        actor,
        critic,
        timesteps=config['timesteps'],
        timesteps_per_batch=config['timesteps_per_batch'],
        max_timesteps_per_episode=config['max_timesteps_per_episode'],
        γ=config['gamma'],
        ε=config['epsilon'],
        α=config['alpha'],
        device=device
    )

    # Run experiment
    trainer.train()

def main(experiment_config_name):
    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device =", device)
        
    # Remove the ".yaml" extension 
    exp_name = experiment_config_name.replace(".yaml", "")

    print(f"Experiment File Selected: {experiment_config_name}\n")

    experiments = read_config_file(experiment_config_name)

    # Check if experiments are loaded successfully
    if experiments is not None:
        
        print("Experiments loaded successfully.\n")
        # Run experiments sequentially
        for experiment_id, config in experiments.items():
            print(f"Experiment ID: {experiment_id}")
            print(f"Experiment Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            print("\nRunning Experiment...\n")
            run_experiment(exp_name, experiment_id, config, device)
            print(f"Experiment {experiment_id} completed.\n")
    else:
        print("Failed to load experiments. Exiting program.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Warning: No experiment_config_name provided.")
        print("Usage: python script.py <experiment_config_name>")
        sys.exit(1)
    
    experiment_config_name = sys.argv[1]
    if not experiment_config_name.endswith(".yaml"):
        print("Error: The configuration file must have a .yaml extension.")
        sys.exit(1)
    
    try:
        main(experiment_config_name)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)