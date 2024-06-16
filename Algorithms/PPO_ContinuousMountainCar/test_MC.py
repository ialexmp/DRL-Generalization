import sys
import gymnasium as gym
import numpy as np
import torch
import os
import pickle

from Network_MC import ActorNetwork, CriticNetwork

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)  # Append the path to system

from Custom_Env.custom_envs import CustomFetchPickAndPlaceEnv

def load_experiment(experiment_name, input_dims, n_actions, device):
    """
    Load the specified experiment data, including models and state data.

    Args:
    - experiment_name (str): Name of the experiment directory.
    - input_dims (int): Dimensionality of the input state space.
    - n_actions (int): Dimensionality of the action space.
    - device (torch.device): Device to load the models on.

    Returns:
    - actor (ActorNetwork): Loaded actor network.
    - critic (CriticNetwork): Loaded critic network.
    - data (dict): Loaded state data.
    - actor_losses (numpy.ndarray): Loaded actor loss data.
    - critic_losses (numpy.ndarray): Loaded critic loss data.
    - rewards_log (numpy.ndarray): Loaded rewards log data.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_path = os.path.join(current_dir, "Results", experiment_name)
    
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f" - Experiment directory not found: {experiment_path}")
    
    print("\nLoading Experiment Data")
    print("="*50)
    
    # Load the actor model
    actor_path = os.path.join(experiment_path, "Actor.pth")
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f" - Actor model not found in path: {actor_path}")
    actor = ActorNetwork(input_dims, n_actions).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    print(f"· Actor loaded successfully from path: {actor_path}")

    # Load the critic model
    critic_path = os.path.join(experiment_path, "Critic.pth")
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f" - Critic model not found in path: {critic_path}")
    critic = CriticNetwork(input_dims).to(device)
    critic.load_state_dict(torch.load(critic_path, map_location=device))
    critic.eval()
    print(f"· Critic loaded successfully from path: {critic_path}")

    # Load the trainer's state data
    state_data_path = os.path.join(experiment_path, "state.data")
    if not os.path.exists(state_data_path):
        raise FileNotFoundError(f" - State data not found in path: {state_data_path}")
    with open(state_data_path, "rb") as f:
        data = pickle.load(f)
    print(f"· State data loaded successfully from path: {state_data_path}")
    
    # Load the actor loss data
    actor_losses_path = os.path.join(experiment_path, "actor_losses.npy")
    if not os.path.exists(actor_losses_path):
        raise FileNotFoundError(f" - Actor losses not found in path: {actor_losses_path}")
    actor_losses = np.load(actor_losses_path)
    print(f"· Actor losses loaded successfully from path: {actor_losses_path}")

    # Load the critic loss data
    critic_losses_path = os.path.join(experiment_path, "critic_losses.npy")
    if not os.path.exists(critic_losses_path):
        raise FileNotFoundError(f" - Critic losses not found in path: {critic_losses_path}")
    critic_losses = np.load(critic_losses_path)
    print(f"· Critic losses loaded successfully from path: {critic_losses_path}")

    # Load the rewards log data
    rewards_log_path = os.path.join(experiment_path, "rewards_log.npy")
    if not os.path.exists(rewards_log_path):
        raise FileNotFoundError(f" - Rewards log not found in path: {rewards_log_path}")
    rewards_log = np.load(rewards_log_path)
    print(f"· Rewards log loaded successfully from path: {rewards_log_path}")
    
    print("="*50)
    return actor, critic, data, actor_losses, critic_losses, rewards_log

def print_hyperparameters(data):
    """
    Print the specific hyperparameters and other relevant information from the state data.

    Args:
    - data (dict): Loaded state data.
    """
    hyperparameters = [
        "timesteps",
        "max_timesteps_per_episode",
        "timesteps_per_batch",
        "γ",
        "ε",
        "α",
        "training_cycles_per_batch",
        "training_time"
    ]
    
    print("\nHyperparameters and Training State:")
    print("="*50)
    for key in hyperparameters:
        value = data.get(key, 'Not available')
        print(f" · {key:<30}: {value}")
    print("="*50)

def test_actor_model(actor, env, num_episodes=5):
    """
    Test the loaded actor model by interacting with the environment.

    Args:
    - actor (ActorNetwork): Loaded actor network.
    - env (gym.Env): Gym environment.
    - num_episodes (int): Number of episodes to run for testing.

    Returns:
    - avg_reward (float): Average total reward over the episodes.
    """
    total_rewards = []
    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0.0
        current_step = 0

        if isinstance(observation, tuple):
            observation = observation[0]

        while not done:
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
          
            with torch.no_grad():
                action_distribution = actor(state)
                action = action_distribution.mean.detach().cpu().numpy().squeeze(axis=0)       
          
            observation, reward, done, _, _ = env.step(action)

            total_reward += reward
            current_step += 1
            print("current_step: ", current_step)
        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    return avg_reward

def main(experiment_name):
    
    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device =", device)
    
    # Create environment
    env_name = 'MountainCarContinuous-v0'
    render = True
    if render: 
        env = gym.make(env_name,render_mode='human')
    else: 
        env = gym.make(env_name)

    # Get state and action space sizes
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    try:
        actor, critic, data, actor_losses, critic_losses, rewards_log = load_experiment(experiment_name, observation_size, action_size, device)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    # Print hyperparameters and state data
    print_hyperparameters(data)
    
    # Test actor model
    num_episodes = data.get('num_episodes', 5)
    avg_reward = test_actor_model(actor, env, num_episodes=num_episodes)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <experiment_path from Results folder>")
        sys.exit(1)

    experiment_name = sys.argv[1]
    main(experiment_name)