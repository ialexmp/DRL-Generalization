from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import torch
import time
import pandas as pd

from Network_MC import ActorNetwork, CriticNetwork
from torch import Tensor
from torch.nn import MSELoss
from typing import Tuple, List
from datetime import timedelta


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)  # Append the path to system

from Custom_Env.custom_envs import CustomFetchPickAndPlaceEnv

EpisodeMemory: Tuple[List[np.array], List[np.array], List[np.array], List[float]]

class Trainer:
    def __init__(
        self,
        exp_name: str,
        experiment_id: str,
        env: CustomFetchPickAndPlaceEnv,
        actor: ActorNetwork,
        critic: CriticNetwork,
        timesteps: int,
        timesteps_per_batch: int,
        max_timesteps_per_episode: int,
        γ: float = 0.99,
        ε: float = 0.2,
        α: float = 3e-4,
        training_cycles_per_batch: int = 5,
        save_every_x_timesteps: int = 200_000,
        device = 'cuda'
    ):  
        self.device = device

        self.exp_name = exp_name
        self.experiment_id = experiment_id
        self.env = env
        self.actor = actor
        self.actor.to(self.device)
        self.critic = critic
        self.critic.to(self.device)

        self.timesteps = timesteps
        self.current_timestep = 0
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.timesteps_per_batch = timesteps_per_batch
        self.save_every_x_timesteps = save_every_x_timesteps
        self.last_save = 0
        self.training_time = None


        # Hyperparameters
        self.γ = γ
        self.ε = ε
        self.α = α
        self.training_cycles_per_batch = training_cycles_per_batch

        # Memory
        self.total_rewards: List[float] = []
        self.terminal_timesteps: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.α)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.α)

        # Initialize some status tracking memory
        self.previous_print_length: int = 0
        self.current_action = "Initializing"
        self.last_save: int = 0

    
    def run_episode(self) -> EpisodeMemory: # type: ignore
        """
        run_episode runs a singular episode and returns the results
        """

        state = self.env.reset()
        observation = state[0]
        observation = torch.tensor(observation).float().to(self.device)

        timesteps = 0
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probabilities: List[float] = []
        rewards: List[float] = []
        while True:
            timesteps += 1

            observations.append(observation.cpu().detach().numpy())
            action_distribution = self.actor(observation)
            action = action_distribution.sample()
            log_probability = action_distribution.log_prob(action).cpu().detach().numpy()
            action = action.cpu().detach().numpy()
            observation, reward, terminated, _, _ = self.env.step(action)
            observation = torch.tensor(observation).float().to(self.device)

            actions.append(action)
            log_probabilities.append(log_probability)
            rewards.append(reward)

            if timesteps >= self.max_timesteps_per_episode:
                terminated = True

            if terminated:
                break

        # Calculate the discounted rewards for this episode
        discounted_rewards: List[float] = self.calculate_discounted_rewards(rewards)

        # Get the terminal reward and record it for status tracking
        self.total_rewards.append(sum(rewards))

        return observations, actions, log_probabilities, discounted_rewards

    def rollout(self) -> EpisodeMemory: # type: ignore
        """
        rollout will perform a rollout of the environment and
        return the memory of the episode with the current
        actor model
        """
        observations: List[np.ndarray] = []
        log_probabilities: List[float] = []
        actions: List[float] = []
        rewards: List[float] = []

        while len(observations) < self.timesteps_per_batch:
            #self.env.render()
            self.current_action = "Rollout"
            obs, chosen_actions, log_probs, rwds = self.run_episode()
            # Combine these arrays into our overall batch
            observations += obs
            actions += chosen_actions
            log_probabilities += log_probs
            rewards += rwds

            # Increment our count of timesteps
            self.current_timestep += len(obs)
            #self.print_status()

        # We need to trim the batch memory to the batch size
        observations = observations[: self.timesteps_per_batch]
        actions = actions[: self.timesteps_per_batch]
        log_probabilities = log_probabilities[: self.timesteps_per_batch]
        rewards = rewards[: self.timesteps_per_batch]

        return observations, actions, log_probabilities, rewards

    def calculate_discounted_rewards(self, rewards: List[float]) -> List[float]:
        """
        calculated_discounted_rewards will calculate the discounted rewards
        of each timestep of an episode given its initial rewards and episode
        length
        """
        discounted_rewards: List[float] = []
        discounted_reward: float = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.γ * discounted_reward)
            # We insert here to append to the front as we're calculating
            # backwards from the end of our episodes
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def calculate_normalized_advantage(
        self, observations: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        calculate_normalized_advantage will calculate the normalized
        advantage of a given batch of episodes
        """
        V = self.critic(observations).detach().squeeze().to(self.device)
        
        # Now we need to calculate our advantage and normalize it
        #advantage = Tensor(np.array(rewards, dtype="float32")) - V
        advantage = rewards - V
        normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        return normalized_advantage

    def training_step(
        self,
        observations: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        normalized_advantage: Tensor,
    ) -> Tuple[float, float]:
        """
        training_step will perform a single epoch of training for the
        actor and critic model

        Returns the loss for each model at the end of the step
        """
        # Get our output for the current actor given our log
        # probabilities
        current_action_distributions = self.actor(observations)
        current_log_probabilities = current_action_distributions.log_prob(actions).to(self.device)

        # We are calculating the ratio as defined by:
        #
        #   π_θ(a_t | s_t)
        #   --------------
        #   π_θ_k(a_t | s_t)
        #
        # ...where our originaly utilized log probabilities
        # are π_θ_k and our current model is creating π_θ. We
        # use the log probabilities and subtract, then raise
        # e to the power of the results to simplify the math
        # for back propagation/gradient descent.
        # Note that we have a log probability matrix of shape
        # (batch size, number of actions), where we're expecting
        # (batch size, 1). We sum our logs as the log(A + B) =
        # log(A) + log(B).
        # log_probabilities = Tensor(np.array(log_probabilities, dtype=np.float32))
        # log_probabilities = torch.sum(log_probabilities, dim=-1)
        # current_log_probabilities = torch.sum(current_log_probabilities, dim=-1)
        
        ratio = torch.exp(current_log_probabilities - log_probabilities).to(self.device)
       

        # Now we calculate the actor loss for this step
        actor_loss = -torch.min(
            ratio * normalized_advantage,
            torch.clamp(ratio, 1 - self.ε, 1 + self.ε) * normalized_advantage,
        ).mean().to(self.device)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Now we do a training step for the critic

        # Calculate what the critic current evaluates our states as.
        # First we have the critic evaluate all observation states,
        # then compare it ot the collected rewards over that time.
        # We will convert our rewards into a known tensor
        V = self.critic(observations).to(self.device)
        reward_tensor = Tensor(rewards).unsqueeze(-1)
        critic_loss = MSELoss()(V, reward_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self):
        """
        train will train the actor and critic models with the
        given training state
        """
        start_time = time.time()
        
        while self.current_timestep <= self.timesteps:
            # Perform a rollout to get our next training
            # batch
            observations, actions, log_probabilities, rewards = self.rollout()
    
            # convert these to numpy arrays and then to tensors
            observations = Tensor(np.array(observations, dtype=np.float32)).to(device=self.device)
            actions = Tensor(np.array(actions, dtype=np.float32)).to(device=self.device)
            log_probabilities = Tensor(np.array(log_probabilities, dtype=np.float32)).to(device=self.device)
            rewards = Tensor(np.array(rewards, dtype=np.float32)).to(device=self.device)

            # Perform our training steps
            for c in range(self.training_cycles_per_batch):
                self.current_action = (
                    f"Training Cycle {c+1}/{self.training_cycles_per_batch}"
                )
                #self.print_status()

                # Calculate our losses
                normalized_advantage = self.calculate_normalized_advantage(
                    observations, rewards
                ).to(self.device)
                
                actor_loss, critic_loss = self.training_step(
                    observations,
                    actions,
                    log_probabilities,
                    rewards,
                    normalized_advantage,
                )
                
                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)

            # Every X timesteps, save our current status
            if self.current_timestep - self.last_save >= self.save_every_x_timesteps:
                self.current_action = "Saving"
                self.print_status()
                self.save()

        print("Training complete!")
        
        end_time = time.time()
        self.training_time = timedelta(seconds=(end_time - start_time))
        print(f"Total training time: {self.training_time} \n")
        
        # Save our results
        self.save()
    
    
    def print_status(self):
        latest_reward = 0.0
        average_reward = 0.0
        best_reward = 0.0
        latest_actor_loss = 0.0
        avg_actor_loss = 0.0
        latest_critic_loss = 0.0
        avg_critic_loss = 0.0
        recent_change = 0.0

        if len(self.total_rewards) > 0:
            latest_reward = self.total_rewards[-1]

            last_n_episodes = 100
            average_reward = np.mean(self.total_rewards[-last_n_episodes:])

            episodes = [
                i
                for i in range(
                    len(self.total_rewards[-last_n_episodes:]),
                    min(last_n_episodes, 0),
                    -1,
                )
            ]
            coefficients = np.polyfit(
                episodes,
                self.total_rewards[-last_n_episodes:],
                1,
            )
            recent_change = coefficients[0]

            best_reward = max(self.total_rewards)

        if len(self.actor_losses) > 0:
            avg_count = 3 * self.timesteps_per_batch
            latest_actor_loss = self.actor_losses[-1]
            avg_actor_loss = np.mean(self.actor_losses[-avg_count:])
            latest_critic_loss = self.critic_losses[-1]
            avg_critic_loss = np.mean(self.critic_losses[-avg_count:])

        msg = f"""
            =========================================
            Timesteps: {self.current_timestep:,} / {self.timesteps:,} ({round((self.current_timestep/self.timesteps)*100, 4)}%)
            Episodes: {len(self.total_rewards):,}
            Currently: {self.current_action}
            Latest Reward: {round(latest_reward)}
            Latest Avg Rewards: {round(average_reward)}
            Recent Change: {round(recent_change, 2)}
            Best Reward: {round(best_reward, 2)}
            Latest Actor Loss: {round(latest_actor_loss, 4)}
            Avg Actor Loss: {round(avg_actor_loss, 4)}
            Latest Critic Loss: {round(latest_critic_loss, 4)}
            Avg Critic Loss: {round(avg_critic_loss, 4)}
            =========================================
        """

        # We print to STDERR as a hack to get around the noisy pybullet
        # environment. Hacky, but effective if paired w/ 1> /dev/null
        print(msg, file=sys.stderr)


    def save(self):
        """
        save will save the models, state, and any additional
        data to a directory named exp_{current_datetime}
        in the /results folder in the same directory as the current file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        directory = os.path.join(current_dir, "Results", f"{self.exp_name}", f"{self.experiment_id}_{current_datetime}")
        os.makedirs(directory, exist_ok=True)
        self.last_save = self.current_timestep

        # ACTOR
        # Model
        actor_path = os.path.join(directory, "Actor.pth")
        torch.save(self.actor.state_dict(), actor_path)
        print(f"· Actor saved successfully in path: {actor_path}")
        # Losses
        actor_losses_path = os.path.join(directory, "actor_losses")
        np.save(actor_losses_path, np.array(self.total_rewards))
        print(f"· Actor Losses saved successfully in path: {actor_losses_path}")
        
        # CRITIC
        # Model
        critic_path = os.path.join(directory, "Critic.pth")
        torch.save(self.critic.state_dict(), critic_path)
        print(f"· Critic saved successfully in path: {critic_path}")
        # Losses
        critic_losses_path = os.path.join(directory, "critic_losses")
        np.save(critic_losses_path, np.array(self.total_rewards))
        print(f"· Critic Losses saved successfully in path: {critic_losses_path}")
        
        # REWARDS
        rewards_path = os.path.join(directory, "rewards_log")
        np.save(rewards_path, np.array(self.total_rewards))
        print(f"· Rewards saved successfully in path: {rewards_path}")

        # GENERATE AND SAVE PLOTS
        self.generate_plots(directory)
        
        # HYPERPARAMETERS
        data = {
            "timesteps": self.timesteps,
            "current_timestep": self.current_timestep,
            "max_timesteps_per_episode": self.max_timesteps_per_episode,
            "timesteps_per_batch": self.timesteps_per_batch,
            "save_every_x_timesteps": self.save_every_x_timesteps,
            "γ": self.γ,
            "ε": self.ε,
            "α": self.α,
            "training_cycles_per_batch": self.training_cycles_per_batch,
            "total_rewards": self.total_rewards,
            "terminal_timesteps": self.terminal_timesteps,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "training_time": self.training_time,
        }
        data_path = os.path.join(directory, "state.data")
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
            print(f"· Hyperparameters and execution time saved successfully in path: {data_path}")
        
    def load(self):
        """
        load will load the models, state, and any additional
        data from the latest experiment directory in the /results
        folder in the same directory as the current file
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, "Results")
        
        # Get the latest experiment directory
        experiment_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        if not experiment_dirs:
            print("No experiment directories found.")
            return
        
        latest_experiment_dir = max(experiment_dirs)
        
        # Load the actor model
        actor_path = os.path.join(results_dir, latest_experiment_dir, "Actor.pth")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
            print(f"· Actor loaded successfully from path: {actor_path}")
        else:
            print(f"Actor model not found in path: {actor_path}")

        # Load the critic model
        critic_path = os.path.join(results_dir, latest_experiment_dir, "Critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"· Critic loaded successfully from path: {critic_path}")
        else:
            print(f"Critic model not found in path: {critic_path}")

        # Load the trainer's state data
        state_data_path = os.path.join(results_dir, latest_experiment_dir, "state.data")
        if os.path.exists(state_data_path):
            with open(state_data_path, "rb") as f:
                data = pickle.load(f)
                self.timesteps = data["timesteps"]
                self.current_timestep = data["current_timestep"]
                self.max_timesteps_per_episode = data["max_timesteps_per_episode"]
                self.timesteps_per_batch = data["timesteps_per_batch"]
                self.save_every_x_timesteps = data["save_every_x_timesteps"]
                self.γ = data["γ"]
                self.ε = data["ε"]
                self.α = data["α"]
                self.training_cycles_per_batch = data["training_cycles_per_batch"]
                self.total_rewards = data["total_rewards"]
                self.terminal_timesteps = data["terminal_timesteps"]
                self.actor_losses = data["actor_losses"]
                self.critic_losses = data["critic_losses"]
            print(f"State data loaded successfully from path: {state_data_path}")
        else:
            print(f"State data not found in path: {state_data_path}")

       
    def generate_plots(self, filepath: str):
        # Define the fraction of the total length to be used as the window size
        fraction = 0.01  # 1% of the total length

        # Calculate the window sizes
        rewards_window_size = max(1, int(len(self.total_rewards) * fraction))
        actor_losses_window_size = max(1, int(len(self.actor_losses) * fraction))
        critic_losses_window_size = max(1, int(len(self.critic_losses) * fraction))

        # Plot total rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.total_rewards, label="Total Rewards", alpha=0.4)
        rewards_moving_avg = pd.Series(self.total_rewards).rolling(window=rewards_window_size, min_periods=1).mean()
        plt.plot(rewards_moving_avg, label=f"Average Rewards (window={rewards_window_size})", color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Rewards per Episode")
        plt.legend()
        rewards_plot_path = os.path.join(filepath, "total_rewards_plot.png")
        plt.savefig(rewards_plot_path)
        print(f"· Total Rewards Plot saved successfully in path: {rewards_plot_path}")
        plt.close()

        # Plot actor losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.actor_losses, label="Actor Losses", color="green", alpha=0.4)
        actor_losses_moving_avg = pd.Series(self.actor_losses).rolling(window=actor_losses_window_size, min_periods=1).mean()
        plt.plot(actor_losses_moving_avg, label=f"Average Actor Losses (window={actor_losses_window_size})", color='darkgreen')
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Actor Losses per Training Step")
        plt.legend()
        actor_losses_plot_path = os.path.join(filepath, "actor_losses_plot.png")
        plt.savefig(actor_losses_plot_path)
        print(f"· Actor Losses Plot saved successfully in path: {actor_losses_plot_path}")
        plt.close()

        # Plot critic losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.critic_losses, label="Critic Losses", color="red", alpha=0.4)
        critic_losses_moving_avg = pd.Series(self.critic_losses).rolling(window=critic_losses_window_size, min_periods=1).mean()
        plt.plot(critic_losses_moving_avg, label=f"Average Critic Losses (window={critic_losses_window_size})", color='darkred')
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Critic Losses per Training Step")
        plt.legend()
        critic_losses_plot_path = os.path.join(filepath, "critic_losses_plot.png")
        plt.savefig(critic_losses_plot_path)
        print(f"· Critic Losses Plot saved successfully in path: {critic_losses_plot_path}")
        plt.close()