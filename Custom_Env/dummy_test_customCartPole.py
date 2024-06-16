import gym
from custom_envs import (
    CustomMountainCarGoalPositionEnv,
    CustomMountainCarPowerEnv,
    CustomMountainCarStartPositionEnv,
    CustomMountainCarFrictionEnv,
    CustomMountainCarRandomPerturbationsEnv,
)

def main():
    # List of custom environment names
    custom_envs = [
        'CustomMountainCarGoalPosition',
        'CustomMountainCarPower',
        'CustomMountainCarStartPosition',
        'CustomMountainCarFriction',
        'CustomMountainCarRandomPerturbations'
    ]

    for env_name in custom_envs:
        print(f"Testing environment: {env_name}")
        
        # Create environment
        #env = gym.make(env_name, render_mode='human')
        env = gym.make("CustomMountainCarRandomPerturbations", render_mode='human')

        # Get state and action space sizes
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        print("State size:", state_size)
        print("Action size:", action_size)

        # Reset environment
        obs, info = env.reset()

        # Render the environment
        env.render()

        # Step through the environment
        done = False
        step_count = 0  # Initialize step count

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, done, _, info = env.step(action)

            step_count += 1  # Increment step count
            
            # Reset environment every 50 steps
            if step_count % 50 == 0:
                obs, info = env.reset()
                step_count = 0  # Reset step count

        # Close the environment
        env.close()

if __name__ == "__main__":
    main()