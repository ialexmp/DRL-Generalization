import gymnasium as gym

def main():
    # Create environment
    env_name = 'Acrobot-v1'
    env = gym.make(env_name, render_mode='human')

    # Get state and action space sizes
    state_size = env.observation_space.shape
    action_size = env.action_space.shape

    print("State size:", state_size)
    print("Action size:", action_size)

    # Reset environment
    obs = env.reset()

    # Render the environment
    env.render()

    # Step through the environment
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        print("Action", action)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        env.render()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
