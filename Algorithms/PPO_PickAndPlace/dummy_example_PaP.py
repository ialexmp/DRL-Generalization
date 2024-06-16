import gymnasium as gym

def main():
    # Create environment
    env_name = 'FetchPickAndPlaceDense-v2'
    env = gym.make(env_name, render_mode='human')

    # Get state and action space sizes
    state_size = env.observation_space["observation"].shape[0] + env.observation_space["achieved_goal"].shape[0] + env.observation_space["desired_goal"].shape[0]
    action_size = env.action_space.shape[0]

    print("State size:", state_size)
    print("Action size:", action_size)


    # Reset environment
    obs = env.reset()

    # Render the environment
    #env.render(mode='human')
    env.render()
    # Step through the environment
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, info, _ = env.step(action)
        env.render()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
