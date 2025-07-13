import matplotlib.pyplot as plt
import imageio
import numpy as np

def get_location_name(index):
    location_dict = {0: 'Red', 1: 'Green', 2: 'Yellow', 3: 'Blue', 4: 'In Taxi'}
    return location_dict.get(index, 'Unknown')

def plot_rewards_vs_episodes(rewards_per_episode,path):
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.savefig(path)
    plt.show()

def epsilon_vs_episodes(epsilon_per_episode,path):
    plt.plot(epsilon_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon per Episode')
    plt.savefig(path)
    plt.show()

def visualize_policy_execution_taxiv3(env, Q, gif_path='policy_execution.gif'):
    state, _ = env.reset()
    #taxi_row, taxi_col, passenger_location, destination = 
    
    #env.decode(state)
    
    print(f"Passenger Pick-up Location: {get_location_name(list(env.unwrapped.decode(state))[2])}")
    print(f"Drop-off Location: {get_location_name(list(env.unwrapped.decode(state))[3])}")

    frames = []
    done = False
    while not done:
        frames.append(env.render())  
        action = np.argmax(Q[state])  
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    imageio.mimsave(gif_path, frames, fps=1)
    # imageio.mimsave(path, pil_images, duration=2)

def plot_avg_rewards_vs_episodes(rewards_per_episode, path, window_size=100):
   
    num_windows = len(rewards_per_episode) // window_size
    avg_rewards = [np.mean(rewards_per_episode[i*window_size:(i+1)*window_size])  for i in range(num_windows)]
   
    episodes = np.arange(1, num_windows + 1) * window_size
 
    plt.plot(episodes, avg_rewards)
    plt.xlabel(f'Episodes (Avg over {window_size})')
    plt.ylabel('Avg Total Reward')
    plt.title(f'Avg Total Reward per {window_size} Episodes')
    plt.savefig(path)
    plt.show()