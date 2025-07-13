import gymnasium as gym
import numpy as np
import random
from plots import plot_rewards_vs_episodes, visualize_policy_execution_lake, epsilon_vs_episodes, plot_avg_rewards_vs_episodes
import torch
from eval import evaluate_policy_taxi
import time

def monte(env, num_episodes=500000, gamma=0.99, epsilon_start=1, epsilon_end=0.01, decay_factor=0.995):
    Q = np.zeros((env.observation_space.n, env.action_space.n))  
    C = np.zeros((env.observation_space.n, env.action_space.n))  
    policy = np.zeros(env.observation_space.n)
    rewards_per_episode = []
    epsilon_per_episode = []
    training_start_time = time.time()
    
    for episode in range(num_episodes):      
        if episode % 100 == 0 and episode > 0:
            avg_rew = np.mean(rewards_per_episode[-100:])
            print(f'Episode {episode} Avg reward {avg_rew}')
            if avg_rew > 0.8:  
                break
            
        epsilon = max(epsilon_end, epsilon_start * (pow(decay_factor, episode / 50)))
        num_step = 0
        start_state, _ = env.reset() 
        total_reward = 0 
        state = start_state
        
        episode_list = []
        while True:  
            num_step += 1
            action = int(epsilon_greed(state, Q, epsilon))
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            episode_list.append((state, action, reward))
            state = next_state
            if terminated or truncated:  
                break

        rewards_per_episode.append(total_reward)
        epsilon_per_episode.append(epsilon)

        G = 0
        W = 1
        for (state, action, reward) in reversed(episode_list):
            G = gamma * G + reward
            C[state, action] += W
            Q[state, action] += (W / C[state, action]) * (G - Q[state, action])
            policy[state] = np.argmax(Q[state])
            if action != policy[state]:
                break 
            W *= 1/(1 - epsilon + (epsilon / env.action_space.n))

    train_end_time = time.time()
    train_duration = train_end_time - training_start_time
    print(f"Convergence completed in {train_duration:.2f} seconds")
    
    return Q, rewards_per_episode, epsilon_per_episode

def epsilon_greed(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.random.choice(Q.shape[1])
    else:
        return np.argmax(Q[state])

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")

Q, rewards_per_episode, epsilon_per_episode = monte(env)

Q_tensor = torch.tensor(Q)
#np.save('/home/RL/ass2/saved_policy/qlearn_sarsa_q_table.npy', Q)

torch.save(Q_tensor, '/home/RL/ass3/saved_policy/mc_frozenlake_q_table.pt')

#plot_rewards_vs_episodes(rewards_per_episode, '/home/RL/ass2/plots/p2.2/Taxi_qlearn_reward_vs_episode.png')
plot_avg_rewards_vs_episodes(rewards_per_episode, '/home/RL/ass3/plots/p1/Avg_frozenlake_mc_reward_vs_episode.png')
visualize_policy_execution_lake(env, Q, gif_path='/home/RL/ass3/plots/p1/frozenlake_mc_policy_execution.gif')
epsilon_vs_episodes(epsilon_per_episode, '/home/RL/ass3/plots/p1/frozenlake_mc_epsilon_vs_episode.png')

#evaluate_policy_taxi(env,'/home/RL/ass3/saved_policy/mc_taxi_q_table.pt','/home/RL/ass3/videos/Taxi_mc_episode_')
env.close()
