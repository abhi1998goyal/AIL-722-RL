import gymnasium as gym
import numpy as np
import random
from plots import plot_rewards_vs_episodes  ,visualize_policy_execution_taxiv3,plot_avg_rewards_vs_episodes,epsilon_vs_episodes
import torch 
import time
from eval import evaluate_policy_taxi
def sarsa(env, num_episodes=50000, alpha=0.05, gamma=0.95, epsilon_start=1,epsilon_end=0.001,decay_factor=0.99):
    Q = np.zeros((env.observation_space.n, env.action_space.n))  
    rewards_per_episode = []
    epsilon_per_episode=[]
    training_start_time = time.time()
    for episode in range(num_episodes):
        #print(f'Episode {episode}')
        if episode % 100 == 0 and episode > 0:
           avg_rew=np.mean(rewards_per_episode[-100:])
           print(f'Episode {episode} Avg reward {avg_rew}')
           if avg_rew>8:
               break
        num_step = 0
        epsilon = max(epsilon_end,epsilon_start*(pow(decay_factor,episode)))
        epsilon_per_episode.append(epsilon)
        start_state,_ = env.reset() 
        total_reward = 0 
        state = start_state
        action = int(epsilon_greed(state, Q, epsilon))
        #.item()
        
        while True:  
            num_step += 1
            next_state, reward, terminated, truncated,info = env.step(action)
            total_reward += reward
            next_action = epsilon_greed(next_state, Q, epsilon)
            
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
            
            if terminated or truncated:  
                break

        rewards_per_episode.append(total_reward)
        #print(f'No. of steps {num_step}')
    train_end_time = time.time()
    train_duration = train_end_time - training_start_time
    print(f"Convergence completed in {train_duration:.2f} seconds")
    
    return Q, rewards_per_episode,epsilon_per_episode

def epsilon_greed(state, Q, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, Q.shape[1] - 1)  
    else:
        return np.argmax(Q[state])

env = gym.make("Taxi-v3", render_mode="rgb_array")

Q, rewards_per_episode,epsilon_per_episode = sarsa(env)
Q_tensor = torch.tensor(Q)

#np.save('/home/RL/ass2/saved_policy/Taxi_sarsa_q_table.npy', Q)
torch.save(Q_tensor, '/home/RL/ass2/saved_policy/Taxi_sarsa_q_table.pt')

#plot_rewards_vs_episodes(rewards_per_episode, '/home/RL/ass2/plots/p2.2/Taxi_sarsa_reward_vs_episode.png')
plot_avg_rewards_vs_episodes(rewards_per_episode, '/home/RL/ass2/plots/p2.2/Avg_Taxi_sarsa_reward_vs_episode.png')
visualize_policy_execution_taxiv3(env, Q, gif_path='/home/RL/ass2/plots/p2.2/Taxi_sarsa_policy_execution_test.gif')
epsilon_vs_episodes(epsilon_per_episode,'/home/RL/ass2/plots/p2.1/Taxiv3_sarsa_epsilon_vs_episode_test.png')
evaluate_policy_taxi(env,'/home/RL/ass2/saved_policy/Taxi_sarsa_q_table.pt','/home/RL/ass2/videos/Taxi_SARSA_episode_')
env.close()