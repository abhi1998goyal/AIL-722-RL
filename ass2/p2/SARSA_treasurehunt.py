import numpy as np
import random
from env import TreasureHunt
from plots import plot_rewards_vs_episodes,plot_avg_rewards_vs_episodes
import torch
import time 
from eval import evaluate_policy_thunt

def sarsa(env,num_episodes=70000, alpha=0.3, gamma=0.95, epsilon_start=1,epsilon_end=0.001,decay_factor=0.99,delta=1e-4):
    Q = np.zeros((env.num_states, env.num_actions))
    #Q = np.full((env.num_states, env.num_actions), 1.0) 
    rewards_per_episode = []
   # Q = np.full((env.num_states, env.num_actions), 1.0) 
    #Q = np.random.rand(env.num_states, env.num_actions) * 0.01 
    #previous_Q = np.copy(Q)
    training_start_time = time.time()
    for episode in range(num_episodes):
        if episode % 100 == 0 and episode > 0:
           avg_rew=np.mean(rewards_per_episode[-100:])
           print(f'Episode {episode} Avg reward {avg_rew}')
           if avg_rew>-0.4:
               break
          
      #  print(f'Episode {episode}')
        epsilon = max(epsilon_end,epsilon_start*(pow(decay_factor,episode)))
        num_step=0
        start_state = env.reset()
        total_reward = 0 
        state=start_state
        action = epsilon_greed(start_state,Q,epsilon)
        while  num_step < 100: #(state!=399 and state!=299 and state!=199 and state!=99) and
            num_step+=1
            next_state,reward = env.step(action)
            total_reward += reward
            next_action = epsilon_greed(next_state,Q,epsilon)
            # state=next_state
            Q[state,action]=Q[state,action] + alpha*(reward + gamma*(Q[next_state,next_action]) - Q[state,action])

            state = next_state
            action = next_action
        rewards_per_episode.append(total_reward)
        # Q[399,:]=0
        # Q[299,:]=0
        # Q[199,:]=0
        # Q[99,:]=0
       # print(f'No. of steps {num_step}')
        # Q_del = np.max(np.abs(Q - previous_Q))
        # previous_Q = np.copy(Q)

        # if Q_del < delta:
        #     print(f'Converged after {episode+1} episodes with Q-change: {Q_del}')
        #     break
    train_end_time = time.time()
    train_duration = train_end_time - training_start_time
    print(f"Convergence completed in {train_duration:.2f} seconds")

    return Q,rewards_per_episode

def epsilon_greed(state,Q,epsilon):
    if random.uniform(0,1)<epsilon:
        return random.randint(0, Q.shape[1]- 1)
    else:
        return np.argmax(Q[state])

locations = {
    'ship': [(0, 0)], 
    'treasure': [(1, 9), (4, 0)],  
    'pirate': [(4, 7), (8, 5)],    
    'land': [(0, 9), (0, 8), (0, 7), (1, 8), (1, 7), (2,7),
             (3, 0), (3, 1), (3, 2), (4, 1), (4, 2), (5, 2)],  
    'fort': [(9, 9)]  
}

env = TreasureHunt(locations)

Q,rewards_per_episode = sarsa(env)

Q_tensor = torch.tensor(Q)

#np.save('/home/RL/ass2/saved_policy/TreasureHunt_sarsa_q_table.npy', Q)
          #, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)

torch.save(Q_tensor, '/home/RL/ass2/saved_policy/TreasureHunt_sarsa_q_table.pt')

optimal_policy = np.zeros([env.num_states, env.num_actions])
for state in range(env.num_states):
    optimal_action = np.argmax(Q[state])
    optimal_policy[state] = np.eye(env.num_actions)[optimal_action]

env.visualize_policy(optimal_policy,'/home/RL/ass2/plots/p2.1/sarsa_vis.png')
#env.visualize_policy_execution(optimal_policy,'/home/RL/ass2/plots/p2.1/sarsa_test.gif')

#plot_rewards_vs_episodes(rewards_per_episode,'/home/RL/ass2/plots/p2.1/TreasureHunt_sarsa_reward_vs_episode.png')
#plot_avg_rewards_vs_episodes(rewards_per_episode,'/home/RL/ass2/plots/p2.1/Avg_TreasureHunt_sarsa_reward_vs_episode.png')

#evaluate_policy_thunt(env,'/home/RL/ass2/saved_policy/TreasureHunt_sarsa_q_table.pt','/home/RL/ass2/videos/TreasureHunt_SARSA_episode_')