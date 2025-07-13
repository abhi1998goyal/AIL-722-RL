from env import TreasureHunt
import numpy as np


def val_iteration(env,in_val,gamma=0.9,delta=1e-8):
    iteration = 0
    Q = np.zeros([env.num_states,env.num_actions])
    optimal_policy = np.zeros([env.num_states, env.num_actions])
    V = in_val
    while True:
        print(f"Iteration: {iteration}")
        old_V = np.copy(V) 
        for s in range(env.num_states):
            max_val=-float('inf')
            new_action=-1
            for a in range(env.num_actions):
                new_val=0
               # a_prob = policy[s, a] 
                for s_next in range(env.num_states):
                    transition_prob = env.T[s, a, s_next]
                    reward = env.reward[s_next]
                    new_val += transition_prob * (reward + gamma * V[s_next])

                if new_val>max_val:
                    max_val=new_val
                    new_action=a
            if new_action!=-1:
                V[s]=max_val
                optimal_policy[s]=np.eye(env.num_actions)[new_action]

        iteration += 1
        diff= np.max(np.abs(old_V - V))
        if diff<delta:
            break
    return optimal_policy,V

          

locations = {
    'ship': [(0, 0)], 
    'treasure': [(1, 9), (4, 0)],  
    'pirate': [(4, 7), (8, 5)],    
    'land': [(0, 9), (0, 8), (0, 7), (1, 8), (1, 7), (2,7),
             
            (3, 0),(3, 1),(3, 2), (4, 1), (4, 2), (5, 2)],  
    'fort': [(9, 9)]  
}
env=TreasureHunt(locations)
in_val = np.zeros(env.num_states)
final_policy=val_iteration(env,in_val)[0]
#env.visualize_policy(final_policy,'/home/RL/ass2/plots/p1.2/policy_vis.png')
env.visualize_policy_execution(final_policy,'/home/RL/ass2/plots/p1.2/output_test.gif')