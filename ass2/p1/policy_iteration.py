from env import TreasureHunt
import numpy as np

def policy_eval(env,policy,gamma=0.9,delta=1e-6):
    V = np.zeros(env.num_states)
    while True:
        diff=0
        for s in range(env.num_states):
            v = V[s]  
            new_val = 0
            for a in range(env.num_actions):
                action_prob = policy[s, a]  
                for s_new in range(env.num_states):
                    transition_prob = env.T[s, a, s_new]
                    reward = env.reward[s_new]
                    new_val += action_prob * transition_prob * (reward + gamma * V[s_new])
            V[s]=new_val
            diff = max(diff,abs(v-V[s]))
        if diff<delta:
            break
    return V


def policy_iteration(env,policy,gamma=0.9,delta=1e-6):
    iteration = 0
    while True:
        V = policy_eval(env,policy,gamma,delta)
        print(f"Iteration: {iteration}")
        conv=True
        for s in range(env.num_states):
            #ship_loc , treasure_loc = env.locations_from_state(s)
            old_action = np.argmax(policy[s])
            new_action =old_action
            max_val=-float('inf')
            
            for a in range(env.num_actions):
                new_val=0
               # a_prob = policy[s, a] 
                for s_next in range(env.num_states):
                    transition_prob = env.T[s, a, s_next]
                    reward = env.reward[s_next]
                    new_val += transition_prob * (reward + gamma * V[s_next])

                print(f"State {s}, Action {a}, new_val: {new_val}")
                if new_val>max_val:
                    max_val=new_val
                    new_action= a
            if new_action!=old_action:
                conv=False
                policy[s] = np.eye(env.num_actions)[new_action]

        iteration += 1
        if conv==True:
            break
    return policy,V

          

locations = {
    'ship': [(0, 0)], 
    'treasure': [(1, 9), (4, 0)],  
    'pirate': [(4, 7), (8, 5)],    
    'land': [(0, 9), (0, 8), (0, 7), (1, 8), (1, 7), (2,7),
             
            (3, 0),(3, 1),(3, 2), (4, 1), (4, 2), (5, 2)],  
    'fort': [(9, 9)]  
}
env=TreasureHunt(locations)
policy = np.ones([env.num_states, env.num_actions]) / env.num_actions 
final_policy=policy_iteration(env,policy)[0]
#env.visualize_policy(final_policy,'/home/RL/ass2/plots/p1.1/policy_vis.png')
env.visualize_policy_execution(final_policy,'/home/RL/ass2/plots/p1.1/output_test.gif')