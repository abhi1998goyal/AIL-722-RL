from env import TreasureHunt, TreasureHuntExtreme
import numpy as np
import pdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random 
import os 
import wandb
wandb.init(mode="online", project="a3_th_pg", name='with_reward_norm_10_10')

class PolicyNetwork(nn.Module):

    def __init__(self, n):
        super(PolicyNetwork, self).__init__()
        ## YOUR CODE HERE ##

    def forward(self, x):
        ## YOUR CODE HERE ##

class PolicyGradientAgent:

    def __init__(self, env, demo_env, cache_size = 100000):
        
        self.env = env
        self.n = self.env.n 
         
        self.dataset = CacheData(cache_size)
        self.pnetwork = PolicyNetwork(n = self.n)
        self.optimizer = torch.optim.Adam(self.pnetwork.parameters(), lr = 0.0001)
        self.df_value = 0.95
        self.demo_env = demo_env
        self.episode_len = 25
        self.df = self.df_value**torch.arange(self.episode_len)

    def get_policy(self, states):
        states = torch.tensor(states).float()
        qsa = self.pnetwork(states)
        qsa = qsa.argmax(dim = -1)
        qsa = torch.nn.functional.one_hot(qsa)
        policy = qsa.numpy()
        return policy
        

    def visualize_policy(self, itr, path = './treasurehunt_v2_pgrad/'):
        os.makedirs(path, exist_ok = True)
        for i, e in enumerate(self.demo_env[:3]):
            states = e.get_all_states()
            policy = self.get_policy(states)
            path_ = os.path.join(path, f'visualize_policy_{i}_{itr}.png')
            e.visualize_policy(policy, path_)

            path_ = os.path.join(path, f'visualize_policy_{i}_{itr}.gif')
            e.visualize_policy_execution(policy, path_)
        
    def validate(self):

        rewards = []
        df = np.exp(self.df_value, np.arange(60).astype(float))
        for env in self.demo_env:
            states = env.get_all_states()
            policy = self.get_policy(states)
            reward = env.get_policy_rewards(policy)
            rewards.append((reward*df).sum())
        rewards = sum(rewards) / len(rewards)
        return rewards
    


    def choose_action(self, state):
        state = torch.tensor(state).float()[None,:]
        ## YOUR CODE HERE ##
        return action


    def learn_policy(self, itrs = 50000):
        losses = []
        for i in tqdm(range(1, itrs)):
            

            ## YOUR CODE HERE  ##
            
            wandb.log({'loss': loss.item(), "steps": i})

            if(i % 100 == 0):
                
                rewards = self.validate()
                wandb.log({'rewards': rewards, "steps": i})
                print(f"loss: {sum(losses)/len(losses)}  reward: {rewards} eps: {self.eps} buffer reward: {self.dataset.average_reward()}")
                losses = []
            
            if(i % 300 == 0):
                self.visualize_policy(i)


N = 10

locations = {
    'ship': [(0,0)],
    'land': [(3,0),(3,1),(3,2),(4,2),(4,1),(5,2),(0,N-3),(0,N-2),(0,N-1),(1,N-3),(1,N-2),(2,N-3)],
    'fort': [(N-1,N-1)],
    'pirate': [(4,N-3),(N-2,N-5)],
    'treasure': [(4,0),(1,N-1)]
} 

demo_env1 = TreasureHuntExtreme(locations = locations, N = N)
demo_envs = [demo_env1]
for i in range(99):
    demo_envs.append(TreasureHuntExtreme(N = N))    
the = TreasureHuntExtreme(N = N)
pagent = PolicyGradientAgent(the, demo_envs)
pagent.learn_policy()
