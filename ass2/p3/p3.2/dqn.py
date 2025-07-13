
#from sumtree import SumTree 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch 
import torch.nn as nn
import random 
import os 
#import wandb
from collections import deque
from env import TreasureHunt_v2
#wandb.init(mode="offline", project="treasurehuntv2_qdn")
#*************Linked List implmentation**************

class CacheData:
    def __init__(self, cache_capacity=100000):
        self.cache_capacity = cache_capacity
        self.data = []
        self.priorities = []
    
    def add_item(self, item, priority):
        
        if len(self.data) >= self.cache_capacity:
           
            self.data.pop(0)
            self.priorities.pop(0)
     
        self.data.append(item)
        self.priorities.append(priority)

    def sample_batch(self, batch_size, beta=0.4):
     
        priorities = np.array(self.priorities)
        sampling_probabilities = priorities / priorities.sum() 

        indices = np.random.choice(len(self.data), batch_size, p=sampling_probabilities)
        batch = [self.data[idx] for idx in indices]
      
        weights = (1.0 / len(self.data) / sampling_probabilities[indices]) ** beta
        weights /= weights.max()  
        
        states, actions, next_states, rewards, dones = zip(*batch)

        return states, actions, next_states, rewards, dones, weights, indices

    def update_priorities(self, indices, td_errors):
       
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5   


class QNetwork(nn.Module):

    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1=nn.Conv2d(4,64,(3,3),stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 9, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()


        #### Your Code Here ###

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
        

class DeepQAgent:

    def __init__(self, env, demo_env,device, cache_size = 100000,tau=0.005,beta_start=0.4):

        self.env = env 
        self.device=device
        self.dataset = CacheData(cache_size)
        self.qnetwork = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)
        self.target_network.load_state_dict(self.qnetwork.state_dict()) 
        self.target_network.eval() 
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr = 0.00005)
        
        #TO VALIDATE
        self.demo_env = demo_env

        self.eps = 1.0  
        self.eps_decay = 0.995
        self.eps_min = 0.01
        self.tau = tau 
        self.beta_start = beta_start
        self.beta = beta_start
        

    def get_policy(self, states):
        states = torch.tensor(states).float().to(self.device)
        qsa = self.qnetwork(states)
        qsa = qsa.argmax(dim = -1)
        qsa = torch.nn.functional.one_hot(qsa,4)
        policy = qsa.cpu().numpy()
        return policy

    def visualize_policy(self, itr, path = '/home/scai/mtech/aib232073/RL/ass2/plots/p3.2'):
        os.makedirs(path, exist_ok = True)
        for i, e in enumerate(self.demo_env[:5]):
            states = e.get_all_states()
            policy = self.get_policy(states)
           # path_ = os.path.join(path, f'TreasureHuntV2_policy_{i}_{itr}.png')
           # e.visualize_policy(policy, path_)

            path_ = os.path.join(path, f'TreasureHuntV2_policy_{i}_{itr}.gif')
            e.visualize_policy_execution(policy, path_)


    def validate(self, num_episodes=10):
            total_rewards = []
        
            for i in range(num_episodes):
                state = self.env.reset()
                episode_reward = 0
                done = False
            
                for k in range(100):
                    state_tensor = torch.tensor(state).float().unsqueeze(0).to(self.device)
                    q_values = self.qnetwork(state_tensor)
                    
                    action = torch.argmax(q_values).cpu().item()
                
                    next_state, reward = self.env.step(action)
                    episode_reward += reward
                    state = next_state

                    done=self.env.check_done()

                    if done:
                        break
            
                total_rewards.append(episode_reward)
        
            average_reward = np.mean(total_rewards)
            return average_reward

        

    def choose_action(self, state):
        if np.random.rand() < self.eps:
            return random.randint(0,3)
        else:
            state_tensor = torch.tensor(state).float().unsqueeze(0).to(self.device)
            q_values = self.qnetwork(state_tensor)
            return torch.argmax(q_values).cpu().item()

    def update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.qnetwork.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def learn_policy(self, itrs = 50000,gamma=0.99,max_episode_length=100):
        
        losses = []
        for i in tqdm(range(1, itrs),leave=False):
        #for i in range(1, itrs):
            state = self.env.reset()
            done=False

            for j in range(max_episode_length):

                action = self.choose_action(state)
                next_state, reward = self.env.step(action)
                td_error = abs(reward)  
                self.dataset.add_item((state, action, next_state, reward),td_error)
                state = next_state
                done=self.env.check_done()
                if done:
                    break
            
                if len(self.dataset.data) >=256 and len(self.dataset.data)%20==0 :
                    states, actions, next_states, rewards, weights, indices = self.dataset.sample_batch(256, beta=self.beta)
                    states = torch.tensor(states).float().to(self.device)
                    actions = torch.tensor(actions).long().to(self.device)
                    next_states = torch.tensor(next_states).float().to(self.device)
                    rewards = torch.tensor(rewards).float().to(self.device)
                    weights = torch.tensor(weights).float().to(self.device)
                    
                    #YOUR LOGIC TO BACKPROPOGATE THE LOSS THROUGH Q NETWORK
                    q_values = self.qnetwork(states)
                
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states)
                        target_q_values = rewards + gamma * next_q_values.max(dim=1)[0]

                    td_errors = (target_q_values - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)).detach().cpu().numpy()

                # loss = nn.functional.mse_loss(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))
                    loss = (weights * nn.functional.mse_loss(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1), reduction='none')).mean()

                    losses.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.update_target_network()

                    self.dataset.update_priorities(indices, td_errors)

            
            #EPSILON DECAY STEP
            #self.eps = max(self.eps * self.eps_decay, self.eps_min)
            self.eps= max(self.eps_min, 1 * (self.eps_decay ** (i/5)))
            self.beta = min(1.0, self.beta + 0.001)
           

            if(i % 1000 == 0):
                self.visualize_policy(i)
                #demo_values = self.qnetwork.get_all(self.demo_state).max().item()
                rewards = self.validate(100)
                #wandb.log({'rewards': rewards, "steps": i})
                print(f"loss: {sum(losses)/len(losses)}  validation reward: {rewards} eps: {self.eps} ")
                #tqdm.write(f"loss: {sum(losses)/len(losses)}  validation reward: {rewards} eps: {self.eps} ")
                #      buffer reward: {self.dataset.average_reward()}"
                losses = []

    

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
demo_envs = []
for i in range(100):
    demo_envs.append(TreasureHunt_v2())    
env = TreasureHunt_v2()
qagent = DeepQAgent(env, demo_envs,device)
qagent.learn_policy()
