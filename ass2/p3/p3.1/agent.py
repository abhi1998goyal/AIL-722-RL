import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    """Define Neural Network Architecture"""

    def __init__(self, state_size, action_size,seed):
        """
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,32)
        self.fc2 = nn.Linear(32, 64)        
        self.fc3 = nn.Linear(64, 64)       
        self.fc4 = nn.Linear(64, action_size) 


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        ####Your Code Here ####
        self.memory = deque(maxlen=buffer_size)

    
    def add(self, state, action, reward, next_state, done):
       """Add a new experience to memory."""
       exp = self.experience(state, action, reward, next_state, done)
       self.memory.append(exp)
        ####Your Code Here ####
        
    def sample(self, batch_size,device):
        """Randomly sample a batch of experiences from memory.
        Return as tensors with device as required
        """
        ####Your Code Here ####
        exp = random.sample(self.memory, k=batch_size)
        
        #states = torch.tensor(np.vstack([e.state for e in exp])).float().to(device)
        states = torch.tensor(np.vstack([e.state for e in exp])).float().to(device)
        # states = torch.from_numpy(np.vstack([e.state for e in exp])).to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in exp])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in exp])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in exp])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in exp]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, configs):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.configs=configs
        self.state_size = configs['state_size']
        self.action_size = configs['action_size']
        random.seed(configs['seed'])
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #configs['device']
        self.tau =configs['tau']
        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, configs['seed']).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, configs['seed']).to(self.device)
        ####Your Code Here ####

        
        #Setup Optimizer
        # Setup Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=configs['lr'])

        
        #Setup Replay memory
        
        self.memory = ReplayBuffer(self.action_size, configs['buffer_size'], configs['seed'])
        
        self.t_step = 0
        
    def save_checkpoint(self,Path):
        """Save Q_network parameters"""
        torch.save(self.qnetwork_local.state_dict(), Path)
        print(f"Saved Checkpoint at {Path}!")
        ####Your Code Here ####

    def load_checkpoint(self,Path):
        """Load Q_network parameters"""
        self.qnetwork_local.load_state_dict(torch.load(Path))
        print(f"Loaded Checkpoint from {Path} !")
        ####Your Code Here ####
        
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        
        Use eps=0 for final deterministic policy
        """

        state = torch.from_numpy(state).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
           action=np.argmax(action_values.cpu().numpy())
        else:
            action=random.choice(np.arange(self.action_size))
        
        # Epsilon-greedy action selection

        return action
    
    def batch_learn(self, experiences, gamma):
        """
        This method updates the value parameters of the neural network using one batch of experience tuples.

        Arguments:
        - experiences (Tuple[torch.Variable]): A tuple containing five elements:
            * s: The states (tensor)
            * a: The actions (tensor)
            * r: The rewards (tensor)
            * s': The next states (tensor)
            * done: Boolean tensors indicating if the episode has terminated (tensor)

        - gamma (float): The discount factor used in the Q-learning update rule (also called Bellman equation).

        Returns:
        None
        """
        # states, actions, rewards, next_states, dones = experiences
        # #q_max=np.max(self.qnetwork_local(states).detach().cpu().numpy(),axis=1)
        # q_max = self.qnetwork_target(next_states).max(1)[0]
        # q_max=q_max*(1 - dones.float())
        # y_ = (rewards.squeeze() + gamma*q_max).float()
        # with torch.no_grad():
        #      #y = np.max(self.qnetwork_target(states).detach().cpu().numpy(),axis=1)
        #      y=self.qnetwork_local(states).max(1)[0]
        # loss = F.mse_loss(y,y_)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        states, actions, rewards, next_states, dones = experiences

        q_max = self.qnetwork_target(next_states).max(1)[0]
        q_max = q_max * (1 - dones.float().squeeze()) 

        y_ = rewards.squeeze() + gamma * q_max

        q_expected = self.qnetwork_local(states).gather(1, actions.long())
        

        loss = F.mse_loss(q_expected.squeeze(), y_)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

 
        

    def target_update(self, local_model, target_model, tau):
        """Soft update time delayed target model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied 
            target_model (PyTorch model): weights will be copied to
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)




        