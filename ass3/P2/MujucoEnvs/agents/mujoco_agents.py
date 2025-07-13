import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
from functools import partial


class RLAgent(BaseAgent):

    '''
    Please implements a policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        #initialize your model and optimizer and other variables you may need
        
        self.PolicyNet=ptu.build_mlp(input_size = self.observation_dim,
                                    output_size = self.action_dim,
                                    n_layers=self.hyperparameters['n_layers'],
                                    size=self.hyperparameters['hidden_size'],
                                    activation = self.hyperparameters['activation'],
                                    output_activation = self.hyperparameters['out_activation'])

        print("PolicyNET:")
        print(self.PolicyNet)
        self.optimizer = optim.Adam(self.PolicyNet.parameters(),lr=self.hyperparameters['lr'])


    @torch.no_grad()
    def get_action_from_dist(self, observation: torch.FloatTensor,std_dev=0.1):
        """
        Peroforms forward pass and samples an action from the action distribution
        """
        
        #*********YOUR CODE HERE******************
        #with torch.no_grad():  
          
          
        return Sampled_action
    

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        """
        Performs forward pass to get mean action without any sampling and without gradient computation
        """
        #*********YOUR CODE HERE******************
        #with torch.no_grad():  
        
        return mean_action  
    
    def forward(self, observation: torch.FloatTensor):
        """
        Performs forward pass to get mean action
         """
        #*********YOUR CODE HERE******************
        
        return mean_action

    def load_state_dict(self,state_dict):
        #Function to load state_dict of the parameters of model
        
    def Qit(self,rewards):
        """
        Calulates returns, i.e. the discounted rewards to go
        
        """
        discounted_rewards = torch.flip(rewards, dims=[0])
        discount_factors = self.hyperparameters['gamma'] ** torch.arange(len(rewards), device=rewards.device, dtype=rewards.dtype)
        returns = torch.cumsum(discounted_rewards * discount_factors, dim=0)
        returns = torch.flip(returns, dims=[0]) 
        
        return returns
        
   
    def update(self,Trajs):
        """Given N Trajectories of states, actions and rewards calculates policy loss
        J_theta =  -1/N Sum(i=1 to N) Sum (t=1 to T)[log(Pi_theta(a_i,t|s_i,t)Q_i,t)]
        and updated parameters of model
       
        For average reward baseline modify this function, one way to compute avg reward can be : 
        Avg_Rew = 1/N Sum(i=1 to N) [ Sum (t=1 to T) [rew_i_t]]
       
        """
        
        
        #for Traj_i in Trajs:
            #States=ptu.from_numpy(Traj_i['observation'])
            #Actions=ptu.from_numpy(Traj_i['action'])
            #Pi_theta_at=self.forward(States)
            #Qit=self.Qit(ptu.from_numpy(Traj_i['reward'])) 
            
        
        return loss
    
    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #Step 1: Sample T-step trajectory using current policy network
        std_dev=max(self.hyperparameters['std_dev_min'],self.hyperparameters['std_dev_max']*(self.hyperparameters['std_dev_DF'])**(itr_num/self.hyperparameters['std_dev_K']))
        get_action_fn=partial(self.get_action_from_dist,std_dev=std_dev)
        Trajs=utils.sample_n_trajectories(env, get_action_fn, ntraj=self.hyperparameters['ntraj'], max_length=self.hyperparameters['max_length'], render= False)
        current_train_envsteps=0
        for i in range(len(Trajs)):
            current_train_envsteps+=len(Trajs[i]['reward']) 
        #Step2 :Train the model using update function on combined trajectories sampled
        #Trajs contains list of dictionary for each trajectory  
        episode_loss=self.update(Trajs)
        if(itr_num%20==0):
            torch.save(self.PolicyNet.state_dict(), "./models/"+ "rl_"+env.spec.id +".pth")
        return {'episode_loss': episode_loss, 'trajectories': Trajs, 'current_train_envsteps': current_train_envsteps,'std_dev':std_dev} #you can return more metadata if you want to



class ActorCriticAgent(RLAgent):
    def __init__(self, observation_dim:int, action_dim:int, args=None, discrete:bool=False, **hyperparameters):
        # Calling the base class constructor to initialize actor policy network and other parameters 
        super().__init__(observation_dim, action_dim, args, discrete, **hyperparameters)
        # Note that actor network gets initialized through the above constructor call, no need to define again
        # Define the Critic network (value fn estimator) similar to RLAgent class, add additional hyperparams in config if required
        # self.CriticNet = 
      
        print("CriticNET:")
        print(self.CriticNet)
        
        # Critic optimizer
        self.critic_optimizer = optim.Adam(self.CriticNet.parameters(), lr=self.hyperparameters['lr'])
        
        
    def update(self, Trajs):
        """Update both the policy (actor) and the critic (Q-value estimator)"""
        # Reset gradients
        self.optimizer.zero_grad()  # Actor optimizer
        self.critic_optimizer.zero_grad()  # Critic optimizer
        
        # Initialize policy loss and critic loss
        policy_loss = 0
        critic_loss = 0
        
        for Traj_i in Trajs:
            States = ptu.from_numpy(Traj_i['observation'])
            Actions = ptu.from_numpy(Traj_i['action'])
            Rewards = ptu.from_numpy(Traj_i['reward'])
            
            # Get action probabilities from the policy (actor)
            
            # Calculate discounted rewards (returns)
            
            # Critic estimates the state value for the given states
            
            # Calculate the critic loss (MSE between estimated Q-values and actual returns)
            
            # Actor's loss is based on the advantage (Q_target - Q_value) and log_probs
            
        # Backpropagate the losses
        policy_loss.backward()
        critic_loss.backward()

        # Perform optimization steps
        self.optimizer.step()  # Update the actor (policy)
        self.critic_optimizer.step()  # Update the critic
        
        return {'policy_loss': policy_loss.item(), 'critic_loss': critic_loss.item()}
    
    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        """Overrides the train_iteration method to include both actor and critic updates"""
        # Sample T-step trajectory using current policy
        std_dev = max(
            self.hyperparameters['std_dev_min'],
            self.hyperparameters['std_dev_max'] * (self.hyperparameters['std_dev_DF']) ** (itr_num / self.hyperparameters['std_dev_K'])
        )
        get_action_fn = partial(self.get_action_from_dist, std_dev=std_dev)
        Trajs = utils.sample_n_trajectories(env, get_action_fn, ntraj=self.hyperparameters['ntraj'], max_length=self.hyperparameters['max_length'], render=False)
        
        current_train_envsteps = sum([len(Traj_i['reward']) for Traj_i in Trajs])

        # Update the actor (policy) and critic using the sampled trajectories
        losses = self.update(Trajs)
        episode_loss = losses['policy_loss']
        critic_loss = losses['critic_loss']
        
        # Optionally save the model
        if itr_num % 20 == 0:
            torch.save(self.PolicyNet.state_dict(), "./models/" + "RLAC_" + env.spec.id + "_actor.pth")
            torch.save(self.CriticNet.state_dict(), "./models/" + "RLAC_" + env.spec.id + "_critic.pth")

        # Return training metadata
        return {
            'episode_loss': episode_loss,
            'critic_loss': critic_loss,
            'trajectories': Trajs,
            'current_train_envsteps': current_train_envsteps,
            'std_dev': std_dev
        }





class SBAgent(BaseAgent):
    def __init__(self, env_name, **hyperparameters):
        #implement your init function
        from stable_baselines3.common.env_util import make_vec_env
        #from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
        
        self.hyperparameters=hyperparameters
        self.algorithm=self.hyperparameters['algorithm']
        self.env_name = env_name
        
        self.env = make_vec_env(self.env_name) #initialize your environment. This variable will be used for evaluation. See train_sb.py
        Policy= "MlpPolicy"
        
        replay_buffer_class=None
        replay_buffer_kwargs=None
        if self.algorithm == 'PPO':
            from stable_baselines3 import PPO
            self.model = PPO(Policy, self.env,verbose=1, tensorboard_log="./data/")
        elif self.algorithm == 'DDPG':
            from stable_baselines3 import DDPG
            self.model = DDPG(Policy, self.env,verbose=1, tensorboard_log="./data/",
                              replay_buffer_class=replay_buffer_class,replay_buffer_kwargs=replay_buffer_kwargs)
        elif self.algorithm == 'SAC':
            from stable_baselines3 import SAC
            self.model = SAC(Policy, self.env,verbose=1, tensorboard_log="./data/",
                            replay_buffer_class=replay_buffer_class,replay_buffer_kwargs=replay_buffer_kwargs)
        elif self.algorithm == 'TD3':
            from stable_baselines3 import TD3
            self.model = TD3(Policy, self.env,verbose=1, tensorboard_log="./data/",
                            replay_buffer_class=replay_buffer_class,replay_buffer_kwargs=replay_buffer_kwargs)
        else:
            raise ValueError("Choose one of the following methods: 'PPO', 'DDPG', 'SAC', 'TD3'")
        
    def learn(self):
        #implement your learn function. You should save the checkpoint periodically to <env_name>_sb3.zip
        from stable_baselines3.common.callbacks import CheckpointCallback,  ProgressBarCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
          save_freq=self.hyperparameters['save_freq'],
          save_path="./logs/",
          name_prefix=f"{self.algorithm}_{self.env_name}_{self.hyperparameters['Run_name']}",
          save_replay_buffer=True,
          save_vecnormalize=True,
        )

        eval_env= make_vec_env(self.env_name)
        eval_callback = EvalCallback(eval_env, best_model_save_path="../models/",
                             log_path="./data/", eval_freq=self.hyperparameters['eval_freq'],
                             deterministic=True, render=False)

        self.model.learn(total_timesteps=self.hyperparameters['total_timesteps'], log_interval=self.hyperparameters['log_interval'],
                         tb_log_name=f"{self.algorithm}_{self.env_name}_{self.hyperparameters['Run_name']}", progress_bar=True, 
                         callback= [eval_callback,checkpoint_callback])
        
    def load_checkpoint(self, checkpoint_path):
        #implement your load checkpoint function
        from stable_baselines3.common.vec_env import DummyVecEnv
        self.model = self.model.load(checkpoint_path)
    
    def get_action(self, observation):
        #implement your get action function
        action, _states = self.model.predict(observation)
        del _states
        return action, None