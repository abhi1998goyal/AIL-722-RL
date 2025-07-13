import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import torch.distributions as distributions

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy
from functools import partial


class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2)You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(self.hyperparameters['buffer_size']) #you can set the max size of replay buffer if you want
        

        #initialize your model and optimizer and other variables you may need
        
        self.PolicyNet=ptu.build_mlp(  input_size = self.observation_dim,
                                    output_size = self.action_dim,
                                    n_layers=self.hyperparameters['n_layers'],
                                    size=self.hyperparameters['hidden_size'],
                                    activation = self.hyperparameters['activation'],
                                    output_activation = self.hyperparameters['out_activation'])

        self.optimizer = optim.AdamW(self.PolicyNet.parameters(),lr=self.hyperparameters['lr'])

        

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        action = self.PolicyNet(observation) #change this to your action
        return action


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        
        action = self.PolicyNet(observation) #change this to your action
        return action 

    
    
    def update(self, observations, actions):
        #*********YOUR CODE HERE******************
        """
        Update the policy network for one step
        """
        self.PolicyNet.train()
        self.optimizer.zero_grad()
        predicted_actions = self.forward(observations)
        loss = torch.nn.functional.mse_loss(predicted_actions, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.PolicyNet.parameters(), self.hyperparameters['gradient_clip'])
        
        self.optimizer.step()
        return loss.item()
    def load_state_dict(self,state_dict):
        #state_dict = {"PolicyNet."+k :v for k,v in state_dict.items()}
        self.PolicyNet.load_state_dict(state_dict)
        
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
        
        #*********YOUR CODE HERE******************
        #Step 1: Sample T-step trajectory using Policy = b_i * (Expert_policy) + (1-b_i)*Curr_policy
        self.PolicyNet.eval()
        def Pol_i(obs :torch.Tensor):
            beta_i=1.0*(self.hyperparameters['beta_decay_factor'])**(itr_num)
            return ptu.from_numpy((beta_i)*self.expert_policy.get_action(obs))+(1-beta_i)*self.get_action(obs)
        Trajs=utils.sample_n_trajectories(env, Pol_i, ntraj=self.hyperparameters['ntraj'], max_length=self.hyperparameters['max_length'], render= False)
        current_train_envsteps=0
        for i in range(len(Trajs)):
            current_train_envsteps+=len(Trajs[i]['reward']) 
        #Step2 :For each trajectory get expert action for each observation
        for traj in Trajs:
            expert_actions=self.expert_policy.get_action(ptu.from_numpy(traj["observation"]))
            traj['action']=expert_actions

        #Step3 :Aggregate this dataset to replay buffer
        self.replay_buffer.add_rollouts(Trajs)

        #Step4 :Train the model using update function on combined dataset 
        episode_loss=0
        for k in range(self.hyperparameters['Train_steps']):
            batch_data=self.replay_buffer.sample_batch(size=self.hyperparameters['batch_size'], required = ['obs', 'acs'])
            curr_loss=self.update(ptu.from_numpy(batch_data['obs']),ptu.from_numpy(batch_data['acs']))
            episode_loss+=curr_loss
        episode_loss/=self.hyperparameters['Train_steps']
        if(itr_num%20==0):
            torch.save(self.PolicyNet.state_dict(), "./models/"+ "imitation_"+env.spec.id +".pth")


        return {'episode_loss': episode_loss, 'trajectories': Trajs, 'current_train_envsteps': current_train_envsteps} #you can return more metadata if you want to




class RLAgent(BaseAgent):

    '''
    Please implements a policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: Please read the note (1), (2), (3) in ImitationAgent class. 
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
        #*********YOUR CODE HERE******************
        with torch.no_grad():  
          mean_action = self.PolicyNet(observation)*self.hyperparameters['output_scaling_factor']
          std = std_dev*torch.eye(mean_action.shape[0]).to(mean_action)  
          # mean_action = self.PolicyNet(observation)
          
          #mean=torch.zeros_like(mean_action).to(mean_action)
          cov =torch.eye(mean_action.shape[1]).to(mean_action)*(std**2)
          ActionDist= torch.distributions.multivariate_normal.MultivariateNormal(mean_action,cov)
          Sampled_action=ActionDist.sample()
          # #Calculate the probability
          # log_prob=ActionDist.log_prob(Sampled_action)
        return Sampled_action #, log_prob 
    

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        #self.PolicyNet.eval()
        with torch.no_grad():
          mean_action = self.PolicyNet(observation)*self.hyperparameters['output_scaling_factor']
        return mean_action #, log_prob 
    
    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        mean_action = self.PolicyNet(observation)*self.hyperparameters['output_scaling_factor']
        return mean_action

    def load_state_dict(self,state_dict):
        #state_dict = {"PolicyNet."+k :v for k,v in state_dict.items()}
        self.PolicyNet.load_state_dict(state_dict)
    
    def Qit(self,rewards):
        discounted_rewards = torch.flip(rewards, dims=[0])
        discount_factors = self.hyperparameters['gamma'] ** torch.arange(len(rewards), device=rewards.device, dtype=rewards.dtype)
        returns = torch.cumsum(discounted_rewards * discount_factors, dim=0)
        returns = torch.flip(returns, dims=[0]) 
        return returns
        
   
    def update(self,Trajs):
        """Given N Trajectories of states, actions and rewards 
        J_theta = 1/N Sum(i=1-N) Sum (t=1 to T)[log(Pi_theta(a_i,t|s_i,t)Q_i,t)]
        """
        #self.PolicyNet.train()
        self.optimizer.zero_grad()
        #Calculate policy loss
        loss=0
        
        #Compute avg reward for baseline:
#        AvgRew=0
#        for Traj_i in Trajs:
#            AvgRew+=Traj_i['reward'].mean()
#        AvgRew/=len(Trajs)
        for Traj_i in Trajs:
            States=ptu.from_numpy(Traj_i['observation'])
            Actions=ptu.from_numpy(Traj_i['action'])
            Pi_theta_at=self.forward(States)
            #print(Pi_theta_at,"Actions:",Actions)
            log_probs= -1/2*torch.sum((Pi_theta_at-Actions)**2,dim=1)  #Std_dev ignored as it is only a constant scaling term in gradient
            Qit=self.Qit(ptu.from_numpy(Traj_i['reward'])) #Subtract AvgReward baseline
            loss-=torch.mean(log_probs*Qit)
        loss.backward()
        self.optimizer.step()
        

        return loss.item()
    
    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        # for param in self.PolicyNet.parameters(): 
        #     print(param)
        #Step 1: Sample T-step trajectory using current policy network
        #self.PolicyNet.eval()
        std_dev=max(self.hyperparameters['std_dev_min'],self.hyperparameters['std_dev_max']*(self.hyperparameters['std_dev_DF'])**(itr_num/self.hyperparameters['std_dev_K']))
        get_action_fn=partial(self.get_action_from_dist,std_dev=std_dev)
        Trajs=utils.sample_n_trajectories(env, get_action_fn, ntraj=self.hyperparameters['ntraj'], max_length=self.hyperparameters['max_length'], render= False)
        current_train_envsteps=0
        for i in range(len(Trajs)):
            current_train_envsteps+=len(Trajs[i]['reward']) 
        
        #Step4 :Train the model using update function on combined trajectories sampled
        #Trajs contains list of dictionary for each trajectory  
        episode_loss=self.update(Trajs)
        if(itr_num%20==0):
            torch.save(self.PolicyNet.state_dict(), "./models/"+ "rl_"+env.spec.id +".pth")
        
        return {'episode_loss': episode_loss, 'trajectories': Trajs, 'current_train_envsteps': current_train_envsteps,'std_dev':std_dev} #you can return more metadata if you want to


class ActorCriticAgent(RLAgent):
    def __init__(self, observation_dim:int, action_dim:int, args=None, discrete:bool=False, **hyperparameters):
        # Call the base class constructor to initialize actor network and other parameters
        super().__init__(observation_dim, action_dim, args, discrete, **hyperparameters)
        
        # Define the Critic network (value fn estimator)
        self.CriticNet = ptu.build_mlp(
            input_size=self.observation_dim,
            output_size=1,  # Critic outputs a scalar value (value fn)
            n_layers=self.hyperparameters['critic_n_layers'],
            size=self.hyperparameters['critic_hidden_size'],
            activation=self.hyperparameters['critic_activation'],
            output_activation='identity'
        )

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
            Pi_theta_at = self.forward(States)
            log_probs = -1 / 2 * torch.sum((Pi_theta_at - Actions) ** 2, dim=1)

            # Calculate discounted rewards (returns)
            Q_targets = self.Qit(Rewards)
            
            # Critic estimates the state value for the given states
            Q_values = self.CriticNet(States).squeeze()

            # Calculate the critic loss (MSE between estimated Q-values and actual returns)
            critic_loss += F.mse_loss(Q_values, Q_targets)

            # Actor's loss is based on the advantage (Q_target - Q_value) and log_probs
            advantage = Q_targets - Q_values
            policy_loss -= torch.mean(log_probs * advantage.detach()) #Advantage Detached

        # Backpropagate the losses
        policy_loss.backward()
        critic_loss.backward()

        # Perform optimization steps
        self.optimizer.step()  # Update the actor (policy)
        self.critic_optimizer.step()  # Update the critic
        
        # Return the losses for monitoring
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
        Policy="MultiInputPolicy" if self.env_name=='PandaPush-v3' else "MlpPolicy"
        replay_buffer_class=None
        replay_buffer_kwargs=None
        if self.hyperparameters['HER']:
            from stable_baselines3 import HerReplayBuffer
            print("Using Hindsight Experience Replay")
            replay_buffer_class=HerReplayBuffer
            # Parameters for HER
            replay_buffer_kwargs=dict(
                n_sampled_goal=self.hyperparameters['HER_n_goals'],
                goal_selection_strategy='future',
            )
        
        if self.algorithm == 'PPO':
            from stable_baselines3 import PPO
            self.model = PPO(Policy, self.env,verbose=1, tensorboard_log="../data/")
        elif self.algorithm == 'DDPG':
            from stable_baselines3 import DDPG
            self.model = DDPG(Policy, self.env,verbose=1, tensorboard_log="../data/",
                              replay_buffer_class=replay_buffer_class,replay_buffer_kwargs=replay_buffer_kwargs)
        elif self.algorithm == 'SAC':
            from stable_baselines3 import SAC
            self.model = SAC(Policy, self.env,verbose=1, tensorboard_log="../data/",
                            replay_buffer_class=replay_buffer_class,replay_buffer_kwargs=replay_buffer_kwargs)
        elif self.algorithm == 'TD3':
            from stable_baselines3 import TD3
            self.model = TD3(Policy, self.env,verbose=1, tensorboard_log="../data/",
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
        
        if self.env_name=='PandaPush-v3':
            self.model = self.model.load(checkpoint_path,env=self.env)
        else:
            self.model = self.model.load(checkpoint_path)
    
    def get_action(self, observation):
        #implement your get action function
        action, _states = self.model.predict(observation)
        del _states
        return action, None