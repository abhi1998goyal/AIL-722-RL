import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent import *
from dqn import *
import os
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
import argparse


def load_parameters(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def main(config_path):
    
    AllConfigs = load_parameters(config_path)
    AgentConfigs = AllConfigs['agent']
    TrainingConfigs = AllConfigs['training']
    EnvConfigs = AllConfigs['env']

    wandb.init(project="single_01",
               mode=TrainingConfigs['wandb_mode'],
               name=TrainingConfigs['wandb_run_name'],
               config={
                    "eps_start": TrainingConfigs['eps_start'],
                    "eps_end": TrainingConfigs['eps_end'],
                    "eps_decay": TrainingConfigs['eps_decay'], 
                    "eps_k": TrainingConfigs['eps_k'],
                    "n_episodes": TrainingConfigs['n_episodes'],
                    "batch_size": TrainingConfigs['batch_size'],
                    "gamma": TrainingConfigs['gamma'],
                    "tau": AgentConfigs['tau'],
                    "max_traj_len": TrainingConfigs['max_traj_len'] }
               )

   
    env = gym.make(EnvConfigs['name'], **EnvConfigs['Options'])

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

   
    agent = Agent(AgentConfigs)

   
    scores = dqn(agent, env, TrainingConfigs)

    # Plot the scores

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="LunarLander DQN Configuration")
    parser.add_argument('--config', type=str, default='/home/RL/ass2/p3/p3.1/dqnconfig.yaml',
                        help='Path to the configuration file (YAML format)')

    args = parser.parse_args()

    main(args.config)
