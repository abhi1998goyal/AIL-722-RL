import os
import numpy as np
import torch
from collections import deque
import wandb

def dqn(agent,env,TrainingConfigs):
    """Deep Q-Learning.        
    """
    rewards = []  

    
    def eps_fn(episode_i,eps_start=TrainingConfigs['eps_start'],eps_end=TrainingConfigs['eps_end'],eps_decay=TrainingConfigs['eps_decay'],eps_k= TrainingConfigs['eps_k']):
    
        eps= max(eps_end, eps_start * (eps_decay ** (episode_i/eps_k)))
        return eps
    
    best_avg_score=-float('inf')
    

    for i_episode in range(1, TrainingConfigs['n_episodes'] + 1):
        eps=eps_fn(i_episode)   
        state = env.reset()[0]
        
        episode_reward=0
        #episode_length=0
        for t in range(TrainingConfigs['max_traj_len']):
          #  episode_length+=1
            action = agent.act(state,eps)
            next_state, reward, terminated, truncated,_= env.step(action)
            done=terminated or truncated
            agent.memory.add(state,action,reward,next_state,done)
            episode_reward+=reward

            if len(agent.memory) % 5 == 0 and len(agent.memory) > TrainingConfigs['batch_size']:
               experiences= agent.memory.sample(TrainingConfigs['batch_size'],agent.device)
               agent.batch_learn(experiences,TrainingConfigs['gamma'])
               agent.target_update(agent.qnetwork_local, agent.qnetwork_target, agent.tau)

            
            if done:
                break

            state=next_state

        rewards.append(episode_reward)
        #print(f'Episode lenght {episode_length}')

        #if i_episode % TrainingConfigs['target_update'] == 0:
            #agent.target_update(agent.qnetwork_local, agent.qnetwork_target, agent.tau)
      
        
        if i_episode % TrainingConfigs['av_window'] == 0:
            avg_rewards = np.mean(rewards[-TrainingConfigs['av_window']:])


            
            if TrainingConfigs['wandblogging']:
                wandb.log({"Reward" : avg_rewards ,"Episode":i_episode, "Epsilon":eps})
            if avg_rewards > best_avg_score:  
                best_avg_score = avg_rewards
                agent.save_checkpoint(TrainingConfigs['checkpoint_path'])
                print('\r━━━> Episode {}\tAverage Score: {:.2f}   Epsilon:{:.3f} | Saved'.format(i_episode, avg_rewards,eps))
            else:
                print('\rEpisode {}\tAverage Score: {:.2f}  Epsilon:{:.3f}'.format(i_episode, avg_rewards,eps))

        if np.mean(rewards[-TrainingConfigs['av_window']:]) >= 200.0:
            print('\n━━━━━━━━━━━━━━━ SOLVED ━━━━━━━━━━━━━━━')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - TrainingConfigs['av_window'],
                                                                                         avg_rewards))
            agent.save_checkpoint(TrainingConfigs['checkpoint_path'])

            return rewards
                
    return rewards