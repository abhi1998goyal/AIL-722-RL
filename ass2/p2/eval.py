import torch
import random
from os import system, name
from time import sleep
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio

def evaluate_policy_taxi(env, q_table,outpath, display_episodes=5):
    q_table = torch.load(q_table)
    total_reward,total_steps=0,0
    display_episodes=5
    for i_episode in range(display_episodes):
        state, info = env.reset()
        rewards,steps= 0,0
        done = False
        frames = []
        frame = env.render()
        frames.append(frame)  
        Traj=0
        while not done:
            action = torch.argmax(q_table[state]).item()  
            state, reward, done, truncated, info = env.step(action)
            frame = env.render()
            frames.append(frame)  
            steps+=1
            rewards+=reward
            Traj+=1
            if(Traj>500):
                break
        frame_rate = 1  
        clip = mpy.ImageSequenceClip(frames, fps=frame_rate)
        clip.write_videofile(f"{outpath}{i_episode}.mp4", codec="libx264")
        total_reward+=rewards
        total_steps+=steps
    print(f"Results after {display_episodes} episodes:")
    print(f"Average timesteps per episode: {total_steps / display_episodes}")
    print(f"Average reward per episode: {total_reward / display_episodes}")


def save_images_as_video(images, path, fps=2):
    with imageio.get_writer(path, fps=fps, codec='libx264') as writer:
        for img in images:          
            if isinstance(img, Image.Image):
                img = np.array(img)
            writer.append_data(img) 


def evaluate_policy_thunt(env, q_table,outpath, display_episodes=5):
    q_table = torch.load(q_table)
    total_reward,total_steps=0,0
    display_episodes=5
    
    for i_episode in range(display_episodes):
        state= env.reset()
        rewards= 0
        steps=0
        done = False
        frames = []
        frame = env.render(return_image = True)
        frame_array = np.array(frame)
        #frames.append(frame)  
        frames.append(frame_array)
        
        Traj=0
       # num_steps=0
        while state!=399 and state!=299 and state!=199 and state!=99 :
        #num_steps<100:
        #state!=399:
            action = torch.argmax(q_table[state]).item()  
            state, reward= env.step(action)
            frame = env.render(return_image = True)
            frame_array = np.array(frame) 
            frames.append(frame_array)
            #frames.append(frame)  
            steps+=1
            rewards+=reward
            Traj+=1
            # if(Traj>500):
            #     break
        total_steps+=steps
        total_reward+=rewards
        #frame_rate = 1  
        print(f'frames {len(frames)}')
        pil_images = [Image.fromarray(arr.astype('uint8')) for arr in frames]
        imageio.mimsave(f'{outpath}{i_episode}.gif', pil_images, duration=2)
        #save_images_as_video(pil_images, f'{outpath}{i_episode}.mp4', fps=2)

    print(f"Results after {display_episodes} episodes:")
    print(f"Average timesteps per episode: {total_steps / display_episodes}")
    print(f"Average reward per episode: {total_reward / display_episodes}")