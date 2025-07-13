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