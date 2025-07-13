import numpy as np
from grid import Grid
from starter_code import HMM
from likelihood import forward_algorithm
from viterbi import viterbi_algorithm
from hmm_param_initilizer import lambda_1
import os
from loglikelihood import logforward_algorithm
import matplotlib.pyplot as plt

lam1=lambda_1()

hmm = HMM(lam1.N, lam1.T_prob,lam1.B_list,lam1)

sensor_probabilities = hmm.calculate_sensor_probabilities()
grid = Grid(M=lam1.N, N=lam1.N)

def visualize_grid_prob():
    for sensor_idx, prob_grid in enumerate(sensor_probabilities):
        print(f"Visualizing sensor {sensor_idx+1} probabilities...")
        for i in range(lam1.N):
            for j in range(lam1.N):
                grid_cell = grid.get_grid(i, j)

                grid_cell[:, :, :] *= prob_grid[i, j] 

        grid.show(f"plots/sensor_{sensor_idx+1}_probabilities.png")
        grid.clear()

#visualize_grid_prob()
#trajectory = get_trajectory(30)
#print(trajectory)


def plot_trajectories(trajectories, output_dir="/home/RL/ass1/plots"):
    grid = Grid(lam1.N, lam1.N)

    for idx, traj in enumerate(trajectories):
        print(f"Plotting trajectory {idx+1}...")
        grid.draw_path(traj, color=[0, 0, 1])
                       
        start_x, start_y = traj[0]
        start_grid = grid.get_grid(start_x-1, start_y-1)
        start_grid[grid.P//2:-grid.P//2, grid.P//2:-grid.P//2] = [1, 0, 0]

        end_x, end_y = traj[-1]
        end_grid = grid.get_grid(end_x-1, end_y-1)
        end_grid[grid.P//2:-grid.P//2, grid.P//2:-grid.P//2] = [0, 1, 0]

        grid.show(f"{output_dir}/trajectory_{idx+1}.png")
        grid.clear()


def save_trajectories_and_observations(trajectories, observations, likelihood ,pred_trajs,output_dir="/home/RL/ass1/reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, "trajectories_and_observations.txt"), "w") as file:
        for idx, (traj, obs,pred_trj) in enumerate(zip(trajectories, observations,pred_trajs)):
            file.write(f"Trajectory {idx + 1} Likelihood {likelihood[idx]}:\n")
            for step in traj:
                file.write(f"{step}\n")
            file.write("\nObservations:\n")
            for ob in obs:
                file.write(f"{ob}\n")
            file.write("\nPredicted Traj using Vitergbi:\n")
            for prd in pred_trj:
                file.write(f"{prd}\n")

            file.write("\n" + "-"*40 + "\n") 

traj_list=[]
for i in range(20):
    seed = i + 1 
    traj = hmm.get_trajectory(30, seed=seed)
    traj_list.append(traj)

#
#print(traj_list[0])

manual_traj = [(1, 1)]
x, y = 1, 1
while x < 15 and y < 15:
    x += 1
    manual_traj.append((x, y))
    y += 1
    manual_traj.append((x, y))

#traj_list.append(manual_traj)
plot_trajectories(traj_list)

traj_sensor_sampled=[]                 
for traj in traj_list:
    traj_sensor_sampled.append(hmm.sample_tranjectory_reading(traj))

#print(traj_sensor_sampled[0])

#save_trajectories_and_observations(traj_list, traj_sensor_sampled)


likelihoods=[]
for traj_readings in traj_sensor_sampled:
    likelihood = forward_algorithm(hmm, traj_readings)
    #likelihood = logforward_algorithm(hmm, traj_readings)
    likelihoods.append(likelihood)
#print(likelihoods[0])



def plot_trajectories_with_predictions(trajectories, predicted_trajectories, output_dir="/home/RL/ass1/plots"):
    grid = Grid(lam1.N, lam1.N)

    for idx, (traj, pred_traj) in enumerate(zip(trajectories, predicted_trajectories)):
        print(f"Plotting trajectory {idx+1} and its prediction...")

        grid.draw_path(traj, color=[0, 0, 1])
        grid.draw_path(pred_traj, color=[0, 1, 0])
                       
        start_x, start_y = traj[0]
        start_grid = grid.get_grid(start_x-1, start_y-1)
        start_grid[grid.P//2:-grid.P//2, grid.P//2:-grid.P//2] = [1, 0, 0]

        end_x, end_y = traj[-1]
        end_grid = grid.get_grid(end_x-1, end_y-1)
        end_grid[grid.P//2:-grid.P//2, grid.P//2:-grid.P//2] = [0, 1, 0]

        grid.show(f"{output_dir}/trajectory_{idx+1}_with_prediction.png")
        grid.clear()

predicted_trajs=[]
for traj_readings in traj_sensor_sampled:
    predicted_traj = viterbi_algorithm(hmm, traj_readings)
    predicted_trajs.append(predicted_traj)

plot_trajectories_with_predictions(traj_list,predicted_trajs)
save_trajectories_and_observations(traj_list, traj_sensor_sampled,likelihoods,predicted_trajs)
#print(viterbi_algorithm(hmm,traj_sensor_sampled[0]))


def manhattan_distance(coord1, coord2):
    return abs(coord1[0] - coord1[1]) + abs(coord2[0] - coord2[1])

def mean_manhattan_distance_vs_time(trajectories, predicted_trajectories):
    Time = len(trajectories[0]) 
    man_dis = np.zeros(Time)
    num_traj = len(trajectories)

    for t in range(Time):
        for true_traj, pred_traj in zip(trajectories, predicted_trajectories):
            man_dis[t] += manhattan_distance(true_traj[t], pred_traj[t])
        man_dis[t] /= num_traj
    
    return man_dis


mean_dis= mean_manhattan_distance_vs_time(traj_list,predicted_trajs)
plt.plot(range(len(mean_dis)), mean_dis, marker='o')
plt.xlabel('Time')
plt.ylabel('Mean Manhattan Distance')
plt.title('Mean Manhattan Distance vs Time')
plt.grid(True)
plt.savefig('/home/RL/ass1/plots/mean_manhattan_distance_vs_time_step.png')