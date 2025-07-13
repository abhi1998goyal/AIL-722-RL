from starter_code import HMM
from hmm_param_initilizer import lambda_1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

R = 10000
Time=20
lam1=lambda_1()
hmm = HMM(lam1.N, lam1.T_prob,lam1.B_list,lam1)

trajectories = [hmm.get_trajectory(Time, i) for i in range(R)]
observations = [hmm.sample_tranjectory_reading(traj) for traj in trajectories]


def sensor_1_prob(i, j):
        if 1 <= i <= 9 and 1 <= j <= 9:
            return (18 - (i-1) - (j-1)) / 18
        return 0
    
def sensor_2_prob(i, j):
    if 1 <= i <= 9 and 7 <= j <= 15:
        return (18 - (i-1) + (j-15)) / 18
    return 0

def sensor_3_prob(i, j):
    if 7 <= i <= 15 and 7 <= j <= 15:
        return (18 + (i-15) + (j-15)) / 18
    return 0

def sensor_4_prob(i, j):
    if 7 <= i <= 15 and 1 <= j <= 9:
        return (18 + (i-15) - (j-1)) / 18
    return 0

def forward_pass(obs):
    alpha = np.zeros((Time, hmm.n))
    alpha[0, :] = hmm.rho * hmm.B[:, HMM.binary_to_int(obs[0])]
    for t in range(1, Time):
        alpha[t, :] = alpha[t - 1, :].dot(hmm.T) * hmm.B[:, HMM.binary_to_int(obs[t])]
    for t in range(0, Time):
        alpha[t,:] = alpha[t,:]/np.sum(alpha[t,:])
    return alpha

def backward_pass(obs):
    beta = np.zeros((Time, hmm.n))
    beta[Time - 1, :] = 1
    for t in range(Time - 2, -1, -1):
        beta[t, :] = (beta[t + 1, :] * hmm.B[:, HMM.binary_to_int(obs[t + 1])]).dot(hmm.T.T)
    for t in range(0, Time):
        beta[t,:] = beta[t,:]/np.sum(beta[t,:])
    return beta

def create_T(hmm, T_probs, smoothing=1e-8):
    pr, pl, pu, pd, ps = T_probs
    T = np.zeros((hmm.n, hmm.n))
    for i in range(hmm.n):
        x = (i % hmm.N) + 1
        y = (i // hmm.N) + 1
        T[i, i] = ps + smoothing 
        if x + 1 <= hmm.N:
            T[i, i + 1] = pr
        else:
            T[i, i] += pr
        if x - 1 > 0:
            T[i, i - 1] = pl
        else:
            T[i, i] += pl
        if y + 1 <= hmm.N:
            T[i, i + hmm.N] = pu
        else:
            T[i, i] += pu
        if y - 1 > 0:
            T[i, i - hmm.N] = pd
        else:
            T[i, i] += pd
    T /= T.sum(axis=1, keepdims=True)
    return T

def compute_log_likelihood(alpha):
    max_alpha = np.max(alpha[-1, :])
    log_likelihood = max_alpha + np.log(np.sum(np.exp(alpha[-1, :] - max_alpha)))
    return log_likelihood

def cal_xi(alpha, beta, obs):
    xi = np.zeros((Time - 1, hmm.n, hmm.n))
    for t in range(Time - 1):
        #denom = np.sum(alpha[t, :] * beta[t, :])
        denom = np.sum(alpha[Time-1,:])
        xi[t, :, :] = np.outer(alpha[t, :], beta[t + 1, :] * hmm.B[:, HMM.binary_to_int(obs[t + 1])]) * hmm.T
        xi[t, :, :] /= denom
    # for t in range(Time - 1):
    #     xi[t,:,:]=xi[t,:,:]/np.sum(xi[t,:,:])
    return xi

def cal_gamma(alpha, beta):
    gamma = np.zeros((Time, hmm.n))
    for t in range(Time):
        gamma[t, :] = alpha[t, :] * beta[t, :]
        gamma[t, :] /= np.sum(alpha[Time-1, :])
    #for t in range(Time):
      #  gamma[t,:] = gamma[t,:]/np.sum(gamma[t,:])
    return gamma


def update_B1(hmm: HMM, gammas, observations, epsilon=1e-8):
    B1 = np.zeros(hmm.n)
    B2 = np.zeros(hmm.n)
    B3 = np.zeros(hmm.n)
    B4 = np.zeros(hmm.n)
    denom = np.zeros(hmm.n)
    for obs_index, gamma in enumerate(gammas):
        obs = observations[obs_index]
        for t in range(len(obs)):
            obs_val1 = obs[t][0]
            obs_val2 = obs[t][1]
            obs_val3 = obs[t][2]
            obs_val4 = obs[t][3]
            for state in range(hmm.n):
                if obs_val1 == 1:
                    B1[state] += gamma[t, state]
                if obs_val2 == 1:
                    B2[state] += gamma[t, state]
                if obs_val3 == 1:
                    B3[state] += gamma[t, state]
                if obs_val4 == 1:
                    B4[state] += gamma[t, state]
                denom[state] += gamma[t, state]

    for state in range(hmm.n):
        x = (state % hmm.N) + 1
        y = (state // hmm.N) + 1
        if 1 <= x <= 9 and 1 <= y <= 9 and denom[state] != 0:
            B1[state] = (B1[state] + epsilon) / denom[state]
        else:
            B1[state] = 0
        if 1 <= x <= 9 and 7 <= y <= 15 and denom[state] != 0:
            B2[state] = (B2[state] + epsilon) / denom[state]
        else:
            B2[state] = 0
        if 7 <= x <= 15 and 7 <= y <= 15 and denom[state] != 0:
            B3[state] = (B3[state] + epsilon) / denom[state]
        else:
            B3[state] = 0
        if 7 <= x <= 15 and 1 <= y <= 9 and denom[state] != 0:
            B4[state] = (B4[state] + epsilon) / denom[state]
        else:
            B4[state] = 0

    return [B1, B2, B3, B4]




def kl_divergence(original, estimated):
    epsilon = 1e-10
    original = np.clip(original, epsilon, 1 - epsilon)
    estimated = np.clip(estimated, epsilon, 1 - epsilon)
    
    kl_div = np.sum(original * np.log(original / estimated), axis=1)
    return np.mean(kl_div)

def plot_sensor_heatmap(B, title, cmap="viridis"):
    grid_size = int(np.sqrt(len(B))) 
    B_grid = B.reshape(grid_size, grid_size)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(B_grid[::-1, :], cmap=cmap, annot=True, fmt=".2f", cbar=True)  
    ax.set_title(title)
    ax.set_yticks(np.arange(grid_size) + 0.5) 
    ax.set_yticklabels(np.arange(grid_size)[::-1])  
    plt.savefig(f'/home/RL/ass1/plots/{title}.png')
    #plt.show()

def visualize_B_on_grid(B_list):
    titles = ["Sensor 1 Emission Probabilities", "Sensor 2 Emission Probabilities", 
              "Sensor 3 Emission Probabilities", "Sensor 4 Emission Probabilities"]

    for i, B in enumerate(B_list):
        plot_sensor_heatmap(B, titles[i])

def baum_welch(hmm: HMM, observations, max_iter=20, smoothing=1e-6):
    #T = np.random.rand(hmm.n, hmm.n)
    original_t = hmm.T
    T = np.full((hmm.n, hmm.n), 1 / hmm.n)
    T /= T.sum(axis=1, keepdims=True)

    hmm.T = T
    
    original_B=hmm.B
   # B =  np.random.rand(hmm.n, 16)
    B = np.full((hmm.n, 16), 1 / 16)
    B /= B.sum(axis=1,keepdims=True)
    hmm.B = B

   # prev_log_likelihood = None


    for iteration in range(max_iter):
        print(f'Iteration {iteration} ')
        new_T_probs = np.zeros(5)
        gammas=[]
        #log_likelihood = 0
        for obs in observations:
            
            alpha = forward_pass(obs)
            beta = backward_pass(obs)
            xi = cal_xi(alpha, beta, obs)
            gamma = cal_gamma(alpha,beta)
            #log_likelihood += compute_log_likelihood(alpha)
            for x in range(hmm.N):
                for y in range(hmm.N):
                    idx = x * hmm.N + y
                    neighbors = [
                        ((x+1, y), 0) if x+1 < hmm.N else None,((x-1, y), 1) if x-1 >= 0 else None,
                        ((x, y+1), 2) if y+1 < hmm.N else None,((x, y-1), 3) if y-1 >= 0 else None,
                        ((x, y), 4)
                    ]
                    neighbors = [n for n in neighbors if n is not None]
                    denom = np.sum([np.sum(xi[:, idx, neighbor[0] * hmm.N + neighbor[1]]) for neighbor, _ in neighbors])
                    
                    

                    if denom > 0:
                        for neighbor, direction in neighbors:
                            neighbor_idx = neighbor[0] * hmm.N + neighbor[1]
                            num = np.sum(xi[:, idx, neighbor_idx])
                            new_T_probs[direction] += num / denom

            gammas.append(gamma)
        
        hmm.T_prob = (new_T_probs ) / np.sum(new_T_probs )
        new_T=create_T(hmm,hmm.T_prob)
        hmm.T=new_T
        
        blist = update_B1(hmm, gammas, observations)
        # b2 = update_B1(hmm, gammas, observations,sensor=2)
        # b3 = update_B1(hmm, gammas, observations,sensor=3)
        # b4 = update_B1(hmm, gammas, observations,sensor=4)
        B_list = blist
        hmm.B=hmm.create_B(B_list)
        #hmm.B[:,0]+=1e-8
        hmm.B /= hmm.B.sum(axis=1, keepdims=True)

        kl_div = kl_divergence(original_B,hmm.B)
        kl_divt = kl_divergence(original_t,new_T)
        #log_likelihood = compute_log_likelihood(alpha
        hmm.T=new_T
        print(f'Transition prob Right,Left,Up,Down,Same {hmm.T_prob}')
        #print(f'MSE : {mse}')
        print(f'KL div T: {kl_divt}  KL div B: {kl_div}')
        #visualize_B_on_grid(blist)

        # plt.figure(figsize=(10, 8))
        # sns.heatmap(hmm.B, cmap="viridis")
        # plt.title("Emission Matrix B Heatmap")
        # plt.show() 
        


    print('Finished.')
    visualize_B_on_grid(blist)
    return hmm.T_prob


        
baum_welch(hmm,observations)

