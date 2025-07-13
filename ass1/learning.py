from starter_code import HMM
from hmm_param_initilizer import lambda_1
import numpy as np
from multiprocessing import Pool
R = 10000
Time=20
lam1=lambda_1()
hmm = HMM(lam1.N, lam1.T_prob,lam1.B_list,lam1)

trajectories = [hmm.get_trajectory(Time, i) for i in range(R)]
observations = [hmm.sample_tranjectory_reading(traj,42) for traj in trajectories]

# def backward_pass(obs):
#     beta=np.zeros((Time,hmm.n))
#     beta[Time-1,:]=1
#     for t in range(Time-2,-1,-1):
#         for state_curr in range(hmm.n):
#             for state_next in range(hmm.n):
#                 beta[t,state_curr]+=beta[t+1,state_next]*hmm.T[state_curr][state_next]*hmm.B[state_next][HMM.binary_to_int(obs[t+1])]

#     return beta

# def forward_pass(obs):
#     alpha=np.zeros((Time,hmm.n))
#     alpha[0,:]=hmm.rho*hmm.B[:,HMM.binary_to_int(obs[0])]
#     for t in range(1,Time):
#         for state_curr in range(hmm.n):
#             for state_prev in range(hmm.n):
#                 alpha[t,state_curr]+=alpha[t-1,state_prev]*hmm.T[state_prev][state_curr]*hmm.B[state_curr][HMM.binary_to_int(obs[t])]

#     return alpha

def forward_pass(obs):
    alpha = np.zeros((Time, hmm.n))
    alpha[0, :] = hmm.rho * hmm.B[:, HMM.binary_to_int(obs[0])]
    for t in range(1, Time):
        alpha[t, :] = alpha[t - 1, :].dot(hmm.T) * hmm.B[:, HMM.binary_to_int(obs[t])]
    return alpha

def backward_pass(obs):
    beta = np.zeros((Time, hmm.n))
    beta[Time - 1, :] = 1
    for t in range(Time - 2, -1, -1):
        beta[t, :] = (beta[t + 1, :] * hmm.B[:, HMM.binary_to_int(obs[t + 1])]).dot(hmm.T.T)
    return beta


# def cal_xi(alpha, beta, obs):
#     xi = np.zeros((Time, hmm.n, hmm.n))
#     for t in range(Time - 1):
#         denom = np.sum(alpha[t, :] * beta[t, :])
#         for i in range(hmm.n):
#             for j in range(hmm.n):
#                 xi[t, i, j] = (alpha[t, i] * hmm.T[i, j] * hmm.B[j, HMM.binary_to_int(obs[t + 1])] * beta[t + 1, j]) / denom
#     return xi  


# def create_T(hmm:HMM,T_porb):
    
#         pr, pl, pu, pd, ps = T_porb[0], T_porb[1], T_porb[2], T_porb[3], T_porb[4]
#         T = np.zeros((hmm.n, hmm.n))
#         for i in range(hmm.n):
            
#             x = (i % hmm.N) + 1 
#             y = (i // hmm.N) + 1
            
#             T[i,i] = ps

#             if(x + 1 <= hmm.N):
#                 T[i,i+1] = pr
#             else:
#                 T[i,i] += pr
          
#             if(x - 1 > 0):
#                 T[i,i-1] = pl
#             else:
#                 T[i,i] += pl
          
#             if(y + 1 <= hmm.N):
#                 T[i,i+hmm.N] = pu
#             else:
#                 T[i,i] += pu

#             if(y - 1 > 0):
#                 T[i,i-hmm.N] = pd
#             else:
#                 T[i,i] += pd
#         return T

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
    #T+=1e-13
    T /= T.sum(axis=1, keepdims=True)

    return T

def cal_xi(alpha, beta, obs):
    xi = np.zeros((Time - 1, hmm.n, hmm.n))
    for t in range(Time - 1):
        denom = np.sum(alpha[t, :] * beta[t, :])
        xi[t, :, :] = np.outer(alpha[t, :], beta[t + 1, :] * hmm.B[:, HMM.binary_to_int(obs[t + 1])]) * hmm.T
        xi[t, :, :] /= denom
    return xi

# def cal_xi(alpha, beta, obs):
#     xi = np.zeros((Time, hmm.n, hmm.n))
#     for t in range(Time - 1):
#         denom = np.sum(alpha[t, :] * beta[t, :])
#         for i in range(hmm.n):
#             for j in range(hmm.n):
#                 xi[t, i, j] = (alpha[t, i] * hmm.T[i, j] * hmm.B[j, HMM.binary_to_int(obs[t + 1])] * beta[t + 1, j]) / denom
#     return xi  

def kl_divergence(orignal, estimated):
    epsilon = 1e-10
    orignal = np.clip(orignal, epsilon, 1 - epsilon)
    estimated = np.clip(estimated, epsilon, 1 - epsilon)
    
    kl_div = np.sum(orignal * np.log(orignal / estimated), axis=1)
    return np.mean(kl_div)

# def process_observation(obs):
#     alpha = forward_pass(obs)
#     beta = backward_pass(obs)
#     xi = cal_xi(alpha, beta, obs)
#     return xi

def baum_welch(hmm: HMM, observations, max_iter=20):
    #T = np.random.rand(hmm.n, hmm.n)
    original_t=hmm.T
    T = np.full((hmm.n, hmm.n), 1 / hmm.n)
    T /= T.sum(axis=1, keepdims=True)
    hmm.T = T

    for iteration in range(max_iter):
        print(f'Iteration {iteration} ')
        new_T_probs = np.zeros(5)


        for obs in observations:
            alpha = forward_pass(obs)
            beta = backward_pass(obs)
            xi = cal_xi(alpha, beta, obs)

            for x in range(hmm.N):
                for y in range(hmm.N):
                    idx = x * hmm.N + y
                    neighbors = [ ((x+1, y), 0) if x+1 < hmm.N else None,
                        ((x-1, y), 1) if x-1 >= 0 else None, ((x, y+1), 2) if y+1 < hmm.N else None,
                        ((x, y-1), 3) if y-1 >= 0 else None,((x, y), 4)
                    ]
                    neighbors = [n for n in neighbors if n is not None]
                    denom = np.sum([np.sum(xi[:, idx, neighbor[0] * hmm.N + neighbor[1]]) for neighbor, _ in neighbors])

                    if denom > 0:
                        for neighbor, direction in neighbors:
                            neighbor_idx = neighbor[0] * hmm.N + neighbor[1]
                            num = np.sum(xi[:, idx, neighbor_idx])
                            new_T_probs[direction] += num / denom
        
        hmm.T_prob = new_T_probs / np.sum(new_T_probs)
        new_T=create_T(hmm,hmm.T_prob)
        
        
        kl=kl_divergence(original_t,new_T)
        hmm.T=new_T
        print(f'KL_divergence {kl}')
        print(f'Transition prob Right,Left,Up,Down,Same {hmm.T_prob}')

    print('Finished')
    return hmm.T_prob


        
baum_welch(hmm,observations)

