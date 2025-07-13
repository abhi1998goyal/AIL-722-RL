import numpy as np
from starter_code import HMM

def logforward_algorithm(hmm: HMM, observ, epsilon=1e-10):
    N = hmm.N
    Time = len(observ)
    alphas = np.zeros((Time, hmm.n))
    rho = hmm.rho
    B = hmm.B
    alphas[0, :] = np.log(rho + epsilon) + np.log(B[:, HMM.binary_to_int(observ[0])] + epsilon)
    
    for t in range(1, Time):
        for state_curr in range(hmm.n):
            log_sum = np.logaddexp.reduce(alphas[t-1, :] + np.log(hmm.T[:, state_curr] + epsilon))
            alphas[t, state_curr] = log_sum + np.log(B[state_curr][HMM.binary_to_int(observ[t])] + epsilon)

    log_likelihood = np.logaddexp.reduce(alphas[Time-1, :])
    return log_likelihood

