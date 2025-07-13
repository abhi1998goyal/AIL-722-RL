import numpy as np
from starter_code import HMM

def forward_algorithm(hmm:HMM, observ):
    N=hmm.N
    Time=len(observ)
    alphas=np.zeros((Time,hmm.n))
    rho=hmm.rho
    B=hmm.B
    alphas[0,:]=rho*B[:,HMM.binary_to_int(observ[0])]
    for t in range(1,Time):
        for state_curr in range(hmm.n):
            for state_prev in range(hmm.n):
                alphas[t,state_curr]+=alphas[t-1,state_prev]*hmm.T[state_prev][state_curr]*B[state_curr][HMM.binary_to_int(observ[t])]

    return np.sum(alphas[Time-1, :])


