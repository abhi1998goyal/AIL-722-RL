import numpy as np
from starter_code import HMM

def n_2_ij(state):
     x = (state % 15) + 1 
     y = (state // 15) + 1
     return (x,y)

def viterbi_algorithm(hmm: HMM, observ):
    N = hmm.N
    Time = len(observ)
    v=np.zeros((Time,hmm.n))
    bt=np.zeros((Time,hmm.n),dtype=int)
    B=hmm.B
    v[0,:]=hmm.rho*B[:,HMM.binary_to_int(observ[0])]
   # state_seq=[]
    for t in range(1,Time):
        for curr_state in range(0,hmm.n):
            max_state=0
            for prev_state in range(0,hmm.n):
                candidate=v[t-1,prev_state]*hmm.T[prev_state][curr_state]*B[curr_state][HMM.binary_to_int(observ[t])]
                if(v[t,curr_state]<candidate):
                   v[t,curr_state]=candidate
                   max_state=prev_state
            bt[t][curr_state]=max_state
        
    last_state=np.argmax(v[Time-1,:])
    state_seq = np.zeros(Time,dtype=int)
    state_seq[Time-1]=last_state
    state_ij_seq=[None]*Time
    state_ij_seq[Time-1]=n_2_ij(last_state)
    for i in range(Time-2,-1,-1):
        state_seq[i]=bt[i+1][state_seq[i+1]]
        state_ij_seq[i]=n_2_ij(bt[i+1][state_seq[i+1]])
    return state_ij_seq




