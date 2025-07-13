import numpy as np 
from hmm_param_initilizer import lambda_1
class HMM:

    def __init__(self, N, T_prob,B_list,lamda:lambda_1):
        '''
        Args:
            N: The grid side length
            T_porb: list of [p_right, p_left, p_up, p_down, p_same]
            B_list: contains 4 Bs correponding to 4 sensors
                each B is 1d np array of shape: B: [N^2]
                the sensor matrices are in order of S1, S2, S3, S4
        '''
        assert N > 0
        assert len(T_prob) == 5
        assert len(B_list) == 4
        for B in B_list:
            assert B.shape[0] == N*N 

        self.lamda = lamda
        #side lenght
        self.N = N 

        #number of states
        self.n = (self.N**2)

        #[n x n]
        self.T_prob = T_prob
        self.T = self.create_T(self.T_prob) 
        
        #the sensor index -> sensor symbol
        self._sensor_symbol = self.create_sensor_symbol()
        
        #the sensor grid
        self.sensor_grid = [[1,9,1,9], [7,15,1,9], [7,15,7,15], [1,9,7,15]]

        #[n x 16]
        self.B_list = B_list
        self.B = self.create_B(self.B_list)

        #[n]
        self.rho = np.zeros(self.n)
        self.rho[0] = 1 
    
    def create_sensor_symbol(self):

        sensor_symbol = []
        for s in range(16):
            symbol = []
            for i in range(4):
                symbol.append(s // int(2**(4-i-1)))
                s = s % (int(2**(4-i-1)))
            sensor_symbol.append(symbol)
        return np.array(sensor_symbol)

    #create the matrix
    def create_B(self, B_list):
        '''
        Args:
            B_list: list of 4 sensor Bs of shape [self.n]
        Return:
            B: [n,16]
        '''
        B = np.ones((self.n, 16))
        for j in range(16):
            s = self._sensor_symbol[j]
            for i, k in enumerate(s):
                B[:,j] *= (k*B_list[i] + (1-k)*(1-B_list[i]))
        
        return B

    def create_T(self, T_porb):
        
        #get the probabilities
        pr, pl, pu, pd, ps = T_porb[0], T_porb[1], T_porb[2], T_porb[3], T_porb[4]
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            
            x = (i % self.N) + 1 
            y = (i // self.N) + 1
            
            #self transition
            T[i,i] = ps

            #move to right
            if(x + 1 <= self.N):
                T[i,i+1] = pr
            else:
                T[i,i] += pr
            
            #move left
            if(x - 1 > 0):
                T[i,i-1] = pl
            else:
                T[i,i] += pl
            
            #move up
            if(y + 1 <= self.N):
                T[i,i+self.N] = pu
            else:
                T[i,i] += pu

            #move down
            if(y - 1 > 0):
                T[i,i-self.N] = pd
            else:
                T[i,i] += pd
        return T

    def calculate_sensor_probabilities(self):
        sensor_probabilities = []
        for sensor_idx, B in enumerate(self.B_list):
            prob_grid = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    state_index = i * self.N + j
                    prob_grid[i, j] = B[state_index]
            sensor_probabilities.append(prob_grid)
        return sensor_probabilities
    
    
    @staticmethod
    def binary_to_int(bin):
       return bin[0] * 8 + bin[1] * 4 + bin[2] * 2 + bin[3]
    
    def get_trajectory(self,length,seed=None):
        if seed is not None:
            np.random.seed(seed)
        start=(1,1)
        traj=[start]
        probs = np.array(self.T_prob)
        cum_probs = np.cumsum(probs)
        for i in range(length-1):
            x,y=traj[-1]
            rand = np.random.rand()

            if rand < cum_probs[0] and x < self.N:
                next_pos = (x + 1, y)
            elif rand < cum_probs[1] and x > 1: 
                next_pos = (x - 1, y)
            elif rand < cum_probs[2] and y < self.N: 
                next_pos = (x, y + 1)
            elif rand < cum_probs[3] and y > 1: 
                next_pos = (x, y - 1)
            else:
                next_pos = (x, y)
            traj.append(next_pos)
        return traj
    
    def sample_tranjectory_reading(self,traj,seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        traj_sensor_reading=[]
        for state in traj:
            sensor_read=[]
            for sensor_prob in [self.lamda.sensor_1_prob, self.lamda.sensor_2_prob, self.lamda.sensor_3_prob, self.lamda.sensor_4_prob]:
                prob=sensor_prob(state[0],state[1])
                reading = np.random.rand() < prob
                if(reading):
                    sensor_read.append(1)
                else:
                    sensor_read.append(0)
            traj_sensor_reading.append(sensor_read)
        return traj_sensor_reading

    def sample(self, t):
        '''
        Args:
            t: integet, trajectory length
        '''
        pass
        
    def forward_inference(self, obs):
        '''
        Args:
            obs: np.array of shape [T]
        '''
        
    def e_step(self, alpha, beta):
        '''
        Args:
            alpha: [R, T, self.n]
            beta: [R, T, self.n]
        '''
        pass
    
    def m_step(self, ksi, gamma):
        '''
        Args:
            T: Transition matrix [self.n , self.n]
            B: Emission matrix [self.n, 16]
            rho: Rho [self.n]
            obs: R observation sequnece [R, t]
        '''
        pass
                     
    def baum_welch(self, obs):
        '''
        Args:
            obs: [R, T]
        '''
        pass
    
    def viterbi(self, obs):
        '''
        Args:
            obs: [R, T]
        '''
        pass

