import numpy as np 
class lambda_1:

    def sensor_1_prob(self,i, j):
        if 1 <= i <= 9 and 1 <= j <= 9:
            return (18 - (i-1) - (j-1)) / 18
        return 0
    
    def sensor_2_prob(self,i, j):
        if 1 <= i <= 9 and 7 <= j <= 15:
            return (18 - (i-1) + (j-15)) / 18
        return 0
    
    def sensor_3_prob(self,i, j):
        if 7 <= i <= 15 and 7 <= j <= 15:
            return (18 + (i-15) + (j-15)) / 18
        return 0

    def sensor_4_prob(self,i, j):
        if 7 <= i <= 15 and 1 <= j <= 9:
            return (18 + (i-15) - (j-1)) / 18
        return 0

    def __init__(self) -> None:
        
        self.N=15
        self.T_prob = [0.4, 0.1, 0.3, 0.1, 0.1]
        
        B_list = []

        for sensor_prob in [self.sensor_1_prob, self.sensor_2_prob, self.sensor_3_prob, self.sensor_4_prob]:
            Bi = np.zeros(self.N*self.N)
            for i in range(1, self.N+1):
                for j in range(1, self.N+1):
                    index = (i-1)*self.N + (j-1)
                    Bi[index] = sensor_prob(i, j)
            B_list.append(Bi)

        self.B_list=B_list
    
   

       