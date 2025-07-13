
import numpy as np
from PIL import Image 

class Grid:

    def __init__(self, M=15, N=15, P = 2, W = 21, B = 3):
        
        self.B = B 
        self.color = [1, 1, 1]
        self.brightness = 1.8
        self.grid = []
        self.P = P
        self.W = W 
        self.M = M 
        self.N = N
        self.ew = self.W + P//2
        self.grid = self.generate_grid()
    
    def generate_grid(self):
        grid = []
        for i in range(self.M):
            row_i = []
            for j in range(self.N):
                grid_ij = np.ones((self.W+self.P, self.W+self.P, 3))
        
                for c in range(3):
                    grid_ij[:,:,c] = self.color[c]*(0.7 + (self.brightness - 0.9)/60*(i+30-j))
        
                #make white
                grid_ij[:self.P//2,:,:] = 0
                grid_ij[-self.P//2:,:,:] = 0

                grid_ij[:,:self.P//2,:] = 0
                grid_ij[:,-self.P//2:,:] = 0
                
                #append to row_i
                row_i.append(grid_ij)
        
            #append to grid
            grid.append(row_i)
        return grid

    def get_grid(self, x, y):
        grid = self.grid[self.M - y - 1][x]
        return grid 

    def draw_one_step_grid(self, sp, dp, grid, color):
        #print(sp, dp)
        if(sp[0] == dp[0]):

            grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],:] *= 0
            for c in range(3):
                grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],c] = color[c]
            #print(grid[sp[0]-self.P//2:dp[0] + self.P//2, sp[1]:dp[1],0].shape)
        else:
            grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,:] *= 0
            for c in range(3):
                grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,c] = color[c]
            #print(grid[sp[0]:dp[0], sp[1]-self.P//2:dp[1] + self.P//2,0].shape)
        
    def draw_one_step(self, sp, dp, color):

        delta_x = dp[0] - sp[0] 
        delta_y = dp[1] - sp[1]
        
        sgrid = self.get_grid(sp[0]-1, sp[1]-1) 
        dgrid = self.get_grid(dp[0]-1, dp[1]-1)
        if(delta_x == delta_y):
            return
        if(delta_x == -1):
            self.draw_one_step_grid((self.ew//2 + 1,0),(self.ew//2 + 1, self.ew//2+1), sgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.ew//2+1,self.W + self.P), dgrid, color)
            
        elif(delta_x == 1):
            self.draw_one_step_grid((self.ew//2+1, 0),(self.ew//2 + 1, self.ew//2 + 1), dgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.ew//2+1, self.W + self.P), sgrid, color)
        elif(delta_y == -1):
            self.draw_one_step_grid((0 ,self.ew//2 + 1),( self.ew//2 + 1, self.ew//2+1), dgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.W + self.P, self.ew//2+1), sgrid, color)
            
        elif(delta_y == 1):
            self.draw_one_step_grid((0, self.ew//2+1),(self.ew//2 + 1, self.ew//2 + 1), sgrid, color)
            self.draw_one_step_grid((self.ew//2 + 1, self.ew//2 + 1),(self.W + self.P,self.ew//2+1), dgrid, color)
            
        return

    def draw_path(self, sequence, color = [0,0,1]):

        start_x, start_y = sequence[0]
        start_grid = self.get_grid(start_x-1,start_y-1)
        start_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2] *= 0
        start_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2,0] += 1
        

        end_x, end_y = sequence[-1]
        end_grid = self.get_grid(end_x-1,end_y-1)
        end_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2] *= 0
        end_grid[self.P//2:-1*self.P//2,self.P//2:-1*self.P//2,1] += 1
        
        for i in range(1,len(sequence)):
            self.draw_one_step(sequence[i-1],sequence[i], color)
        
        
    
    def show(self, path = "demo.png"):

        grid = [np.concatenate(row_i, axis = 1) for row_i in self.grid]
        grid = np.concatenate(grid, axis = 0)
        grid = np.clip(grid, 0, 1)
        grid_big = np.zeros((grid.shape[0] + 2*self.B, grid.shape[1] + 2*self.B, 3))
        grid_big[:,:,0] = 1
        grid_big[:,:,1] = 1
        grid_big[self.B:-1*self.B,self.B:-1*self.B] = grid

        grid = (grid_big*255).astype('uint8')

        grid = Image.fromarray(grid)
        grid.save(path)
    
    def clear(self):
        self.grid = self.generate_grid()

