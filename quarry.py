import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import timeit, functools

class Quarry:
    
    def __init__(self, N):
        self.N=N # size of environment
        self.quarry=np.zeros((N, N)) # Numpy array that holds the information about the environment
        self.quarry[0]=self.quarry[-1]=self.quarry[:, 0]=self.quarry[:, -1]= 1 #put obstacles in each border cell
        flat_quarry=self.quarry.flatten() #flatten so as to allow picking random indices to put in obstacles and resources
        inner_quarry_inds=np.asarray(np.where(flat_quarry!=1)).flatten() #define inner_quarry indices (i.e. not including borders)
        rand_inds = np.random.choice(inner_quarry_inds, size=(2*(self.N//2)))  #pick random indices of the 'inner quarry' 
        for i in rand_inds[int((len(rand_inds)/2)):]:
            flat_quarry[i]=1
        for i in rand_inds[:int((len(rand_inds)/2))]:
            flat_quarry[i]=2
        self.quarry=flat_quarry.reshape(self.quarry.shape).astype('int')#reshape quarry back to original shape 
        self.start_quarry = self.quarry.copy() # create a copy to restore quarry cells after agent moves
        self.original_quarry = self.quarry.copy()
        self.resource_load = None # shows how much resources is currently on board the agent
        self.resource_delivered = 0 # shows what value of resource has been delivered to base
        self.resource_target = 10*self.N//4 # determines what value of resource must be delivered to base to terminate
        self.done=False # termination condition indicator
        self.base_visits=0 # record how many visits to base are made
        
             
        index_states = np.arange(0, N*N)#a random index for each coordinate of the grid   
        np.random.shuffle(index_states)
        self.coord_to_index_state = index_states.reshape(N,N)   

    def step(self, action):
        reward=0
        old_i, old_j = self.position_agent # save as another var in case of obstacle
        i,j = self.position_agent
        e,x = self.position_base
        previous_location_status=int(self.quarry[old_i, old_j])
        if action =='up': i-=1   # define actions 
        if action =='down': i+=1      
        if action =='left': j-=1    
        if action =='right': j+=1      
        self.position_agent = i,j # modify the position of the agent
        
        '''calculate total reward'''
        location_type=int(self.quarry[i,j])
        if location_type==0:#free cell
            pass
        if location_type==1: #obstacle
            reward-= 5
        if location_type==2: #resource
            rand = random.random()
            if rand>0.7: # stochastic chance of resource contamination
                reward-=self.resource_load
                self.resource_load=0
            else:    
                reward+=10
                self.resource_load+=10
                self.resource_mined=True
        if location_type==3: #base when agent not already in base
            if self.resource_load>0:
                self.base_visits+=1
            reward+=self.resource_load**2
            self.resource_delivered+=self.resource_load
            self.resource_load=0
        reward-=1 # minus one for each time step
            
        '''restore previous cell status'''
        self.quarry=self.start_quarry.copy()
        if location_type==1: # if obstacle
            self.position_agent = old_i, old_j # go back to previous position.
            self.quarry[old_i, old_j]=4
        elif self.resource_mined:
            self.start_quarry[i,j]=0 # resource has been collected, show 0 when agent moves from this location
            self.quarry[i,j]=4 # agent occupies location
            self.position_agent = i,j # modify the position of the agent
            self.resource_mined=False
        else:
            self.position_agent = i,j 
            self.quarry[i,j]=4
        self.quarry[e,x]=3
        
        '''calculate observations'''
        self.target = np.asarray(self.position_base) -  np.asarray(self.position_agent) #relative co-ordinates of the exit
        i,j = self.position_agent
        self.proximity = self.quarry[i-1:i+2,j-1:j+2] # not used in simple quarry
        
        # update time
        self.t+=1

        # verify termination condition
        self.done=False
        if location_type==3 and self.resource_delivered >= self.resource_target:
            self.done=True
        if self.t==self.time_limit:
            self.out_of_time==True
            self.done=True
        
        state_prime = self.position_agent # in this particular environment the state is the same thing as the position i,j in the grid 
        state_prime = self.coord_to_index_state[state_prime[0], state_prime[1]]
    
        return state_prime, reward, self.done
    
    
    def reset(self):
        self.base_visits=0
        self.done=False
        self.resource_mined=False
        self.out_of_time=False
        self.resource_load = 0
        flat_quarry=self.original_quarry.copy().flatten() #flatten so as to allow picking a random index for agent start position
        inner_quarry_inds=np.asarray(np.where(flat_quarry==0)).flatten()#define inner_quarry indices (i.e. not including borders)
        start_exit_inds = np.random.choice(inner_quarry_inds, size=1, replace=False)#pick random index of the 'inner quarry' to be the base
        start_ind = start_exit_inds[0]
        flat_quarry[start_ind]=4 # 4 denotes the location of the base, where the agent starts and finishes
        self.quarry=flat_quarry.reshape(self.quarry.shape)
        self.position_agent = np.argwhere(self.quarry==4).reshape(2)
        self.position_base = self.position_agent 
        
        # Calculate observations
        self.target = self.position_base - self.position_agent #relative co-ordinates of the base
        i,j = self.position_agent
        self.proximity = self.quarry[i-1:i+2,j-1:j+2] # not used in the simple quarry
        self.reward=0
        
        # run time
        self.time_limit = self.N*10
        self.t=0

        state = self.coord_to_index_state[ self.position_agent[0], self.position_agent[1]]
        return state
    

    def show_env(self): # shows a visual representation of the environment, together with some status indicators
        plt.imshow(self.quarry, cmap='rainbow')
        for i in range(self.quarry.shape[0]):
            for j in range(self.quarry.shape[1]):
                if self.quarry[i,j]==2:
                    plt.text(j, i, "{}".format('R'), ha="center", va="center", label='Resource')
                if self.quarry[i,j]==3:
                    plt.text(j, i, "{}".format('Base'), ha="center", va="center", label='Base')
                if self.quarry[i,j]==4:
                    plt.text(j, i, "{}".format('Agent'), ha="center", va="center", label='Agent')
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,
            left=False,
            labelleft=False,# ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        plt.tight_layout()
        plt.title('Time Step = '+str(self.t)+'\nResources Mined Value='+ str(self.resource_load)+'\nResources Delivered='+str(self.resource_delivered), loc='left')
        plt.xlabel('\n R=Resource, Turquoise=Obstacle', loc='left')
        plt.xlabel('\n R=Resource, Turquoise=Obstacle', loc='left')