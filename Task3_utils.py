import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical

device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')

class CriticNet(nn.Module):#define network for critic

    def __init__(self, input_size, size_hidden, output_size):
        
        super().__init__()
        self.fc1 = nn.Linear(input_size,size_hidden)
        self.fc2 = nn.Linear(size_hidden,size_hidden)
        self.o = nn.Linear(size_hidden,output_size) ###think about it as a regression NN
        
        
    def forward(self, x):
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output  = self.o(h2) # single output as it's effectively a regression
  
        return output

class PolicyNet(nn.Module):

    def __init__(self, input_size, size_hidden, output_size):
        
        super().__init__()
        self.fc1 = nn.Linear(input_size,size_hidden)
        self.fc2 = nn.Linear(size_hidden,size_hidden)
        self.o = nn.Linear(size_hidden,output_size)
        
        
    def forward(self, x):
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output  = torch.softmax(self.o(h2), dim=1) # softmax to get the probabilities
  
        return output



def process_state(state):  #actually its obs that are the argument, but they are are concatenated here into a tensor state
    state = np.concatenate([state['relative_coordinates'], np.array(state['resource_load']).reshape(-1), np.array(state['resource_target_remaining']).reshape(-1), state['surroundings'].flatten()])
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)#needs to be a matrix
    return state

#create trajectories for step 3 of the PPO algo
def create_trajectory(env, critic_net, critic_optimizer, policy_net, policy_optimizer, clip_factor, max_len_episode):
    actions=['up', 'down', 'left', 'right']
    trajectory=[]
    state = env.reset()
    state = process_state(state)
    value = critic_net(state)  #get state value now in order to compute advantage value later
    done=False
    base_visits=0
    for i in range(max_len_episode): # use maximum episode length as limiter instead of time limit
        action_probs = policy_net(state)
        c = Categorical(action_probs) #use Categorical functionality for ease of coding, as recommended in labs
        actionidx = c.sample() # take a sample index from the categorial object
        log_prob = c.log_prob(actionidx) # obtain the log prob of the action sample from the the categorial object
        action = actions[actionidx.item()] # item gets value of action
        raw_state, reward, done = env.step(action) #take a step
        reward = np.clip(reward, -clip_factor, clip_factor)  #clip the rewards
        trajectory.append((state.detach().cpu(), value.detach().cpu(), actionidx.cpu(), log_prob.detach().cpu(), reward)) # append to trajectory list
        state = process_state(raw_state) # process state information into tensor concatenation to work with the neural networks
        value = critic_net(state) # update value for next iteration
        base_visits=env.base_visits
        if done: 
            break
    return trajectory, base_visits

def create_rewardsTG(rewards, gamma):#create rewards to go 
    rewardsTG=[]
    g_prev = 0
    for r in reversed(rewards):#reverse the rewards list to start at the end
        g = r+g_prev*gamma # calculate rewardTG by adding to next , discounted by gamma
        rewardsTG.append(g)  #add to list
        g_prev=g #reassign next reward
    rewards=list(reversed(rewardsTG)) # turn the list the right way around
    return rewards #create list from iterator

def norm_rewards(rewards): # redundant - not used
    return (rewards - rewards.mean()) / rewards.std() 

def train_policy(policy_net, policy_optimizer, eps, log_probs_k, adv, states, actionsidx): # eps normally 0.1 - 0.3
    
    adv = adv.to(device)
    actionsidx=actionsidx.to(device)
    probs = policy_net(states.to(device)) # forward()
    probs = torch.gather(probs, 1, actionsidx) # get the probs of the actions taken
    log_probs = torch.log(probs)   # get the log probs
    #log(a/b) = log(a)-log(b)
    #a/b = exp(log(a)-log(b))
    ratio = torch.exp(log_probs - log_probs_k.to(device)) #calculate ratio using properties of logs
    ratio_adv = ratio*adv
    clipped_adv = torch.where(adv>=0, (1+eps)*adv,(1-eps)*adv) # calculate clipped advantages
    loss = -(torch.min(ratio_adv, clipped_adv)).mean() # loss is the negative of the gain 
    policy_optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters(): #  Gradient clipping to make sure weights are between -1,1
        param.grad.data.clamp_(-1, 1) 
    policy_optimizer.step()
    
    return loss

def train_value(critic_net, critic_optimizer, rewardsTG, states):
    values = critic_net(states.to(device))
    loss = F.mse_loss(values, rewardsTG.to(device)) #calculate MSE loss
    critic_optimizer.zero_grad()
    loss.backward()
    for param in critic_net.parameters(): #  Gradient clipping to make sure weights are between -1,1
        param.grad.data.clamp_(-1, 1) 
    critic_optimizer.step()
    return loss

def PPO_step(env, critic_net, critic_optimizer, policy_net, policy_optimizer, clip_factor, gamma, eps, n_trajectories, max_len_episode):
    '''Delivers steps 3-8 of PPO Algo'''
    trajectories=[]
    base_visits_lst=[]
    for i in range(n_trajectories):#create trajectories
        trajectory, base_visits = create_trajectory(env, critic_net, critic_optimizer, policy_net, policy_optimizer, clip_factor, max_len_episode)
        trajectories.append(trajectory)
        base_visits_lst.append(base_visits)
    policy_losses=[]
    value_losses =[]
    mean_rewards =[]
    mean_n_steps = []
    mean_base_visits=[]
    for t, bv in zip(trajectories, base_visits_lst):  

        states, values, actions, log_probs_k, rewards = zip(*t) # unzip trajectory elements
        states=torch.cat(states)
        values=torch.cat(values)
        actions=torch.cat(actions).unsqueeze(1)
        log_probs_k = torch.cat(log_probs_k).unsqueeze(1) # reshaping to make it work
        rewardsTG = create_rewardsTG(rewards, gamma) # call rewards to go
        rewardsTG = torch.tensor(rewardsTG).float().unsqueeze(1)#reshape
        advantages = rewardsTG - values # calculate advantages
        
        #training
        policy_loss=train_policy(policy_net, policy_optimizer, eps, log_probs_k, advantages, states, actions)
        value_loss =train_value(critic_net, critic_optimizer, rewardsTG, states)
        policy_losses.append(policy_loss.detach().cpu().item())
        value_losses.append(value_loss.detach().cpu().item()) 
        mean_rewards.append(np.mean(rewards))
        mean_n_steps.append(len(rewards))
        bv=np.asarray(bv)
        mean_base_visits.append(bv) # redundant -base visits not reported
    return np.mean(policy_losses), np.mean(value_losses), np.mean(mean_rewards), np.mean(mean_n_steps), np.mean(mean_base_visits)

