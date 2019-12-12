#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import copy


# In[2]:


class NN2048(nn.Module):
    def __init__(self, input_size=16, filter1=512, filter2=4096, drop_prob=0.):
        super(NN2048, self).__init__()
        self.conv_a = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,1), padding=0)
        self.conv_b = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,2), padding=0)
        self.conv_c = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,2), padding=0)
        
        self.conv_aa = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_ab = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.conv_ba = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_bb = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.conv_cc = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,2), padding=0)
        
        self.relu = nn.ReLU()
        self.W_aa = nn.Linear(filter2 * 8, 1)
        self.W_ab = nn.Linear(filter2 * 9, 1)
        self.W_ba = nn.Linear(filter2 * 9, 1)
        self.W_bb = nn.Linear(filter2 * 8, 1)
        self.W_cc = nn.Linear(filter2 * 4, 1)

    def flatten(self, x):
        N = x.size()[0]
        return x.view(N, -1)
        
    def forward(self, x):
        x = x.float()
        a = self.relu(self.conv_a(x))
        b = self.relu(self.conv_b(x))
        c = self.relu(self.conv_c(x))
        aa = self.flatten(self.relu(self.conv_aa(a)))
        ab = self.flatten(self.relu(self.conv_ab(a)))
        ba = self.flatten(self.relu(self.conv_ba(b)))
        bb = self.flatten(self.relu(self.conv_bb(b)))
        cc = self.flatten(self.relu(self.conv_cc(c)))
        out = self.W_aa(aa) + self.W_ab(ab) + self.W_ba(ba) + self.W_bb(bb) + self.W_cc(cc)
        return out


# In[3]:


def make_input(grid):
    r = np.zeros(shape=(16, 4, 4))
    for i in range(4):
        for j in range(4):
            r[grid[i, j],i, j]=1
    return r

def add_two(mat):
    indexs=np.argwhere(mat==0)
    index=np.random.randint(0,len(indexs))
    mat[tuple(indexs[index])] = 1
    return mat


# In[4]:


singleScore=[0,0,4,16,48,128,320,768,1792,4096,9216,20480,45056,98304,212992,458752,983040]
moveDict=np.load('move.npy')

def move(list):
    return moveDict[list[0],list[1],list[2],list[3],:]

def lookup(x):
    return singleScore[x]

lookup = np.vectorize(lookup)

def getScore(matrix):
    return np.sum(lookup(matrix))

def getMove(grid):
    board_list = []
    for i in range(4):
        newGrid=moveGrid(grid, i)
        if not isSame(grid,newGrid):
            board_list.append((newGrid, i, getScore(newGrid)))
    return board_list
        
def moveGrid(grid,i):
    # new=np.zeros((4,4),dtype=np.int)
    new = None
    if i==0:
        # move up
        grid=np.transpose(grid)
        new = np.stack([move(grid[row,:]) for row in range(4)], axis = 0).astype(int).T
    elif i==1:
        # move left
        new = np.stack([move(grid[row,:]) for row in range(4)], axis = 0).astype(int)
    elif i==2:
        # move down
        grid=np.transpose(grid)
        new = np.stack([np.flip(move(np.flip(grid[row,:]))) for row in range(4)], axis = 0).astype(int).T
    elif i==3:
        # move right
        new = np.stack([np.flip(move(np.flip(grid[row,:]))) for row in range(4)], axis = 0).astype(int)
    return new

def isSame(grid1,grid2):
    return np.all(grid1==grid2)


# In[5]:


def Vchange(grid, v):
    g0 = grid
    g1 = g0[:,::-1,:]
    g2 = g0[:,:,::-1]
    g3 = g2[:,::-1,:]
    r0 = grid.swapaxes(1,2)
    r1 = r0[:,::-1,:]
    r2 = r0[:,:,::-1]
    r3 = r2[:,::-1,:]
    xtrain = np.array([g0,g1,g2,g3,r0,r1,r2,r3])
    ytrain = np.array([v]*8)
    return xtrain, ytrain

def gen_sample_and_learn(model, optimizer, loss_fn, is_train = False, explorationProb=0.1):
    model.eval()
    game_len = 0
    game_score = 0
    last_grid1 = np.zeros((4,4),dtype=np.int)
    last_grid1 = add_two(last_grid1)
    last_grid2 = make_input(last_grid1)
    last_loss = 0

    while True:
        grid_array = add_two(last_grid1)
        board_list = getMove(grid_array)
        if board_list:
            boards = np.array([make_input(g) for g,m,s in board_list])
            p = model(torch.from_numpy(boards).cuda()).flatten().detach()        
            game_len += 1
            best_v = None
            for i, (g,m,s) in enumerate(board_list):
                v = 1 + p[i].item()
                if best_v is None or v > best_v:
                    best_v = v
                    best_score = s
                    best_grid1 = board_list[i][0]
                    best_grid2 = boards[i]
                    
        else:
            best_v = 0
            best_grid1 = None
            best_grid2 = None
            
        if is_train:
            x, y = Vchange(last_grid2, best_v)
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).unsqueeze(dim=1).cuda().float()
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            last_loss = loss.item()
            loss.backward()
            optimizer.step()
            model.eval()
                
        if not board_list:
            break
            
        # gibbs sampling or espilon-greedy
        if is_train and random.random() < explorationProb:
            idx = random.randint(0, len(board_list) - 1)
            game_score = board_list[idx][2]
            last_grid1 = board_list[idx][0]
            last_grid2 = boards[idx]
        else:
            game_score = best_score
            last_grid1 = best_grid1
            last_grid2 = best_grid2
        
    return game_len, 2**grid_array.max(), game_score, last_loss


# In[6]:


lr = 1e-3
weight_decay = 0
beta1 = 0.9

model = NN2048().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, 0.999))
loss=nn.MSELoss()


# In[7]:


import os
experiment_dir = ""

def save_model(state, filename='model.pth.tar'):
    filename = os.path.join(experiment_dir, filename)
    torch.save(state, filename)
    
def load_model(model, optimizer, filename, model_only = False):
    checkpoint_path = os.path.join(experiment_dir, filename)
    ckpt_dict = torch.load(checkpoint_path, map_location="cuda:0")

    model.load_state_dict(ckpt_dict['state_dict'])
    if not model_only:
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        epoch = ckpt_dict['epoch']
        running_mean = ckpt_dict['running_mean']
    else:
        epoch = None
        running_mean = None
    return model, optimizer, epoch, running_mean


# In[ ]:


num_epochs = 1000
best_model = NN2048().cuda()
model500 = NN2048().cuda()

def train(model, optimizer, loss, epoch = 0, running_mean = 2048):
    ls = [1024] * 10
    while epoch != num_epochs:
        epoch += 1
        game_len, max_score, game_score, last_loss = gen_sample_and_learn(model, optimizer, loss, True, 0)
        print ('Game # ', epoch, 'Game length ', game_len, 'Max score ', max_score, 'Game score ', game_score, flush=True)
        ls.pop(0)
        ls.append(max_score)
        if sum(ls) / 10 > running_mean or epoch % 500 == 0:
            running_mean = sum(ls) / 10
            name = epoch if epoch % 500 == 0 else epoch // 100
            filename = "model_step_"+str(name)+".pth.tar"
            save_model({
                'epoch': epoch,
                'running_mean': running_mean,
                'state_dict': model.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)
            model.cuda()
            if epoch == 500:
                model500 = copy.deepcopy(model)
            elif epoch % 500 != 0:
                best_model = copy.deepcopy(model)
            
train(model, optimizer, loss)


# In[ ]:


num_epochs = 100

def test(model):
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        game_len, max_score, game_score, last_loss = gen_sample_and_learn(model, None, None, False)
        print ('Game # ', epoch, 'Game length ', game_len, 'Max score ', max_score, 'Game score ', game_score, flush=True)


# In[ ]:

print ("Test of model at 1000 epoch")
test(model)

print ("Test of model at 500 epoch")
model500.cuda()
test(model500)

print ("Test of best train model")
best_model.cuda()
test(best_model)


# In[ ]:




