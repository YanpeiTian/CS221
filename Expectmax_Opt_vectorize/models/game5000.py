import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random

class NN2048(nn.Module):
    def __init__(self, input_size=16, filter1=512, filter2=4096, drop_prob=0.):
        super(NN2048, self).__init__()
        self.conv_a = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,1), padding=0)
        self.conv_b = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,2), padding=0)
        self.conv_aa = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_ab = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.conv_ba = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_bb = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.relu = nn.ReLU()
        self.W_aa = nn.Linear(filter2 * 8, 1)
        self.W_ab = nn.Linear(filter2 * 9, 1)
        self.W_ba = nn.Linear(filter2 * 9, 1)
        self.W_bb = nn.Linear(filter2 * 8, 1)

    def flatten(self, x):
        N = x.size()[0]
        return x.view(N, -1)
        
    def forward(self, x):
        x = x.float()
        a = self.relu(self.conv_a(x))
        b = self.relu(self.conv_b(x))
        aa = self.flatten(self.relu(self.conv_aa(a)))
        ab = self.flatten(self.relu(self.conv_ab(a)))
        ba = self.flatten(self.relu(self.conv_ba(b)))
        bb = self.flatten(self.relu(self.conv_bb(b)))
        out = self.W_aa(aa) + self.W_ab(ab) + self.W_ba(ba) + self.W_bb(bb)
        return out

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
                v = (s - game_score) + p[i].item()
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
            loss = loss_fn(pred, y) / 2
            last_loss = loss.item()
            loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 10.0) #
            optimizer.step()
            model.eval()
#             if game_len % 50 == 0:
#                 print (game_len, last_loss)
                
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


lr = 5e-4
weight_decay = 0
beta1 = 0.9

model = NN2048().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, 0.999))
loss=nn.MSELoss()

import os
experiment_dir = ""

def save_model(state, filename='model.pth.tar'):
    filename = os.path.join(experiment_dir, filename)
    torch.save(state, filename)

save_model({
    'epoch': 0,
    'state_dict': model.cpu().state_dict(),
    'optimizer': optimizer.state_dict(),
})
model.cuda()

num_epochs = 5000

def train(model, optimizer, loss):
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        game_len, max_score, game_score, last_loss = gen_sample_and_learn(model, optimizer, loss, True, 0)
        print ('Game # ', epoch, 'Game length ', game_len, 'Max score ', max_score, 'Game score ', game_score, flush=True)
        if epoch % 500 == 0:
            filename = "model_"+str(epoch)+".pth.tar"
            save_model({
                'epoch': epoch,
                'state_dict': model.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)
            model.cuda()
    
train(model, optimizer, loss)


num_epochs = 100

def test(model):
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        game_len, max_score, game_score, last_loss = gen_sample_and_learn(model, None, None, False)
        print ('Game # ', epoch, 'Game length ', game_len, 'Max score ', max_score, 'Game score ', game_score, flush=True)

test(model)