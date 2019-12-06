import random
import numpy as np
import logic

FAIL_SCORE=-1e10

heu=[13.5, 12.1, 10.2, 9.9,9.9, 8.8, 7.6, 7.2,6.0, 5.6, 3.7, 1.6,1.2, 0.9, 0.5, 0.3]

def getMove(grid,depth):
    scores=np.ones(4) * (FAIL_SCORE-1)
    for i in range(4):
        newGrid=moveGrid(grid,i)
        if not isSame(grid,newGrid):
            scores[i] = expectiMax(newGrid,1,depth)
    return np.random.choice(np.where(scores == scores.max())[0])

def moveGrid(grid,i):
    # new=np.zeros((4,4),dtype=np.int)
    new = None
    if i==0:
        # move up
        grid=np.transpose(grid)
        new = np.stack([logic.move(grid[row,:]) for row in range(4)], axis = 0).astype(int).T
    elif i==1:
        # move left
        new = np.stack([logic.move(grid[row,:]) for row in range(4)], axis = 0).astype(int)
    elif i==2:
        # move down
        grid=np.transpose(grid)
        new = np.stack([np.flip(logic.move(np.flip(grid[row,:]))) for row in range(4)], axis = 0).astype(int).T
    elif i==3:
        # move right
        new = np.stack([np.flip(logic.move(np.flip(grid[row,:]))) for row in range(4)], axis = 0).astype(int)
    return new

def isSame(grid1,grid2):
    return np.all(grid1==grid2)

def getUtility(grid):
    return np.dot(heu,2**grid.flatten())
    # return np.dot(heu,singleScore[grid.flatten()])

# grid:   Game grid
# depth:  search depth
# agent:  0 is player, 1 is computer
def expectiMax(grid,agent,depth):
    score=0

    if(logic.game_state(grid)=='lose'):
        return FAIL_SCORE
    if depth==0:
        return logic.getScore(grid) #getUtility(grid)

    # Player's turn
    if agent==0:
        for i in range(4):
            newGrid=moveGrid(grid,i)
            score = max(score, expectiMax(newGrid,1,depth))
        return score
    # Computer's turn
    elif agent==1:
        count=0
        score=0
        for i in range(4):
            for j in range(4):
                if grid[i][j]==0:
                    grid[i][j]=1
                    count+=1
                    score+=expectiMax(grid,0,depth - 1)
                    grid[i][j]=0
        if count==0:
            return FAIL_SCORE
        else:
            return score/count
