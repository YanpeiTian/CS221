import random
import numpy as np
import logic

FAIL_SCORE=-1e10

heu=[13.5, 12.1, 10.2, 9.9,9.9, 8.8, 7.6, 7.2,6.0, 5.6, 3.7, 1.6,1.2, 0.9, 0.5, 0.3]

def getMove(grid,depth):
    scores=[]
    for i in range(4):
        newGrid=moveGrid(grid,i)
        if(isSame(grid,newGrid)==False):
            scores.append(expectiMax(newGrid,1,depth))
        else:
            scores.append(FAIL_SCORE-1)
    index=[i for i in range(len(scores)) if scores[i]==max(scores)]
    return random.choice(index)

def moveGrid(grid,i):
    new=np.zeros((4,4),dtype=np.int)
    if i==0:
        # move up
        grid=np.transpose(grid)
        for row in range(4):
            new[row,:]=logic.move(grid[row,:])
        new=np.transpose(new)
    if i==1:
        # move left
        for row in range(4):
            new[row,:]=logic.move(grid[row,:])
    if i==2:
        # move down
        grid=np.transpose(grid)
        grid=np.fliplr(grid)
        for row in range(4):
            new[row,:]=logic.move(grid[row,:])
        new=np.fliplr(new)
        new=np.transpose(new)
    if i==3:
        # move right
        grid=np.fliplr(grid)
        for row in range(4):
            new[row,:]=logic.move(grid[row,:])
        new=np.fliplr(new)
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
        return getUtility(grid)

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
        indexs=[]
        for i in range(4):
            for j in range(4):
                if(grid[i][j]==0):
                    indexs.append((i,j))

        if len(indexs)>4:
            indexs=random.sample(indexs,4)

        for index in indexs:
            grid[index]=1
            score+=expectiMax(grid,0,depth - 1)
            grid[index]=0

        if len(indexs)==0:
            return FAIL_SCORE
        else:
            return score/len(indexs)
