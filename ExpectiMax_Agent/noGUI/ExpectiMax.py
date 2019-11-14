import random
import numpy as np
import logic

FAIL_SCORE=-1e10
moves=["'w'","'a'","'s'","'d'"]

def getMove(grid,depth):
    scores=[FAIL_SCORE]*4

    for i in range(4):
        newGrid=np.array(moveGrid(grid,i))
        if(isSame(grid,newGrid)==False):

            score=expectiMax(newGrid,1,depth)
            scores[i]=score

    index=[i for i in range(4) if scores[i]==max(scores)]
    return moves[random.choice(index)]

def moveGrid(grid,i):
    if i==0:
        new,_=logic.up(grid)
        return new
    if i==1:
        new,_=logic.left(grid)
        return new
    if i==2:
        new,_=logic.down(grid)
        return new
    if i==3:
        new,_=logic.right(grid)
        return new

def isSame(grid1,grid2):
    for i in range(4):
        for j in range(4):
            if grid1[i][j]!=grid2[i][j]:
                return False
    return True

# for progress report, using this board layout to calculate utility
tileWeights = [[51.2, 32.6, 26.4, 20.2],
               [22.6, 18.8, 13.2, 10.8],
               [8.9,  5.1,  2.6,  1],
               [1.2,  1,    0.8,  .1]]
def getUtility(grid):
    multiplier = 1
    for r in range(4):
        for c in range(4):
            multiplier += grid[r][c] * tileWeights[r][c]
    return multiplier

# grid:   Game grid
# depth:  search depth
# agent:  0 is player, 1 is computer
def expectiMax(grid,agent,depth):
    score=0

    if(logic.game_state(grid)=='lose'):
        return FAIL_SCORE
    if depth==0:
        return logic.getScore(grid) * getUtility(grid)

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
                if(grid[i][j]==0):
                    grid[i][j]=2
                    count+=1
                    score+=expectiMax(grid,0,depth - 1)
                    grid[i][j]=0
        if count==0:
            return FAIL_SCORE
        else:
            return score/count