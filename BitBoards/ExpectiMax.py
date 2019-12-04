import random
import numpy as np
import logic

FAIL_SCORE=-1e10

heu=[13.5, 12.1, 10.2, 9.9,9.9, 8.8, 7.6, 7.2,6.0, 5.6, 3.7, 1.6,1.2, 0.9, 0.5, 0.3]

def getMove(grid,depth):
    # print("getting move")
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
    # print("expectimax move grid direction i " + str(i))
    return logic.move(grid, i)

def isSame(grid1,grid2):
    return grid1 == grid2

def getUtility(grid):
    # return np.dot(heu3,2**grid.flatten())
    # return np.dot(heu,logic.singleScore[logic.toList((grid))])
    return logic.getScore(grid)
    # return logic.getUtility(grid)

# grid:   Game grid
# depth:  search depth
# agent:  0 is player, 1 is computer
def expectiMax(grid,agent,depth):
    score=0

    if logic.gameOver(grid):
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
        # idea: board & 0xf to extract 4 bits
        # then check if 4 bits == 0, if ya, then make new board and recurse expectimax
        # right shift by 4
        board = grid
        offset = 0
        while board != 0:
            if board & 0xf == 0:
                count += 1
                # board |= 0x1 # add a 2 tile
                # newBoard = (board << offset) | grid
                newBoard = (1 << offset) | grid
                score += expectiMax(newBoard, 0, depth -1)
            board = board >> 4
            offset += 4
        # for i in range(4):
        #     for j in range(4):
        #         if(grid[i][j]==0):
        #             grid[i][j]=1
        #             count+=1
        #             score+=expectiMax(grid,0,depth - 1)
        #             grid[i][j]=0
        if count==0:
            return FAIL_SCORE
        else:
            return score/count
