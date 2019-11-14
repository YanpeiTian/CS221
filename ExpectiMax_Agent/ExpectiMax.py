import random
import numpy as np
import logic

FAIL_SCORE=-1e10
moves=["'w'","'a'","'s'","'d'"]

def getMove(grid,depth):
    scores=[FAIL_SCORE]*4

    # count=np.sum([1 for i in range(16) if grid[i//4][i%4]==0])
    # if count<5:
    #     depth+=1

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
            if grid1[i,j]!=grid2[i,j]:
                return False
    return True

###################### the following functions can be called in getUtility for the final report
###################### not being used anywhere yet

FOUR_SEQ = 32
THREE_SEQ = 8
TWO_SEQ = 4

# higher utility for monotonic rows like 64 32 8 2 or the other way around
def getRowUtility(grid, row):
    row = grid[row]
    prev = row[0]
    index = 1
    # decreasing sequence
    while index < 4 and row[index] <= prev:
        prev = row[index]
        index += 1
    bestSeq = index

    # increasing sequence
    prev = row[0]
    index = 1
    while index < 4 and row[index] >= prev:
        prev = row[index]
        index += 1

    bestSeq = max(bestSeq, index)


    if bestSeq == 4:
        return FOUR_SEQ
    elif bestSeq == 3:
        return THREE_SEQ
    elif bestSeq == 2:
        return TWO_SEQ
    else:
        return 1


# higher utilitiy for monotonic column
def getColUtility(grid, col):
    prev = grid[0][col]
    index = 1

    # decreasing sequence
    while index < 4 and grid[index][col] <= prev:
        prev = grid[index][col]
        index += 1
    bestSeq = index

    # increasing sequence
    prev = grid[0][col]
    index = 1
    while index < 4 and grid[index][col] >= prev:
        prev = grid[index][col]
        index += 1

    bestSeq = max(bestSeq, index)

    if bestSeq == 4:
        return FOUR_SEQ
    elif bestSeq == 3:
        return THREE_SEQ
    elif bestSeq == 2:
        return TWO_SEQ
    else:
        return 1

# the more empty tiles, the higher the reward
def getEmtpyTileUtility(grid):
    count = 0
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0:
                count += 1
    return count ** 2

############################################################################

# TODO: for final report, can use the following getUtility function to evaluate reward
# takes into account board layout and empty tiles,
# def getUtility(grid):
#     multiplier = 1
#     for i in range(4):
#         multiplier *= getRowUtility(grid, i)
#         multiplier *= getColUtility(grid, i)
#     multiplier *= getEmtpyTileUtility(grid)
#     return logic.getScore(grid) * multiplier


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
    if agent == 0:
        for i in range(4):
            newGrid = moveGrid(grid,i)
            score = max(score, expectiMax(newGrid,1,depth))
        return score
    # Computer's turn
    elif agent == 1:
        count = 0
        score = 0
        for i in range(4):
            for j in range(4):
                if grid[i][j] == 0:
                    grid[i][j] = 2
                    count += 1
                    score += expectiMax(grid,0,depth - 1)
                    grid[i][j] = 0
        if count==0:
            return FAIL_SCORE
        else:
            return score/count
