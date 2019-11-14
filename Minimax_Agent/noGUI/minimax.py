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

            score = minimax(newGrid, 1, depth) # computer starts first
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

# grid:   Game grid
# depth:  search depth
# agent:  0 is player, 1 is computer
# def minimax(grid, agent, depth):
#     score=0
#
#     if(logic.game_state(grid)=='lose'):
#         return FAIL_SCORE
#     if depth==0:
#         return logic.getScore(grid)
#
#     # Player's turn
#     if agent==0:
#         for i in range(4):
#             newGrid=moveGrid(grid,i)
#             score = max(score, minimax(newGrid,1,depth))
#         return score
#     # Computer's turn
#     elif agent==1:
#         count=0
#         score= float('inf')
#         for i in range(4):
#             for j in range(4):
#                 if(grid[i][j]==0):
#                     grid[i][j]=2
#                     count+=1
#                     score = min(score, minimax(grid,0,depth - 1))
#                     grid[i][j]=0
#         if count==0:
#             return FAIL_SCORE
#         else:
#             return score
def minimax(grid, agent, depth):
    def alphaBetaPruning(grid, agent, depth, alpha, beta):
        # base case
        if (logic.game_state(grid) == 'lose'):
            return FAIL_SCORE
        if depth == 0:
            return logic.getScore(grid)

        # Player's turn
        if agent == 0:
            score = 0
            for i in range(4):
                newGrid = moveGrid(grid, i)
                futureScore = alphaBetaPruning(newGrid, 1, depth, alpha, beta)
                # new score should be in range [alpha, beta]
                if futureScore > beta:
                    return futureScore
                alpha = max(alpha, futureScore) # update alpha/lower bound
                score = max(score, futureScore)
            return score
        # Computer's turn
        elif agent == 1:
            count = 0
            score = float('inf')
            for i in range(4):
                for j in range(4):
                    if (grid[i][j] == 0):  # for each cell with a 0
                        grid[i][j] = 2  # create a 2 tile
                        count += 1
                        futureScore = alphaBetaPruning(grid, 0, depth - 1, alpha, beta)  # agent goes next
                        # new score should be in range [alpha, beta]
                        if futureScore < alpha:
                            return futureScore
                        beta = min(beta, futureScore) # update beta/upper bound
                        score = min(score, futureScore)
                        grid[i][j] = 0  # reset the 2 tile back to 0
            if count == 0:  # if no cell with 0 val, fail
                return FAIL_SCORE
            else:
                return score

    score = alphaBetaPruning(grid, agent, depth, float('-inf'), float('inf'))
    return score