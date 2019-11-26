import random
import numpy as np
import logic

FAIL_SCORE = -1e10
moves = ["'w'", "'a'", "'s'", "'d'"]


def getMove(grid, numBackgroundRuns):
    # index = direction
    # make n background runs for each direction, follow random policy until game ends
    # group by initial direction
    # calculate util of each of the 4 directions
    # return the direction with the max util
    # estimated Q_pi(s, a) = average of utility where prev state = s, action = a (Page 29 mdp2 lecture)

    # index 0 = up avg steps, 1 = left avg, 2 = down avg, 3 = right avg
    accumlatedScoresAllDirections = []
    for i in range(4):
        # initial direction is i, then make n copies of the grid afer moving in i direction
        grids = [moveGrid(grid, i)] * numBackgroundRuns
        scores = randomPolicyUntilGameOver(grids)
        accumlatedScoresAllDirections.append(sum(scores)) # index i => sum of accumulated scores if initial direction is i

    index = [i for i in range(4) if accumlatedScoresAllDirections[i] == max(accumlatedScoresAllDirections)]
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
        for row in range(4):
            new[row,:]=np.flip(logic.move(np.flip(grid[row,:])))
        new=np.transpose(new)
    if i==3:
        # move right
        for row in range(4):
            new[row,:]=np.flip(logic.move(np.flip(grid[row,:])))
    return new

# input: a list of n identical grids
# output: sum of scores at each step of each grid after following random policy until game over
def randomPolicyUntilGameOver(grids):
    scores = []
    # for each copy of the grid
    for i in range(len(grids)):
        score = 0
        # keep playing randomly until game is lost
        while logic.game_state(grids[i]) != 'lose':
            grids[i], lost = randomMove(grids[i]) # agent make wasd random move
            score += logic.getScore(grids[i]) # accumulate score
            if lost:
                break
            grids[i], _ = logic.add_two(grids[i]) # computer generates a 2
            # print("moving grid with random policy")
            # print(grids[i])
        scores.append(score)
    return scores

# return the tuple (new grid, game over boolean) after making one random move
def randomMove(grid):
    prevGrid = grid
    if logic.game_state(grid) == 'lose':
        return grid, True
    while np.all(prevGrid == grid):
        grid = moveGrid(grid, random.choice(range(4)))
    return grid, logic.game_state(grid) == 'lose'
