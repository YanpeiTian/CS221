import random
import numpy as np
import logic

FAIL_SCORE = -1e10
moves = ["'w'", "'a'", "'s'", "'d'"]


def getMove(grid, numBackgroundRuns):
    # index = direction
    # make n background runs for each direction, follow random policy until game ends
    # group by initial direction
    # calculate avg of each of the 4 directions
    # return the direction with the max avg score
    # estimated Q_pi(s, a) = average of utility where prev state = s, action = a (Page 29 mdp2 lecture)

    # index 0 = up avg steps, 1 = left avg, 2 = down avg, 3 = right avg
    listOfSteps = []
    for i in range(4):
        # print("current direction is " + str(i))
        # if initial direction is 'i'
        grids = [moveGrid(grid, i)] * numBackgroundRuns
        # get a list of steps it took to reach end state if initial direction is i
        stepsTaken = randomPolicyUntilGameOver(grids)
        # print("returned from randomPolicyUntilGameOver, i is " + str(i))
        listOfSteps.append(stepsTaken)

    avgSteps = []
    for i in range(4):
        avgSteps.append(sum(listOfSteps[i]) / numBackgroundRuns)

    index = [i for i in range(4) if avgSteps[i] == max(avgSteps)]
    return random.choice(index)

def moveGrid(grid,i):
    new=np.zeros((4,4),dtype=np.int)
    # print("input grid is ")
    # print(grid)
    # print("direction is up")
    if i==0:
        # move up
        grid=np.transpose(grid)
        # print("grid transpose is ")
        # print(grid)
        for row in range(4):
            new[row,:]=logic.move(grid[row,:])
            # print("row by row printing new grid")
            # print(new)
        new=np.transpose(new)
        # print("printing new grid")
        # print(new)
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

# return a list of steps taken
def randomPolicyUntilGameOver(grids):
    stepsTaken = []
    for i in range(len(grids)):
        steps = 0
        while logic.game_state(grids[i]) != 'lose':
            grids[i], lost = randomMove(grids[i])
            steps += 1
            if lost:
                break
            grids[i], _ = logic.add_two(grids[i])
            # print("moving grid with random policy")
            # print(grids[i])
        stepsTaken.append(steps)
        # print("move grid game over, returning from randomPolicyUntilGameOver")
    return stepsTaken

# return the tuple (new grid, game over boolean) after making one random move
def randomMove(grid):
    prevGrid = grid
    if logic.game_state(grid) == 'lose':
        return grid, True
    while np.all(prevGrid == grid):
        grid = moveGrid(grid, random.choice(range(4)))
    return grid, logic.game_state(grid) == 'lose'
