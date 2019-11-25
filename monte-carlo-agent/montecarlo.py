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
        # if initial direction is 'i'
        grids = [moveGrid(grid, i)] * numBackgroundRuns
        # get a list of steps it took to reach end state if initial direction is i
        stepsTaken = randomPolicyUntilGameOver(grids)
        listOfSteps.append(stepsTaken)

    avgSteps = []
    for i in range(4):
        avgSteps.append(sum(listOfSteps[i]) / numBackgroundRuns)

    index = [i for i in range(4) if avgSteps[i] == max(avgSteps)]
    return moves[random.choice(index)]

def moveGrid(grid, i):
    if i == 0:
        new, _ = logic.up(grid)
        return new
    if i == 1:
        new, _ = logic.left(grid)
        return new
    if i == 2:
        new, _ = logic.down(grid)
        return new
    if i == 3:
        new, _ = logic.right(grid)
        return new

def isSame(grid1, grid2):
    for i in range(4):
        for j in range(4):
            if grid1[i][j] != grid2[i][j]:
                return False
    return True

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
            grids[i] = logic.add_two(grids[i])
        stepsTaken.append(steps)
        #     print("moving grid with random policy")
        #     print(grids[i])
        # print("move grid game over, returning from randomPolicyUntilGameOver")
    return stepsTaken

# return the tuple (new grid, game over boolean) after making one random move
def randomMove(grid):
    prevGrid = grid
    if logic.game_state(grid) == 'lose':
        return grid, True
    while prevGrid == grid:
        grid = moveGrid(grid, random.choice(range(4)))
    return grid, logic.game_state(grid) == 'lose'
