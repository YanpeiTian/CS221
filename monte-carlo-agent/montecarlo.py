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
    scoresAllDirections = []
    for i in range(4):
        # initial direction is i, then make n copies of the grid afer moving in i direction
        grids = [moveGrid(grid, i)] * numBackgroundRuns
        scores = randomPolicyUntilGameOver(grids)
        scoresAllDirections.append(sum(scores))

    index = [i for i in range(4) if scoresAllDirections[i] == max(scoresAllDirections)]
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

# input: a list of n identical grids
# output: sum of scores at each step of each grid after following random policy until game over
def randomPolicyUntilGameOver(grids):
    scores = []
    # for each copy of the grid
    for i in range(len(grids)):
        score = 0
        # keep playing randomly until game is lost
        while logic.game_state(grids[i]) != 'lose':
            grids[i], lost = randomMove(grids[i])
            score += logic.getScore(grids[i])
            if lost:
                break
            grids[i] = logic.add_two(grids[i])
        scores.append(score)
    return scores

# return the tuple (new grid, game over boolean) after making one random move
def randomMove(grid):
    prevGrid = grid
    if logic.game_state(grid) == 'lose':
        return grid, True
    while prevGrid == grid:
        grid = moveGrid(grid, random.choice(range(4)))
    # print(prevGrid)
    # print(grid)
    return grid, logic.game_state(grid) == 'lose'
