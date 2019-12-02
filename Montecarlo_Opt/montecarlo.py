import random
import logic

def getMove(grid, numBackgroundRuns):
    ls = []
    for i in range(4):
        nextGrid = logic.moveGrid(grid, i)
        score = 0
        if not logic.isSame(grid, nextGrid):
            score = randomPolicyUntilGameOver(nextGrid, numBackgroundRuns)
        ls.append((score, i, nextGrid))

    return random.choice([grid for score, _, grid in ls if score == max(ls)[0]])


def randomPolicyUntilGameOver(grid, numBackgroundRuns):
    score = 0
    for i in range(numBackgroundRuns):
        curGrid = grid.copy()
        while True:
            curGrid = logic.add_two(curGrid)
            board_list = randomMove(curGrid)
            if not board_list:
                break
            curGrid = random.choice(board_list)
        score += logic.getScore(curGrid)
    return score


def randomMove(grid):
    board_list = []
    for i in range(4):
        newGrid = logic.moveGrid(grid, i)
        if not logic.isSame(grid, newGrid):
            board_list.append(newGrid)
    return board_list