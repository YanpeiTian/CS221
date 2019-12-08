import random
import numpy as np
import time
import ExpectiMax
import logic

ITERATION=2
DEPTH=3
stat=[]

def main():
    for iteration in range(ITERATION):
        print("Iteration: "+str(iteration+1))
        start=time.time()
        step=0
        matrix=logic.new_game()
        matrix=logic.add_two(matrix)
        while True:
            matrix=logic.add_two(matrix)
            if logic.gameOver(matrix):
                break
            # print("given this board")
            # logic.printBoard(matrix)
            move=ExpectiMax.getMove(matrix,DEPTH)
            matrix=ExpectiMax.moveGrid(matrix,move)
            # print("expectimax recommends this move " + str(move))
            # print("resulting in this board")
            # logic.printBoard(matrix)
            step+=1

        print("Step= "+str(step))
        print("Max= "+str(2**logic.getMax(matrix)))
        print("Score= "+str(logic.getScore(matrix)))
        print('Depth= '+str(DEPTH))
        print('Time= '+str(time.time()-start))
        print('')
        # stat.append((step,2**np.max(matrix),logic.getScore(matrix)))

if __name__ == '__main__':
    main()
