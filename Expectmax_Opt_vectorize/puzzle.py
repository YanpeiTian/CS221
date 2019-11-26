import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import ExpectiMax
import logic

ITERATION=10 #100
DEPTH=2
stat=[]

def main():
    for iteration in range(ITERATION):
        print("Iteration: "+str(iteration+1))
        start=time.time()
        step=0
        matrix=np.zeros((4,4),dtype=np.int)
        matrix=logic.add_two(matrix)
        while True:
            matrix=logic.add_two(matrix)
            if logic.game_state(matrix)=='lose':
                break

            move=ExpectiMax.getMove(matrix,DEPTH)
            matrix=ExpectiMax.moveGrid(matrix,move)
            step+=1

        print("Step= "+str(step))
        print("Max= "+str(2**np.max(matrix)))
        print("Score= "+str(logic.getScore(matrix)))
        print('Depth= '+str(DEPTH))
        print('Time= '+str(time.time()-start))
        print('')
        stat.append((step,2**np.max(matrix),logic.getScore(matrix)))

if __name__ == '__main__':
    main()
