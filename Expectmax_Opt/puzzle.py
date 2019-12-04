import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import ExpectiMax
import logic
import multiprocessing

ITERATION=100
DEPTH=4

def iteration(iter):
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

        print("Iteration: "+str(iter+1))
        print("Step= "+str(step))
        print("Max= "+str(2**np.max(matrix)))
        print("Score= "+str(logic.getScore(matrix)))
        print('Depth= '+str(DEPTH))
        print('Time= '+str(time.time()-start))
        print('')

        return (step,2**np.max(matrix),logic.getScore(matrix))


def main():
    p=multiprocessing.Pool(processes=ITERATION)
    data=p.map(iteration,[i for i in range(ITERATION)])
    p.close()
    np.savetxt('stat.txt',data)

if __name__ == '__main__':
    main()
