import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import montecarlo
import logic

ITERATION=10
NUM_BACKGROUND_RUNS=5
stat=[]

def main():
    for iteration in range(ITERATION):
        print("Iteration: "+str(iteration+1))
        start=time.time()
        step=0
        matrix=np.zeros((4,4),dtype=np.int)
        matrix = logic.add_two(matrix)
        while True:
            matrix = logic.add_two(matrix)
            if logic.game_state(matrix) == 'lose':
                break
            if logic.game_state(matrix)=='lose':
                break
            matrix = montecarlo.getMove(matrix, NUM_BACKGROUND_RUNS)
            step += 1
            # move=montecarlo.getMove(matrix, NUM_BACKGROUND_RUNS)
            # # print("got a move " + str(move))
            # matrix=montecarlo.moveGrid(matrix,move)
            # step+=1
        print(matrix)
        print("Step= "+str(step))
        print("Max= "+str(2**np.max(matrix)))
        print("Score= "+str(logic.getScore(matrix)))
        print('Depth= ' + str(NUM_BACKGROUND_RUNS))
        print('Time= '+str(time.time()-start))
        print('')
        stat.append((step,2**np.max(matrix),logic.getScore(matrix)))

if __name__ == '__main__':
    main()
