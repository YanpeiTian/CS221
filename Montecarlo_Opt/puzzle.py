import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import montecarlo
import logic

ITERATION=1
NUM_BACKGROUND_RUNS=100
stat=[]

def main():
    for iteration in range(ITERATION):
        print("Iteration: "+str(iteration+1))
        start=time.time()
        step=0
        matrix=np.zeros((4,4),dtype=np.int)
        matrix, _ = logic.add_two(matrix)
        while True:
            matrix, success = logic.add_two(matrix)
            if not success:
                break
            # print(matrix)
            if logic.game_state(matrix)=='lose':
                break

            move=montecarlo.getMove(matrix, NUM_BACKGROUND_RUNS)
            # print("got a move " + str(move))
            matrix=montecarlo.moveGrid(matrix,move)
            #print(matrix)
            step+=1

        print("Step= "+str(step))
        print("Max= "+str(2**np.max(matrix)))
        print("Score= "+str(logic.getScore(matrix)))
        print('Depth= ' + str(NUM_BACKGROUND_RUNS))
        print('Time= '+str(time.time()-start))
        print('')
        stat.append((step,2**np.max(matrix),logic.getScore(matrix)))

if __name__ == '__main__':
    main()
