import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import ExpectiMax
import logic

ITERATION=20 #100
DEPTH=3
stat=[]

def main():
    f= open("expectimax2.txt","w+")
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

        f.write("Step %d, max %d, Score %d, \r\n" % (step, 2**np.max(matrix), logic.getScore(matrix)))
        f.flush()

    f.close()

if __name__ == '__main__':
    main()
