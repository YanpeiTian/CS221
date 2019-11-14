import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import minimax

import logic
import constants as c

ITERATION=5
DEPTH=2
stat=[]

def main():
    for iteration in range(ITERATION):
        print("Iternaton: "+str(iteration+1))
        step=0
        matrix=np.zeros((4,4))
        matrix=logic.add_two(matrix)
        while True:
            matrix=logic.add_two(matrix)
            if logic.game_state(matrix)=='lose':
                break
            while True:
                move=minimax.getMove(matrix,DEPTH)
                if move=="'w'":
                    newMatrix,_=logic.up(matrix)
                if move=="'a'":
                    newMatrix,_=logic.left(matrix)
                if move=="'s'":
                    newMatrix,_=logic.down(matrix)
                if move=="'d'":
                    newMatrix,_=logic.right(matrix)
                if minimax.isSame(newMatrix,matrix)==False:
                    matrix=newMatrix
                    break
            step+=1
        print("Step= "+str(step))
        print("Max= "+str(np.max(matrix)))
        print("Score= "+str(logic.getScore(matrix)))
        print('')
        stat.append((step,np.max(matrix),logic.getScore(matrix)))
    np.savetxt("stat.txt",stat)

if __name__ == '__main__':
    main()
