import random
import numpy as np

singleScore=[0,0,4,16,48,128,320,768,1792,4096,9216,20480,45056,98304,212992,458752,983040]
moveDict=np.load('move.npy')


def add_two(mat):
    indexs=np.argwhere(mat==0)
    index=np.random.randint(0,len(indexs))
    # if np.random.uniform()<0.9:
    #     mat[tuple(indexs[index])] = 1
    # else:
    #     mat[tuple(indexs[index])] = 2
    mat[tuple(indexs[index])] = 1
    return mat


def game_state(mat):
    for i in range(len(mat)-1):
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return 'not over'
    for i in range(len(mat)):  # check for any zero entries
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return 'not over'
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return 'not over'
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return 'not over'
    return 'lose'

def move(list):
    return moveDict[list[0],list[1],list[2],list[3],:]

def getScore(matrix):
    score=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]>2:
                score=score+singleScore[matrix[i][j]]
    return score
