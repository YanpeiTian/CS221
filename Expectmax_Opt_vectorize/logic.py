import random
import numpy as np

singleScore=[0,0,4,16,48,128,320,768,1792,4096,9216,20480,45056,98304,212992,458752,983040]
moveDict=np.load('move.npy')


def add_two(mat):
    indexs=np.argwhere(mat==0)
    index=np.random.randint(0,len(indexs))
    mat[tuple(indexs[index])] = 1
    return mat


def game_state(mat):
    return 'not over' if np.any(mat == 0) or np.any(mat[:, 0:-1]==mat[:, 1:]) or np.any(mat[0:-1, :]==mat[1:, :]) else 'lose'

def move(list):
    return moveDict[list[0],list[1],list[2],list[3],:]

def lookup(x):
    return singleScore[x]

lookup = np.vectorize(lookup)

def getScore(matrix):
    return np.sum(lookup(np.where(matrix > 2, matrix, 0)))
