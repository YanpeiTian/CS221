import random
import numpy as np
import logic0

# 64-bit int representing a board. Format:
# [63-60][59-56][55-52][51-48]
# [47-44][43-40][39-36][35-32]
# [31-28][27-24][23-20][19-16]
# [15-12][11- 8][ 7- 4][ 3- 0]

def new_game():
    return 0

def frees(board):
    if board == 0:
        return 16
    free = 0
    # print("free initialized to 0")
    for i in range(16):
        if board & 0xf == 0:
            free += 1
        # free += ~(board | board >> 1 | board >> 2 | board >> 3) & 1
        # print("frees updated to " + str(free))
        board = board >> 4
    return free


def add_two(board):
    # print(type(board))
    # print(board)
    p = random.choice(range(frees(board)))
    i = 0
    while p != -1:
        if board >> i*4 & 0xf == 0:
            p -= 1
        i += 1
    return board | 1 << 4*(i-1)

def transpose(board):
    # res = np.uint64(0)
    # print(type(board))
    # print("in transpose, board is " + str(board))
    res = 0
    res |= board << 4*9 & 0x000f000000000000
    res |= board << 4*6 & 0x00f0000f00000000
    res |= board << 4*3 & 0x0f0000f0000f0000
    res |= board        & 0xf0000f0000f0000f
    res |= board >> 4*3 & 0x0000f0000f0000f0
    res |= board >> 4*6 & 0x00000000f0000f00
    res |= board >> 4*9 & 0x000000000000f000
    # print("transpose is returning " + str(res))
    return res

def reverse(board):
    # print("in reverse, board is " + str(board))
    board = (board << 8 & 0xff00ff00ff00ff00) | (board >> 8 & 0x00ff00ff00ff00ff)
    board = (board << 4 & 0xf0f0f0f0f0f0f0f0) | (board >> 4 & 0x0f0f0f0f0f0f0f0f)
    # print("reverse returns board " + str(board))
    # print("bit length " + str(int.bit_length(board)))
    return board

def move_row_right(board, row):
    # b = np.uint64(0)
    # print("in move row right, board is " + str(board))
    b = (board >> 16*row) & 0xffff
    board = logic0.move_table[b] & 0xffff;
    return board << 16*row


def up(board):
    # print("in up, board is " + str(board))
    return transpose(left(transpose(board)))


def down(board):
    # print("in down, board is " + str(board))
    return transpose(right(transpose(board)))


def left(board):
    # print("in left, board is " + str(board))
    num = reverse(right(reverse(board)))
    # print("left is returning " + str(num))
    return num


def right(board):
    # print("in right, board is " + str(board))
    return move_row_right(board, 0) | move_row_right(board, 1) | move_row_right(board, 2) | move_row_right(board, 3)


def move(board, dir):
    if dir == 0:
        return up(board)
    elif dir == 1:
        return left(board)
    elif dir == 2:
        return down(board)
    else:
        return right(board)

def canMoveDirection(board, dir):
    return move(board, dir) != board

def gameOver(board):
    for dir in range(4):
        if canMoveDirection(board, dir):
            return False
    return True

# return a list of exponents
def toList(board):
    ls = []
    for i in range(16):
        val = board >> 4 * i & 0xf
        ls.append(val)
    return ls

def printBoard(board):
    print("Printing current board configuration")
    ls = toList(board)
    for i in range(16):
        print(ls[i], end='  ')
        if (i+1) % 4 == 0:
            print()
    print()

singleScore=[0,0,4,16,48,128,320,768,1792,4096,9216,20480,45056,98304,212992,458752,983040]
def getScore(board):
    ls = toList(board)
    score = 0
    for val in ls:
        score += singleScore[val]
    return score

heu=[13.5, 12.1, 10.2, 9.9,9.9, 8.8, 7.6, 7.2,6.0, 5.6, 3.7, 1.6,1.2, 0.9, 0.5, 0.3]
def getUtility(board):
    ls = toList(board) # list of exponents
    powerList = [2**exponent for exponent in ls] # list of actual values
    multiplier = np.dot(heu, powerList)
    return multiplier

    # score = 0
    # for val in ls:
    #     score += singleScore[val]
    # return score * multiplier

def getMax(board):
    ls = toList(board)
    return max(ls)
