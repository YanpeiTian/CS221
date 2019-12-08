import random
import numpy as np
import logic0
import utilLookupTable

# 64-bit int representing a board. Format:
# [63-60][59-56][55-52][51-48]
# [47-44][43-40][39-36][35-32]
# [31-28][27-24][23-20][19-16]
# [15-12][11- 8][ 7- 4][ 3- 0]

# count bit position from right to left
# b63 b62 b61 b60 ... b3 b2 b1 b0
# so the first row will be represented first in the 64 bit integer
# the last row will be the last 16 bits of the 64 bit integer

# test cases
# from Expectmax_Opt_vectorize, got board:
# [[1 1 0 0]
#  [6 0 0 0]
#  [3 4 0 0]
#  [2 1 3 0]]
# from Expectmax_Opt_vectorize, got utility
#  882.3000000000001

# transcribed board into binary
# 0x0001000100000000 0110000000000000 0011010000000000 0010000100110000
# 4352               24576            13312            8496
# first row 4352
# powerlist of 1 1 0 0 dotted with 13.5, 12.1, 10.2, 9.9
# = 71.3
# second row 24576
# powerlist of 6 0 0 0 dotted with 9.9, 8.8, 7.6, 7.2
# = 657.2
# powerlist of 3 4 0 0 dotted with 6.0, 5.6, 3.7, 1.6
# = 142.9
# powerlist of 2 1 3 0 dotted with 1.2, 0.9, 0.5, 0.3
# = 10.9
# their sum is 882.3
# printBoard(1225084652633465136)
# print(toList(1225084652633465136))
# print(getUtility(1225084652633465136))


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


def move_row_left(board, row):
    b = (board >> 16*row) & 0xffff
    board = logic0.move_table[b] & 0xffff;
    return board << 16*row


def down(board):
    return transpose(right(transpose(board)))


def up(board):
    return transpose(left(transpose(board)))


def right(board):
    return reverse(left(reverse(board)))


def left(board):
    return move_row_left(board, 0) | move_row_left(board, 1) | move_row_left(board, 2) | move_row_left(board, 3)



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
    ls.reverse()
    return ls

def printBoard(board):
    print("Printing current board configuration")
    ls = toList(board)
    print(ls)
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
def getUtility2(board):
    ls = toList(board) # list of exponents
    # print(ls)
    powerList = [2**exponent for exponent in ls] # list of actual values
    # print(powerList)
    # print(heu)
    return np.dot(heu, powerList)

def getUtility(board):
    # official = getUtility2(board)
    # 16 bits at a time
    # row 1
    # printBoard(board)
    # rowOne = board & 0xff
    # util = utilLookupTable.utilRowOne[rowOne]
    # print("row 4 util " + str(util))
    #
    # board = board >> 16
    # rowTwo = board & 0xff
    # util += utilLookupTable.utilRowTwo[rowTwo]
    # print("row 3 util " + str(utilLookupTable.utilRowTwo[rowTwo]))
    #
    # board = board >> 16
    # rowThree = board & 0xff
    # util += utilLookupTable.utilRowThree[rowThree]
    # print("row 2 util " + str(utilLookupTable.utilRowThree[rowThree]))
    #
    #
    # board = board >> 16
    # rowFour = board & 0xff
    # util += utilLookupTable.utilRowFour[rowFour]
    # print("row 1 util " + str(utilLookupTable.utilRowFour[rowFour]))

    # printBoard(board)
    rowFour = board & 0xffff
    util = utilLookupTable.utilRowFour[rowFour]
    # print("printing board's binary = " + str(bin(board)))
    # print("row 4 16 bits to int = " + str(rowFour))
    # print(bin(rowFour))
    # print("row 4 util " + str(util))

    board = board >> 16
    rowThree = board & 0xffff
    util += utilLookupTable.utilRowThree[rowThree]
    # print("printing board's binary = " + str(bin(board)))
    # print("row 3 16 bits to int = " + str(rowThree))
    # print(bin(rowThree))
    # print("row 3 util " + str(utilLookupTable.utilRowThree[rowThree]))

    board = board >> 16
    rowTwo = board & 0xffff
    util += utilLookupTable.utilRowTwo[rowTwo]
    # print("printing board's binary = " + str(bin(board)))
    # print("row 2 16 bits to int = " + str(rowTwo))
    # print(bin(rowTwo))
    # print("row 2 util " + str(utilLookupTable.utilRowTwo[rowTwo]))


    board = board >> 16
    rowOne = board & 0xffff
    util += utilLookupTable.utilRowOne[rowOne]
    # print("printing board's binary = " + str(bin(board)))
    # print("row 1 16 bits to int = " + str(rowOne))
    # print(bin(rowOne))
    # print("row 1 util " + str(utilLookupTable.utilRowOne[rowOne]))


    # if official != util:
    #     print("should get " + str(official))
    #     print("but got " + str(util) + " instead")
    #     print()
    return util


def getMax(board):
    ls = toList(board)
    return max(ls)
