import collections
import numpy as np
# takes a list of size 4, turn it into 16 bit int
def listToInt(ls):
    return ls[0] << 12 | ls[1] << 8 | ls[2] << 4 | ls[3]
    # return ls[0] | ls[1] << 4 | ls[2] << 8 | ls[3] << 12

maxExponent = 16
def increment(ls):
    # increment from right to left
    ls[3] += 1
    if ls[3] == maxExponent:
        ls[3] = 0
        ls[2] += 1
        if ls[2] == maxExponent:
            ls[2] = 0
            ls[1] += 1
            if ls[1] == maxExponent:
                ls[1] = 0
                ls[0] += 1
    #             if ls[0] ==
    #
    # if ls[index] == 16 and index == 0:
    #     return None
    # if ls[index] == 16:
    #     ls[index] = 0
    #     index -= 1
    #     return index
    # ls[index] += 1
    # return index

def printDict(dict, i):
    if i == 0:
        print("utilRowOne = {", end = '')
    elif i == 1:
        print("utilRowTwo = {", end = '')
    elif i == 2:
        print("utilRowThree = {", end = '')
    else:
        print("utilRowFour = {", end = '')
    counter = 0
    for key, value in dict.items():
        print(str(key) + ": " + str(value) + ", ", end = '')
        counter += 1
        if counter == 10:
            counter = 0
            print()
    print("}")

heu = [[13.5, 12.1, 10.2, 9.9],
     [9.9, 8.8, 7.6, 7.2],
     [6.0, 5.6, 3.7, 1.6],
     [1.2, 0.9, 0.5, 0.3]]

def main():
    index = 3 # goes to 0
    ls = [0, 0, 0, 0]  # represents a row in exponents
    # dictList[0] = a dict for first row
    # dictList[1] = a dict for second row etc
    dictList = []
    for i in range(4):
        dictList.append(collections.defaultdict(int))
    # each dict is board to utility

    # try 16 for now, change to 64 later
    while ls[0] != maxExponent:

        val = listToInt(ls)
        # if ls == [2, 1, 3, 0]:
        #     print("2130 became val " + str(val))
        # print(val)
        powerList = [2**val for val in ls]

        for j in range(4):
            # if ls == [2, 1, 3, 0]:
            #     print("2130 powerlist dotted with the jth row of heu is")
            #     print(np.dot(heu[j], powerList))
            # say j is 0, ls is  0 0 0 1, powerlist is 0 0 0 2
            # dictList[j] is the jth row, heu[j] is also the jth row
            dictList[j][val] = np.dot(heu[j], powerList)
        # index = increment(ls, index)
        increment(ls)

    for i in range(4):
        printDict(dictList[i], i)


if __name__ == '__main__':
    main()

