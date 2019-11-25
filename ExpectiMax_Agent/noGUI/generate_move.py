import numpy as np
import logic

def main():
    move_left=np.zeros((16,16,16,16,4))
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    row=[a,b,c,d]
                    mat=[row for i in range(4)]
                    newMat,_=logic.left(mat)
                    move_left[a,b,c,d,:]=newMat[0]
    np.save('move',move_left)

if __name__ == '__main__':
    main()
