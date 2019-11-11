import random
import numpy as np
import logic

FAIL_SCORE=-1e10
moves=["'w'","'a'","'s'","'d'"]

def getMove(grid,depth):
    move=0
    high=FAIL_SCORE

    for i in range(4):
        newGrid=np.array(moveGrid(grid,i))
        if(isSame(grid,newGrid)==False):

            score=expectiMax()
            
            if score>high:
                high=score
                move=i

    return moves[move]

def moveGrid(grid,i):
    if i==0:
        new,_=logic.up(grid)
        return new
    if i==1:
        new,_=logic.left(grid)
        return new
    if i==2:
        new,_=logic.down(grid)
        return new
    if i==3:
        new,_=logic.right(grid)
        return new

def isSame(grid1,grid2):
    for i in range(4):
        for j in range(4):
            if grid1[i,j]!=grid[i,j]:
                return False
    return True


# double expectimax(Board b, int steps, int agent, TranspositionTable& t) {
#     Board dup(b);
#     double score = 0;
#     if (!canMove(b)) return -DBL_MAX;
#     if (steps == 0) return b.calculateScore(false);
#     else if (agent == 1){
#         int c = 0;
#         #pragma omp parallel for
#         for (int i=0; i<16; i++) {
#             if (b.getTile(i) == 0 && c < steps+2) {
#                 dup.state = b.state;
#                 dup.setTile(i,1);
#                 score += 0.9 * expectimax(dup,steps-1,0,t);
#                 dup.state = b.state;
#                 dup.setTile(i,2);
#                 score += 0.1 * expectimax(dup,steps-1,0,t);
#                 c++;
#             }
#         }
#         if (c == 0) return -DBL_MAX;
#         return score/c;
#     }
#     else if (agent == 0) {
#         for (int i : valid_moves(b)) {
#             dup.state = b.state;
#             dup.move(i,false,false);
#             score = max(score, expectimax(dup,steps-1,1,t));
#         }
#         return score;
#     }
#     return 0;
# }
