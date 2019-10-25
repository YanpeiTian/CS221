import random
from tkinter import Frame, Label, CENTER
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

import logic
import constants as c

REFRESH_RATE=1
SLEEP_TIME=1
N_ITERATION=1000

stat=[]

class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)

        self.grid()
        self.master.title('2048')
        # self.master.bind("<Key>", self.key_down)

        # self.gamelogic = gamelogic
        # self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
        #                  c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
        #                  c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
        #                  c.KEY_LEFT_ALT: logic.left, c.KEY_RIGHT_ALT: logic.right,
        #                  c.KEY_H: logic.left, c.KEY_L: logic.right,
        #                  c.KEY_K: logic.up, c.KEY_J: logic.down}
        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                          c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right}

        self.grid_cells = []
        self.grid_cells = []
        self.score=0
        self.step=0
        self.done=0

        self.init_grid()
        self.init_matrix()
        self.update_grid_cells()
        self.agent()
        self.mainloop()

    # Agent of the game
    def agent(self):
        # simple greedy Agent
        keys=["'w'","'a'","'s'","'d'"]
        scores=[0]*4
        matrix=np.zeros((4,4,4))
        for i in range(len(keys)):
            matrix[i], _ = self.commands[keys[i]](self.matrix)
            scores[i]=self.getScore(matrix[i])

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        self.key_down(keys[chosenIndex])
        self.after(REFRESH_RATE, self.agent)

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME,
                           width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                             width=c.SIZE / c.GRID_LEN,
                             height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING,
                          pady=c.GRID_PADDING)
                t = Label(master=cell, text="",
                          bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                          justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def gen(self):
        return random.randint(0, c.GRID_LEN - 1)

    def init_matrix(self):
        self.matrix = logic.new_game(4)
        self.history_matrixs = list()
        self.matrix = logic.add_two(self.matrix)
        self.matrix = logic.add_two(self.matrix)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(
                        text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(
                        new_number), bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number])
        self.update_idletasks()

        # if self.done==1:
        #     time.sleep(5)
        #     stat.append(self.score)
        #     self.quit()

    def key_down(self, key):
        # key = repr(event.char)
        if self.done==1:
            # time.sleep(SLEEP_TIME)
            self.score=self.getScore(self.matrix)
            max=np.max(np.max(self.matrix))
            stat.append((self.score,max,self.step))
            self.quit()

        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                done = False
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(
                        text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(
                        text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.done=1
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(
                        text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(
                        text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.done=1
            self.step+=1

    def generate_next(self):
        index = (self.gen(), self.gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (self.gen(), self.gen())
        self.matrix[index[0]][index[1]] = 2

    def getScore(self,matrix):
        score=0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]>2:
                    score=score+self.singleScore(matrix[i][j])
        return score

    def singleScore(self,n):
        if n==4:
            return 4
        else:
            return n+2*self.singleScore(n/2)


for i in range(N_ITERATION):
    print("This is "+str(i+1)+" iteration.")
    gamegrid = GameGrid()
    gamegrid.destroy()

print(stat)
stat=np.array(stat)
np.savetxt('greedy_agent.dat', stat)

x=[i+1 for i in range(len(stat))]
fig, ax = plt.subplots()
ax.plot(x, stat[:,0])

ax.set(xlabel='Individual Trial', ylabel='Total Score',
       title='Greedy Agent Performance')

fig.savefig("greedy_agent_score.png")

scores=np.array(stat[:,0],dtype=np.int)
scores.sort()
x=[i for i in range(scores[0],scores[-1])]
cdf=[0]*len(x)
for i in range(len(cdf)):
    count=0
    for j in range(len(scores)):
        if scores[j]<=x[i]:
            count+=1
    cdf[i]=count/len(scores)

fig, ax = plt.subplots()
ax.plot(x, cdf)

ax.set(xlabel='Score', ylabel='Probability',
       title='Cumulative Distribution Function of Scores')

fig.savefig("greedy_agent_score_cdf.png")


print("Average score is: "+str(np.average(stat[:,0])))
print("Standard deviation is: "+str(np.std(stat[:,0])))

dict={}
for i in range(len(stat)):
    if stat[i,1] in dict.keys():
        dict[stat[i,1]]+=1
    else:
        dict[stat[i,1]]=1
print(dict)

fig, ax = plt.subplots()
x=[i+1 for i in range(len(stat))]
ax.plot(x, stat[:,2])

ax.set(xlabel='Individual Trial', ylabel='Total Step',
       title='Greedy Agent Performance, Step')

fig.savefig("greedy_agent_step.png")

steps=np.array(stat[:,2],dtype=np.int)
steps.sort()
x=[i for i in range(steps[0],steps[-1])]
cdf=[0]*len(x)
for i in range(len(cdf)):
    count=0
    for j in range(len(steps)):
        if steps[j]<=x[i]:
            count+=1
    cdf[i]=count/len(steps)

fig, ax = plt.subplots()
ax.plot(x, cdf)

ax.set(xlabel='Number of Steps', ylabel='Probability',
       title='Cumulative Distribution Function of Total Steps')

fig.savefig("greedy_agent_step_cdf.png")
