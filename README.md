# CS 221 - 2048 AI Challenge

This is a class project for CS 221 at Stanford University.

## How To Run

We are using Python 3 for this project.

Inside every directory, you will find a `puzzle.py`, which is 
the main function of the game. You can start the game by running
```
python puzzle.py
```


You can configure the depth for expectimax and number of background runs for Monte-Carlo 
in `puzzle.py` as well. For example, to configure expectimax depth, simply go to `puzzle.py` and change the 
```
DEPTH=3
```
to any depth you would like. The higher the depth, the longer it takes to 
perform one move.


## Visualization

In the `Starter_Code` directory, the game is playable with keyboard strokes. 

In all other directories, the game will start playing to simulate AI v.s. the computer agent (which generates a 2-tile at a random location).

Some directories (such as `Expectimax_Opt`, `Expectimax_Opt_vectorized`, and `BitBoards`) 
don't have a GUI set up. This is because we want to decrease time needed to execute the program.

For any regular boards, you can print the board using the `print()` function in `puzzle.py`. For bit boards, we implemented
a `logic.printBoard()` function for you to use in main.


