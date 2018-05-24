# Learning Gridworld with Q-learning
## Game
Implementation of Q-learning with neural network for a gridworld game. Gridworld is a simple text based game in which there is a 5x5 grid 
(size of the grid can be easily changed) of tiles and 3 objects placed therein: a player (P), obstacle (O) and a goal (+). 
The player can move up/down/left/right and the point of the game is to get to the goal where the player will receive a numerical 
reward (+1). The player has to avoid obstacle because if he land on the it he is penalized with a negative reward (-1). 
Additionally, each move results in small negative reward (-0.05) in order to motivate the player to reach the goal as fast as possible. 
Hitting grid wall results in negative reward as well (-0.25). Example play grid is presented below:

![alt text](https://github.com/KarolloS/q_learning/blob/master/grid.jpg)

## Implementation
This repository contains implementation of Q-learning based on [TensorFlow](https://www.tensorflow.org/). In order to deal 
with the exploration problem, agent take a random movement with the probability 0.9*(1 - epoch/number_of_epochs) + 0.1. 
Implementation includes experience replay on order to deal with catastrophic forgetting problem. 

* `game.py` - raw implementation of gridworld game
* `game_learning_1` - agent learns the simplest case of the game - agent starting position, obstacle and goal position are always 
at the same 
* `game_learning_1` - agent learns the most difficult case of the game - agent starting position, obstacle and goal are always 
placed randomly 

More about Q-learning can be found [here](https://arxiv.org/pdf/1312.5602v1.pdf)

## Results
The best achieved accuracy (number of goals reached to the number of all testing games) for the **most difficult problem** 
(agent, obstacle and goal placed randomly) equals **0.96** with the average game score of **0.834**. This suggests that agent reached 
the goal in only few moves - it really learned how to play the game. Example game is presented below:

![alt text](https://github.com/KarolloS/q_learning/blob/master/game.jpg)

