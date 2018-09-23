# HW02 - Push the car up!

You are required to implement a Q-Learning agent that learns to get the 
car on top of the mountain.

### Dependencies
```
pip install numpy pandas gym 
```
### Important files
You will need only to modify ``QLearning.py`` to implement the ``TODO`` tasks mentioned in the comments. 

Please note that original problem has continuous observation space but we discretize this space for easier understanding. You can find function `` discretize(state) `` in  `` helpers.py`` that will help you discretize any state.

### Assessment 
To run your test
```
python3 main.py
```
After correct implementation your agent should at least pass ``9/10`` trials successfully. Good luck!