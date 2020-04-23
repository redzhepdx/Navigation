[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Setup
The project uses Jupyter Notebook. This command needs to be run to install the needed packages:
```
pip install -r requirements.txt
```

### Training an Agent
- The last kernel trains in `Navigation.ipynb` notebook trains agent until it reaches the target score. It is mandatory to run all previous kernels to load environment and instantiate a Double DQN agent.
- Optional : You can change the learning algorithm's hyper-parameters by updating the config dictionary in the notebook

#### Initial Config
```
config = {
    "BUFFER_SIZE" : int(1e5), # replay buffer size
    "LR" : 5e-4,              # learning rate
    "BATCH_SIZE" : 64,        # minibatch size
    "UPDATE_EVERY" : 4,       # how often to update the network
    "GAMMA" : 0.99,           # discount factor
    "TAU" : 1e-3              # for soft update of target parameters
}
```

### Project Structure and Instructions
- `agent.py` -> Contains the implementation of Double DQN Agent
- `network.py` -> Contains the implementations of Critic and Actor Neural Networks. [Pytorch]
- `buffer.py` -> Contains the implementation of memory module(Standard Experience Replay)
- `Navigation.ipynb` -> Execution of the algorithm. Training Agents and Unity Visualizations
- `report.pdf` -> description of the methods and application
- `*.pth files` -> pre-trained local q neural network of the agent [checkpoint.pth]