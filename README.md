# Solving-Random-Environment-with-Robot-Learning
## The Environment
The environment is generated entirely randomly, consisting of a start state (blue square) and a goal state (green star); the objective of the robot (red circle) is to reach the goal state as quickly as possible. However the environment contains a central obstacle which is also of a random size and random position, and uneven terrain with areas of high impedance (light areas) through which the robot will move extremely slowly, and areas of low impedance (dark areas) through which the robot will move quickly. At the beginning the robot does not know where the obstacle is or where the areas of high impedance are, however it knows the location of the goal state. An example of such an environment is given below. 

<img width="200" alt="image" src="https://user-images.githubusercontent.com/72558653/228891958-10b38a9f-b192-4634-93fd-01e499a48e6b.png">

## Problem Description
The program flow consists of two phases, training and testing, with a time limit of 600 seconds and 200 seconds, respectively. The robot is required to explore the environment during the training phase to reduce model uncertainty and learn the environment dynamics. During the testing phase, the robot must reach the goal state within the time limit by using the learned model.

## Approach
The approach for this project is as follows:
* Use a simple strategy for training, starting with a period of random exploration followed by a period of planning towards the goal, using the model.
* Implement a planning algorithm using the model for testing.
* Determine exploration and planning strategies, as well as parameters such as the episode length, planning horizon, and number of samples during planning.
* Explore trade-offs between physical movement and planning, as well as between exploration and exploitation.
* Design a reward function and test its effectiveness.
* Determine when and if the robot should be reset to its initial state.
* Experiment with dynamic parameters, such as episode length and planning horizon.
* Bias exploration if necessary, using knowledge of the goal state or providing demonstrations.
* Use the model's uncertainty to determine how much to trust a particular planned path.

## High-Level Description
The overall algorithm uses cross-entropy planning to calculate a sequence of actions aiming to reach the next milestone on route to the goal state. The strategy for model predictive control is closed-loop planning, which replans a new sequence of actions every 2 steps. The algorithm uses a short planning horizon of 2 steps, which is computationally efficient and leads to the robot making direct movements even at the initial training stage when the underlying model dynamics are unknown. The algorithm uses a dense reward function (the distance between robot's current state and the next milestone), with slight modifications to the reward depending on the stage of training. In order to avoid the robot getting stuck at a local optimum where the robot traverses through the obstacle (significantly slowing the
robot down), the training and testing strategy incorporates ‘milestones’ which helps guide the robot outside of the obstacle path. The goal of training is thus to refine the milestones which are later used in the testing phase. At testing time, the robot exploits the best path explored, by planning towards each milestone then finally to the goal state, which results in always getting around the central obstacle.

## Trade-Offs
During the training phase, the algorithm balances physical movement with planning, and exploration with exploitation. It explores trade-offs between:
* Amount of physical movement vs. amount of planning
* Amount of exploration vs. amount of exploitation
* Effectiveness of a dense vs. sparse reward function

## Running the code
1) Clone this repository to your local machine using git clone.
2)Navigate to the project directory.
3)Install the necessary dependencies by running pip install -r requirements.txt.
4) Run the program using python robot-learning.py.


