# conflict_rez
Conflict resolution for multiple vehicles in confined spaces

Xu Shen (xu_shen@berkeley.edu), Francesco Borrelli

Project Webpage: https://bit.ly/rl-cr

![rl-control-side-by-side - 2022-11-01 10-44-17 00_00_00-00_00_30](https://user-images.githubusercontent.com/31999415/199302031-47ae6032-ff42-4f83-978f-6d083e58e17c.gif)


## Install
1. Clone this repo
2. Run `pip install -e .` in the root level of this repo (A virtualenv is recommended). However, due to some buggy dependencies between package versions, you need to manually switch the following packages to specific versions (while ignoring the warning given by the pip dependency resolver):
    1. `pip install pettingzoo==1.20.1`
    2. `pip install supersuit==3.5.0`
    3. `pip install stable-baselines3==1.6.0`
    4. `pip install gym==0.25.0`

## Testing
1. A pretrained model can be downloaded [here](https://drive.google.com/file/d/10atWJc3hnfziuEjkJWI-BA6LhUGnzoJr/view?usp=sharing).
2. Run `python confrez/rl/experiment.py` to see the steps taken by the DQN policy to resolve the conflict in the discrete environment.
3. Run `python confrez/rl/record_states_history.py` to generate a `.pkl` file that records the steps taken by the RL agents, which will serve as the configuration strategies for the trajectory planning problems.
4. Run `python confrez/control/vehicle.py` to plan a single-vehicle collision free trajectory following the strategy-guided configurations.
5. (Centralized method) Run `python confrez/control/multi_vehicle_planner.py` to solve the multi-vehicle trajectory planning problem to resolve conflict. 
6. (Distributed method) Run `python confrez/control/vehicle_follower.py` so that each vehicle generates its own strategy-guided reference trajectory, and then follows it with distributed MPC to avoid collisions.

## Training
1. Run `python confrez/rl/train.py` to train a new policy.
2. You may set the `random_reset` argument to `True` so that you train the policy with a random subset out of 4 vehicles, which may lead to a more generalizable policy. But it also require longer training time.
3. Different random seeds and different parameter tuning will lead to different behaviors in the trained policy. With your new policy, the vehicles may (almost for sure) take different actions and resolve the conflict in another way.
