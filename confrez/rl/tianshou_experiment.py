import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou_train import *

env = DummyVectorEnv([lambda: get_env()])
policy, optim, agents = get_agents()

for agent in agents:
    #shouldn't be in this way when num_agents > 1!
    policy.policies[agent].load_state_dict(torch.load('./log/rps/dqn/policy.pth', map_location=torch.device('cpu')))


for agent in agents:
    policy.policies[agent].set_eps(0)
collector = Collector(policy, env, exploration_noise=True)
result = collector.collect(n_episode=1, render=1/1000)
rews, lens = result["rews"], result["lens"]
print(f"Final reward: {rews[:].mean()}, length: {lens.mean()}")
