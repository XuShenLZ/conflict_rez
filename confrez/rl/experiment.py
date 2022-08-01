from confrez.rl.pklot_env import parallel_env
from confrez.rl.utils import ProcessMonitor
from stable_baselines3 import DQN, PPO

env = parallel_env()
model_env = parallel_env()
monitor = ProcessMonitor(env)

# model = PPO.load("PPO-CNN_07-31-2022_10-12-36")
model = DQN.load("DQN-CNN_08-01-2022_12-10-59")

observations = env.reset(seed=0)
env.render()

max_cycles = 500

for step in range(max_cycles):
    actions = {
        agent: model.predict(observations[agent].copy(), deterministic=True)[0]
        for agent in env.agents
    }
    monitor.show(observations=observations, actions=actions, notes="Before step")
    observations, rewards, dones, infos = env.step(actions)
    env.render()

    if all(dones.values()):
        observations = env.reset()
        print("All agents are done. Reset")
