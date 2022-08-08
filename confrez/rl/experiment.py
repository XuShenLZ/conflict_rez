from time import sleep
from confrez.rl.pklot_env import parallel_env
from confrez.rl.utils import ProcessMonitor
from stable_baselines3 import DQN, PPO
import supersuit as ss

env = parallel_env(n_vehicles=1)
env = ss.resize_v1(env, 140, 140)
# monitor = ProcessMonitor(env)

# model = PPO.load("PPO-CNN_07-31-2022_10-12-36")
model = DQN.load("DQN-CNN-1v-new-color_08-07-2022_15-04-22")

observations = env.reset(seed=0)
env.render()

max_cycles = 500

for step in range(max_cycles):
    actions = {
        agent: model.predict(observations[agent].copy(), deterministic=True)[0]
        for agent in env.agents
    }
    # monitor.show(observations=observations, actions=actions, notes="Before step")
    observations, rewards, dones, infos = env.step(actions)
    env.render()

    sleep(1)

    if all(dones.values()):
        observations = env.reset()
        env.render()
        print("All agents are done. Reset")
