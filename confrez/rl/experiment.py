from confrez.rl.pklot_env import parallel_env
from confrez.rl.utils import ProcessMonitor
from stable_baselines3 import DQN, PPO
import supersuit as ss

env = parallel_env(n_vehicles=4)
env = ss.resize_v1(env, 140, 140)
# monitor = ProcessMonitor(env)

# model = PPO.load("PPO-CNN_07-31-2022_10-12-36")
model = DQN.load("DQN-CNN-4v-new-color_08-06-2022_23-26-16")

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

    if all(dones.values()):
        observations = env.reset()
        env.render()
        print("All agents are done. Reset")
