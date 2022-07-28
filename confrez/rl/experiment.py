from confrez.rl.pklot_env import parallel_env
from confrez.rl.utils import ProcessMonitor
from stable_baselines3 import PPO

env = parallel_env()
monitor = ProcessMonitor(env)

model = PPO.load("policy")

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
    monitor.show(observations=observations, actions=actions, notes="After step")

    if all(dones.values()):
        observations = env.reset()
        print("All agents are done. Reset")
