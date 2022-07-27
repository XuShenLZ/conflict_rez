from confrez.rl.pklot_env import parallel_env

env = parallel_env()

observations = env.reset(seed=0)
env.render()

max_cycles = 500

for step in range(max_cycles):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    print(env.occupancy)
    observations, rewards, dones, infos = env.step(actions)
    env.render()
    print(env.occupancy)
    print(f"step: {step}")
