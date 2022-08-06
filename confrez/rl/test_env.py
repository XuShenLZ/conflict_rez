from confrez.rl import pklot_env
from confrez.rl.pklot_env import parallel_env
from confrez.rl.pklot_env import env as aec_env

from confrez.rl.utils import ProcessMonitor

from pettingzoo.test import (
    parallel_api_test,
    max_cycles_test,
    render_test,
    performance_benchmark,
    test_save_obs,
)
from pettingzoo.test.seed_test import parallel_seed_test

parallel_api_test(parallel_env(), num_cycles=300)
print("passed parallel api test")

parallel_seed_test(parallel_env, num_cycles=10, test_kept_state=True)
print("passed parallel seed test")

# max_cycles_test(pklot_env)
# print("passed max_cycle test")

render_test(aec_env)
print("passed render test")

# performance_benchmark(aec_env())

# test_save_obs(aec_env())

env = parallel_env(n_vehicles=3)

monitor = ProcessMonitor(env)

observations = env.reset(seed=0)
env.render()

max_cycles = 500

for step in range(max_cycles):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    monitor.show(observations=observations, actions=actions, notes="Before step")
    observations, rewards, dones, infos = env.step(actions)
    env.render()
    print(f"step: {step}")
