from matplotlib import pyplot as plt

from confrez.rl import pklot_env_unicycle_cont as pklot_env_cont
from confrez.rl.pklot_env_unicycle_cont import parallel_env
from confrez.rl.pklot_env_unicycle_cont import env as aec_env

from pettingzoo.test import (
    parallel_api_test,
    max_cycles_test,
    render_test,
    performance_benchmark,
    test_save_obs,
)
# from pettingzoo.test.seed_test import parallel_seed_test
#
# parallel_api_test(parallel_env(), num_cycles=300)
# print("passed parallel api test")
#
# parallel_seed_test(parallel_env, num_cycles=10, test_kept_state=True)
# print("passed parallel seed test")
#
# # max_cycles_test(pklot_env)
# # print("passed max_cycle test")
#
# render_test(aec_env)
# print("passed render test")
#
# # performance_benchmark(aec_env())
#
# # test_save_obs(aec_env())
#
env_config = pklot_env_cont.EnvParams(
    reward_stop=-5, reward_dist=-0.1, reward_heading=-0.1, reward_time=-0.1, reward_collision=-10, reward_goal=100,
    window_size=140
)
env = parallel_env(params=env_config, n_vehicles=4, random_reset=True, render_mode='rgb_array')
env.reset()
#
# observations = env.reset(seed=0)
img = env.render()

plt.imshow(img)
plt.show()
