from matplotlib import pyplot as plt

from confrez.rl import pklot_env_unicycle_cont as pklot_env_cont
from confrez.rl.pklot_env_unicycle_cont import parallel_env
from confrez.rl.pklot_env_unicycle_cont import env as aec_env

env_config = pklot_env_cont.EnvParams(
    reward_stop=-5, reward_dist=-0.1, reward_heading=-0.1, reward_time=-0.1, reward_collision=-10, reward_goal=100,
    window_size=140
)
env = parallel_env(params=env_config, n_vehicles=1, random_reset=True, render_mode='rgb_array')
env.reset()
#
# observations = env.reset(seed=0)
img = env.render()

plt.imshow(img)
plt.show()
