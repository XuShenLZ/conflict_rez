from typing import Callable, List
from confrez.rl.pklot_env import parallel_env
from confrez.rl.utils import TensorboardCallback
import supersuit as ss
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from scipy import interpolate

from datetime import datetime
from os import path as os_path

cwd = os_path.dirname(__file__)

set_random_seed(0)

MODEL_NAME = "DQN-CNN-4v-fixed"

env = parallel_env(n_vehicles=4, seed=1, random_reset=False)
env = ss.black_death_v3(env)
env = ss.resize_v1(env, 140, 140)
# env = ss.color_reduction_v0(env, mode="B")
# env = ss.frame_stack_v2(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 24, num_cpus=12, base_class="stable_baselines3")

eval_env = parallel_env(n_vehicles=4, seed=1, random_reset=False)
eval_env = ss.black_death_v3(eval_env)
eval_env = ss.resize_v1(eval_env, 140, 140)
# env = ss.color_reduction_v0(env, mode="B")
# env = ss.frame_stack_v2(env, 3)
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
eval_env = ss.concat_vec_envs_v1(
    eval_env, 24, num_cpus=12, base_class="stable_baselines3"
)

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")


def step_schedule(
    initial_value: float, steps: List[float], levels: List[float]
) -> Callable[[float], float]:
    """
    Step learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    weight_func = interpolate.interp1d(steps, levels, "next", fill_value="extrapolate")

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        weight = weight_func(progress_remaining).__float__()
        return weight * initial_value

    return func


model = DQN(
    "CnnPolicy",
    env,
    learning_rate=step_schedule(0.0005, [1, 0.8, 0.6, 0.3], [1, 0.5, 0.1, 0.05]),
    verbose=3,
    buffer_size=100000,
    learning_starts=500,
    exploration_fraction=0.7,
    exploration_final_eps=0.2,
    tensorboard_log=f"{cwd}/DQN-CNN_tensorboard/",
)

model.learn(
    total_timesteps=1e8,
    tb_log_name=f"{MODEL_NAME}_{timestamp}",
    callback=TensorboardCallback(eval_env=eval_env, n_eval_episodes=10, eval_freq=400),
)
model.save(f"{MODEL_NAME}_{timestamp}")
print("Training Finished")
