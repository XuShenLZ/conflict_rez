from confrez.rl.pklot_env import parallel_env
import supersuit as ss
from stable_baselines3 import PPO, DQN

# from stable_baselines3.ppo import CnnPolicy

from datetime import datetime
from os import path as os_path

cwd = os_path.dirname(__file__)


env = parallel_env()
env = ss.black_death_v3(env)
# env = ss.color_reduction_v0(env, mode="B")
# env = ss.frame_stack_v2(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 16, num_cpus=8, base_class="stable_baselines3")

now = datetime.now()
timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

# model = PPO(
#     CnnPolicy,
#     env,
#     verbose=3,
#     n_steps=64,
#     n_epochs=5,
#     tensorboard_log=f"{cwd}/PPO-CNN_tensorboard/",
# )

model = DQN(
    "CnnPolicy",
    env,
    verbose=3,
    buffer_size=50000,
    learning_starts=500,
    tensorboard_log=f"{cwd}/DQN-CNN_tensorboard/",
)

model.learn(total_timesteps=2000000, tb_log_name=f"DQN-CNN_{timestamp}")
model.save(f"DQN-CNN_{timestamp}")
print("Training Finished")
