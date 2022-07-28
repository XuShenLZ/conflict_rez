from confrez.rl.pklot_env import parallel_env
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

env = parallel_env()
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 16, num_cpus=8, base_class="stable_baselines3")

model = PPO(
    CnnPolicy,
    env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.00062211,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=5,
    clip_range=0.3,
    batch_size=256,
)

model.learn(total_timesteps=2000000)
model.save("policy")
