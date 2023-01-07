from confrez.rl.pklot_env import parallel_env
from stable_baselines3 import DQN
import supersuit as ss

import pickle

file_name = "4v_rl_traj"
env = parallel_env(n_vehicles=4, random_reset=False)

_, init_infos = env.reset(seed=0, return_info=True)
states_history = {agent: [init_infos[agent]["states"]] for agent in env.agents}

env = ss.resize_v1(env, 140, 140)
model = DQN.load("DQN-CNN-4v-new-color_08-06-2022_23-26-16")

observations = env.reset(seed=0)

dones = {agent: False for agent in env.agents}

while not all(dones.values()):
    actions = {
        agent: model.predict(observations[agent].copy(), deterministic=True)[0]
        for agent in env.agents
    }
    observations, rewards, dones, infos = env.step(actions)
    for agent in infos:
        states_history[agent].append(infos[agent]["states"])


with open(file_name + ".pkl", "wb") as f:
    pickle.dump(states_history, f)

print(f"States history saved in {file_name}.pkl")
