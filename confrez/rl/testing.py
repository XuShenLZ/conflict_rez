from time import sleep
import supersuit as ss
import pklot_env_unicycle_cont as pklot_env_cont
from pklot_env_unicycle_cont import parallel_env, raw_env
from PIL import Image
import numpy as np
import pygame
from pygame.locals import *

env_config = pklot_env_cont.EnvParams(
    reward_stop=-10, reward_dist=-1, reward_heading=0, reward_time=-1, reward_collision=-10, reward_goal=10000,
    window_size=500
)
env = raw_env(n_vehicles=1, render_mode='human', random_reset=True, params=env_config, max_cycles=2000)

observations = env.reset()

frame_list = []
done, trunc = False, False
step = 0

# for _ in range(10):
#     env.reset()
total_rew = 0
while False in env.terminations.values():
    agent = env.agent_selection
    step += 1
    pygame.event.get()
    # actions = env.action_space(agent).sample()
    env.render()
    if done or trunc:
        env.step(None)
    while True:
        event = pygame.event.get()
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            actions = [0.5, 0]
            break
        if keys[K_s]:
            actions = [-0.5, 0]
            break
        if keys[K_d]:
            actions = [0, -0.2]
            break
        if keys[K_a]:
            actions = [0, 0.2]
            break
    #
    # elif agent == 'vehicle_1':
    #     durations = [30, 13, 40]
    #     total_duration = [sum(durations[:i + 1]) for i in range(len(durations))]
    #     possible_actions = [[2.5, 0], [0, 1.22], [2.5, 0]]
    #     for i, time in enumerate(total_duration):
    #         if step // 2 < time:
    #             actions = possible_actions[i]
    #             break

    env.step(actions)
    env.render()

    _, reward, done, trunc, _ = env.last()
    total_rew += reward
    # img = Image.fromarray(env.render())
    # frame_list.append(img)
    print(f"step: {step}, {reward}")
    # sleep(0.1)

print(f"total_reward {total_rew}")
# frame_list[0].save(
#     "out.gif", save_all=True, append_images=frame_list[1:], duration=3, loop=0
# )
