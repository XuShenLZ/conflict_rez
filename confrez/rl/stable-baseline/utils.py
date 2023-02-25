from typing import Dict, Union

import gym
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
from confrez.rl.pklot_env import parallel_env
import matplotlib.pyplot as plt


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        verbose=0,
    ):
        super(TensorboardCallback, self).__init__(verbose)

        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=False,
                deterministic=False,
            )

            self.logger.record("eval/mean_epi_rewards", episode_rewards)

            self.logger.record("eval/mean_epi_lengths", episode_lengths)

        return True


class ProcessMonitor(object):
    """
    A visualization helper to show the current observations of all agents, and the actions to apply
    """

    def __init__(self, env: parallel_env) -> None:
        self.n_agents = env.n_vehicles
        self._action_to_inputs = env._action_to_inputs

        self.n_col = 2
        self.n_row = int(np.ceil(self.n_agents / self.n_col))

        self.observation_size = env.window_size

    def show(self, observations: Dict, actions: Dict, notes: str = None):
        for i, agent in enumerate(actions.keys()):
            plt.subplot(self.n_row, self.n_col, i + 1)
            plt.imshow(observations[agent])
            inputs = self._action_to_inputs[actions[agent]]
            plt.title(f"{agent}: d={inputs[0]}, a={inputs[1]:.2f}")

        if notes is not None:
            plt.suptitle(notes)

        plt.tight_layout()
        plt.show()
