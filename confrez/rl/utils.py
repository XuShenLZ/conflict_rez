from typing import Dict

import numpy as np
from confrez.rl.pklot_env import parallel_env
import matplotlib.pyplot as plt


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
