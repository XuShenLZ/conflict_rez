import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from pytope import Polytope
import numpy as np

from collections import defaultdict
from confrez.pytypes import VehicleState

from confrez.vehicle_types import VehicleBody


def compute_sets(file_name: str, L=2.5) -> Dict[str, List[Dict[str, Polytope]]]:
    """
    compute sets based on RL state history
    """
    with open(file_name + ".pkl", "rb") as f:
        rl_states_history = pickle.load(f)

    print(rl_states_history)

    V = [[0, 0], [0, L], [L, 0], [L, L]]

    base_sets = {
        "front": defaultdict(lambda: Polytope(V)),
        "back": defaultdict(lambda: Polytope(V)),
    }

    base_sets["front"][(1, 1)] = Polytope(
        [
            [-L / 4, L / 4],
            [L / 2, L],
            [L, L],
            [L, L / 2],
            [L / 4, -L / 4],
        ]
    )

    base_sets["front"][(-1, 1)] = Polytope(
        [
            [L * 3 / 4, -L / 4],
            [0, L / 2],
            [0, L],
            [L / 2, L],
            [L * 5 / 4, L / 4],
        ]
    )

    base_sets["front"][(1, -1)] = Polytope(
        [
            [-L / 4, L * 3 / 4],
            [L / 2, 0],
            [L, 0],
            [L, L / 2],
            [L / 4, L * 5 / 4],
        ]
    )

    base_sets["front"][(-1, -1)] = Polytope(
        [
            [0, 0],
            [0, L / 2],
            [L * 3 / 4, L * 5 / 4],
            [L * 5 / 4, L * 3 / 4],
            [L / 2, 0],
        ]
    )

    base_sets["back"][(1, 1)] = Polytope(
        [
            [0, 0],
            [0, L / 2],
            [L * 3 / 4, L * 5 / 4],
            [L * 5 / 4, L * 3 / 4],
            [L / 2, 0],
        ]
    )

    base_sets["back"][(-1, 1)] = Polytope(
        [
            [-L / 4, L * 3 / 4],
            [L / 2, 0],
            [L, 0],
            [L, L / 2],
            [L / 4, L * 5 / 4],
        ]
    )

    base_sets["back"][(1, -1)] = Polytope(
        [
            [L * 3 / 4, -L / 4],
            [0, L / 2],
            [0, L],
            [L / 2, L],
            [L * 5 / 4, L / 4],
        ]
    )

    base_sets["back"][(-1, -1)] = Polytope(
        [
            [-L / 4, L / 4],
            [L / 2, L],
            [L, L],
            [L, L / 2],
            [L / 4, -L / 4],
        ]
    )

    rl_sets: Dict[str, List[Dict[str, Polytope]]] = {
        agent: [] for agent in rl_states_history
    }

    for agent in rl_sets:
        for state in rl_states_history[agent]:
            dir = (
                state["front"][0] - state["back"][0],
                state["front"][1] - state["back"][1],
            )
            body_sets = {}
            for body in ["front", "back"]:
                offset = np.array(state[body]) * L
                body_sets[body] = base_sets[body][dir] + offset
            rl_sets[agent].append(body_sets)

    return rl_sets


def compute_initial_states(
    file_name: str, vehicle_body: VehicleBody, L=2.5
) -> Dict[str, VehicleState]:
    with open(file_name + ".pkl", "rb") as f:
        rl_states_history = pickle.load(f)

    initial_states = {agent: VehicleState() for agent in rl_states_history}
    for agent in rl_states_history:
        front = rl_states_history[agent][0]["front"]
        back = rl_states_history[agent][0]["back"]
        dir = (front[0] - back[0], front[1] - back[1])
        psi = np.arctan2(dir[1], dir[0])
        initial_states[agent].e.psi = psi

        if dir[1] == 0:
            center = np.array([max(front[0], back[0]) * L, (front[1] + 0.5) * L])
        else:
            center = np.array([(front[0] + 0.5) * L, max(front[1], back[1]) * L])

        wb = vehicle_body.wb
        initial_states[agent].x.x = center[0] - wb / 2 * np.cos(psi)
        initial_states[agent].x.y = center[1] - wb / 2 * np.sin(psi)

    return initial_states


def main():
    """
    main function
    """
    file_name = "1v_rl_traj"
    rl_sets = compute_sets(file_name)
    init_states = compute_initial_states(
        file_name=file_name, vehicle_body=VehicleBody()
    )
    print(init_states)

    for i, body_sets in enumerate(rl_sets["vehicle_0"]):
        ax = plt.subplot(2, 4, i + 1)
        body_sets["front"].plot(ax, facecolor="g")
        body_sets["back"].plot(ax, facecolor="r")
        ax.set_xlim(xmin=-2.5, xmax=15 * 2.5)
        ax.set_ylim(ymin=-2.5, ymax=15 * 2.5)
        ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
