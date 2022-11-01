from itertools import product
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from pytope import Polytope
from math import ceil
import numpy as np

from collections import defaultdict
from confrez.control.bezier import BezierPlanner
from confrez.control.utils import pi_2_pi, plot_car, plot_rl_agent
from confrez.pytypes import VehicleState
from confrez.vehicle_types import VehicleBody

np.random.seed(0)

COLORS = {
    "vehicle_0": {"front": (255, 119, 0), "back": (128, 60, 0)},
    "vehicle_1": {"front": (0, 255, 212), "back": (0, 140, 117)},
    "vehicle_2": {"front": (164, 164, 164), "back": (64, 64, 64)},
    "vehicle_3": {"front": (255, 0, 149), "back": (128, 0, 74)},
}


def compute_sets(file_name: str, L=2.5) -> Dict[str, List[Dict[str, Polytope]]]:
    """
    compute sets based on RL state history
    """
    with open(file_name + ".pkl", "rb") as f:
        rl_states_history = pickle.load(f)

    V = [[0, 0], [0, L], [L, 0], [L, L]]

    # When facing towards left/right/up/down, the set is a standard square
    base_sets = {
        "front": defaultdict(lambda: Polytope(V)),
        "back": defaultdict(lambda: Polytope(V)),
    }

    # # Pentagon-shape sets for all other directions
    # base_sets["front"][(1, 1)] = Polytope(
    #     [
    #         [-L / 4, L / 4],
    #         [L / 2, L],
    #         [L, L],
    #         [L, L / 2],
    #         [L / 4, -L / 4],
    #     ]
    # )

    # base_sets["front"][(-1, 1)] = Polytope(
    #     [
    #         [L * 3 / 4, -L / 4],
    #         [0, L / 2],
    #         [0, L],
    #         [L / 2, L],
    #         [L * 5 / 4, L / 4],
    #     ]
    # )

    # base_sets["front"][(1, -1)] = Polytope(
    #     [
    #         [-L / 4, L * 3 / 4],
    #         [L / 2, 0],
    #         [L, 0],
    #         [L, L / 2],
    #         [L / 4, L * 5 / 4],
    #     ]
    # )

    # base_sets["front"][(-1, -1)] = Polytope(
    #     [
    #         [0, 0],
    #         [0, L / 2],
    #         [L * 3 / 4, L * 5 / 4],
    #         [L * 5 / 4, L * 3 / 4],
    #         [L / 2, 0],
    #     ]
    # )

    # base_sets["back"][(1, 1)] = Polytope(
    #     [
    #         [0, 0],
    #         [0, L / 2],
    #         [L * 3 / 4, L * 5 / 4],
    #         [L * 5 / 4, L * 3 / 4],
    #         [L / 2, 0],
    #     ]
    # )

    # base_sets["back"][(-1, 1)] = Polytope(
    #     [
    #         [-L / 4, L * 3 / 4],
    #         [L / 2, 0],
    #         [L, 0],
    #         [L, L / 2],
    #         [L / 4, L * 5 / 4],
    #     ]
    # )

    # base_sets["back"][(1, -1)] = Polytope(
    #     [
    #         [L * 3 / 4, -L / 4],
    #         [0, L / 2],
    #         [0, L],
    #         [L / 2, L],
    #         [L * 5 / 4, L / 4],
    #     ]
    # )

    # base_sets["back"][(-1, -1)] = Polytope(
    #     [
    #         [-L / 4, L / 4],
    #         [L / 2, L],
    #         [L, L],
    #         [L, L / 2],
    #         [L / 4, -L / 4],
    #     ]
    # )

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


def convert_rl_states(
    states: Dict[str, Tuple[int, int]], vehicle_body: VehicleBody, L: float = 2.5
) -> VehicleState:
    vehicle_state = VehicleState()
    front = states["front"]
    back = states["back"]
    dir = (front[0] - back[0], front[1] - back[1])
    psi = np.arctan2(dir[1], dir[0])
    vehicle_state.e.psi = psi

    if dir[1] == 0:
        center = np.array([max(front[0], back[0]) * L, (front[1] + 0.5) * L])
    elif dir[0] == 0:
        center = np.array([(front[0] + 0.5) * L, max(front[1], back[1]) * L])
    else:
        center = np.array([max(front[0], back[0]) * L, max(front[1], back[1]) * L])

    wb = vehicle_body.wb
    # Vehicle reference point
    vehicle_state.x.x = center[0] - wb / 2 * np.cos(psi)
    vehicle_state.x.y = center[1] - wb / 2 * np.sin(psi)

    return vehicle_state


def interp_along_sets(file_name: str, vehicle_body: VehicleBody, N: int):
    """
    Compute piecewise Bezier interpolation of vehicle reference point (x, y, psi) using the RL sets
    """
    with open(file_name + ".pkl", "rb") as f:
        rl_states_history = pickle.load(f)

    path = {agent: [] for agent in rl_states_history}

    planner = BezierPlanner(offset=2.5)

    for agent in rl_states_history:
        for i in range(len(rl_states_history[agent]) - 1):
            start_state = convert_rl_states(
                states=rl_states_history[agent][i], vehicle_body=vehicle_body
            )
            end_state = convert_rl_states(
                states=rl_states_history[agent][i + 1], vehicle_body=vehicle_body
            )

            if rl_states_history[agent][i + 1] == rl_states_history[agent][i]:
                # Vehicle does not move
                path_segment = np.array(
                    [
                        [start_state.x.x, start_state.x.y, start_state.e.psi]
                        for _ in range(N)
                    ]
                )
            elif start_state.e.psi == end_state.e.psi:
                # Vehicle does not turn
                path_segment = np.array(
                    [
                        [start_state.x.x, start_state.x.y, start_state.e.psi]
                        for _ in range(N)
                    ]
                )
                path_segment[:, 0] = np.linspace(
                    start_state.x.x, end_state.x.x, N, endpoint=False
                )
                path_segment[:, 1] = np.linspace(
                    start_state.x.y, end_state.x.y, N, endpoint=False
                )
            else:
                if (
                    rl_states_history[agent][i + 1]["front"]
                    == rl_states_history[agent][i]["back"]
                ):
                    # Vehicle goes backward
                    angle_offset = np.pi
                else:
                    angle_offset = 0

                start_state.e.psi = pi_2_pi(start_state.e.psi + angle_offset)
                end_state.e.psi = pi_2_pi(end_state.e.psi + angle_offset)

                path_segment = planner.interpolate(
                    start_state=start_state, end_state=end_state, N=N
                )

                path_segment[:, 2] -= angle_offset

            path[agent].append(path_segment)

        final_state = convert_rl_states(
            states=rl_states_history[agent][-1], vehicle_body=vehicle_body
        )
        path[agent].append(
            np.array([[final_state.x.x, final_state.x.y, final_state.e.psi]])
        )

        path[agent] = np.vstack(path[agent])
        path[agent][:, 2] = np.unwrap(path[agent][:, 2])

    return path


def compute_initial_states(
    file_name: str, vehicle_body: VehicleBody, L=2.5
) -> Dict[str, VehicleState]:
    with open(file_name + ".pkl", "rb") as f:
        rl_states_history = pickle.load(f)

    initial_states = {
        agent: convert_rl_states(
            states=rl_states_history[agent][0], vehicle_body=vehicle_body
        )
        for agent in rl_states_history
    }

    return initial_states


def compute_obstacles(
    L: float = 2.5, vb: VehicleBody = VehicleBody()
) -> List[Polytope]:
    obstacles = []
    # Bottom left
    obstacles.append(
        Polytope(
            [
                [1.5 * L - vb.w / 2, 3 * L],
                [1.5 * L - vb.w / 2, 5.5 * L],
                [5.5 * L + vb.w / 2, 5.5 * L],
                [5.5 * L + vb.w / 2, 3 * L],
            ]
        )
    )
    # Bottom center
    obstacles.append(
        Polytope(
            [
                [7.5 * L - vb.w / 2, 3 * L],
                [7.5 * L - vb.w / 2, 5.5 * L],
                [7.5 * L + vb.w / 2, 5.5 * L],
                [7.5 * L + vb.w / 2, 3 * L],
            ]
        )
    )
    # Bottom right
    obstacles.append(
        Polytope(
            [
                [9.5 * L - vb.w / 2, 3 * L],
                [9.5 * L - vb.w / 2, 5.5 * L],
                [12.5 * L + vb.w / 2, 5.5 * L],
                [12.5 * L + vb.w / 2, 3 * L],
            ]
        )
    )
    # Top left
    obstacles.append(
        Polytope(
            [
                [1.5 * L - vb.w / 2, 8.5 * L],
                [1.5 * L - vb.w / 2, 11 * L],
                [5.5 * L + vb.w / 2, 11 * L],
                [5.5 * L + vb.w / 2, 8.5 * L],
            ]
        )
    )
    # Top center
    obstacles.append(
        Polytope(
            [
                [7.5 * L - vb.w / 2, 8.5 * L],
                [7.5 * L - vb.w / 2, 11 * L],
                [8.5 * L + vb.w / 2, 11 * L],
                [8.5 * L + vb.w / 2, 8.5 * L],
            ]
        )
    )
    # Top right
    obstacles.append(
        Polytope(
            [
                [10.5 * L - vb.w / 2, 8.5 * L],
                [10.5 * L - vb.w / 2, 11 * L],
                [12.5 * L + vb.w / 2, 11 * L],
                [12.5 * L + vb.w / 2, 8.5 * L],
            ]
        )
    )

    return obstacles


def compute_parking_lines(L: float = 2.5):
    """
    compute parking lines to draw
    """
    lines = [
        np.array([[L, 3 * L], [13 * L, 3 * L]]),
        np.array([[L, 11 * L], [13 * L, 11 * L]]),
    ]

    for i in range(1, 14):
        lines.append(np.array([[i * L, 3 * L], [i * L, 5.5 * L]]))
        lines.append(np.array([[i * L, 8.5 * L], [i * L, 11 * L]]))

    return lines


def compute_static_vehicles(L: float = 2.5, vb: VehicleBody = VehicleBody()):
    """
    compute static vehicles
    """
    vehicles = []

    # Lower lane
    for i in range(1, 5):
        random_offset = np.random.sample() * 0.7 * L
        vehicles.append(
            Polytope(
                [
                    [(i + 0.5) * L - vb.w / 2, 5.5 * L - random_offset],
                    [(i + 0.5) * L + vb.w / 2, 5.5 * L - random_offset],
                    [(i + 0.5) * L + vb.w / 2, 5.5 * L - random_offset - vb.l],
                    [(i + 0.5) * L - vb.w / 2, 5.5 * L - random_offset - vb.l],
                ]
            )
        )

    for i in [5, 7]:
        vehicles.append(
            Polytope(
                [
                    [(i + 0.5) * L - vb.w / 2, 5.5 * L],
                    [(i + 0.5) * L + vb.w / 2, 5.5 * L],
                    [(i + 0.5) * L + vb.w / 2, 5.5 * L - vb.l],
                    [(i + 0.5) * L - vb.w / 2, 5.5 * L - vb.l],
                ]
            )
        )

    for i in range(9, 13):
        random_offset = np.random.sample() * 0.7 * L
        vehicles.append(
            Polytope(
                [
                    [(i + 0.5) * L - vb.w / 2, 5.5 * L - random_offset],
                    [(i + 0.5) * L + vb.w / 2, 5.5 * L - random_offset],
                    [(i + 0.5) * L + vb.w / 2, 5.5 * L - random_offset - vb.l],
                    [(i + 0.5) * L - vb.w / 2, 5.5 * L - random_offset - vb.l],
                ]
            )
        )

    # Upper lane
    for i in range(1, 5):
        random_offset = np.random.sample() * 0.7 * L
        vehicles.append(
            Polytope(
                [
                    [(i + 0.5) * L - vb.w / 2, 8.5 * L + random_offset],
                    [(i + 0.5) * L + vb.w / 2, 8.5 * L + random_offset],
                    [(i + 0.5) * L + vb.w / 2, 8.5 * L + random_offset + vb.l],
                    [(i + 0.5) * L - vb.w / 2, 8.5 * L + random_offset + vb.l],
                ]
            )
        )

    for i in [5, 7, 8]:
        vehicles.append(
            Polytope(
                [
                    [(i + 0.5) * L - vb.w / 2, 8.5 * L],
                    [(i + 0.5) * L + vb.w / 2, 8.5 * L],
                    [(i + 0.5) * L + vb.w / 2, 8.5 * L + vb.l],
                    [(i + 0.5) * L - vb.w / 2, 8.5 * L + vb.l],
                ]
            )
        )

    for i in range(10, 13):
        random_offset = np.random.sample() * 0.7 * L
        vehicles.append(
            Polytope(
                [
                    [(i + 0.5) * L - vb.w / 2, 8.5 * L + random_offset],
                    [(i + 0.5) * L + vb.w / 2, 8.5 * L + random_offset],
                    [(i + 0.5) * L + vb.w / 2, 8.5 * L + random_offset + vb.l],
                    [(i + 0.5) * L - vb.w / 2, 8.5 * L + random_offset + vb.l],
                ]
            )
        )

    return vehicles


def main():
    """
    main function
    """
    L = 2.5
    file_name = "4v_rl_traj"
    rl_sets = compute_sets(file_name)
    init_states = compute_initial_states(
        file_name=file_name, vehicle_body=VehicleBody()
    )
    print(init_states)
    obstacles = compute_obstacles()

    interp_waypoints = interp_along_sets(
        file_name=file_name, vehicle_body=VehicleBody(), N=30
    )

    # ================= Plot strategy guided sets of all vehicles ====================
    plt.figure()
    max_num_sets = max(list(map(lambda x: len(x), rl_sets.values())))
    ncol = 4
    nrow = ceil(max_num_sets / ncol)
    for agent in rl_sets:
        for i, body_sets in enumerate(rl_sets[agent]):
            ax = plt.subplot(nrow, ncol, i + 1)
            for p in obstacles:
                p.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))
            body_sets["front"].plot(
                ax,
                # alpha=0.5,
                facecolor=np.array(COLORS[agent]["front"]) / 255.0,
            )
            body_sets["back"].plot(
                ax,
                # alpha=0.5,
                facecolor=np.array(COLORS[agent]["back"]) / 255.0,
            )
            ax.set_xlim(xmin=0, xmax=13 * 2.5)
            ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
            # ax.axis('off')
            ax.set_aspect("equal")

    plt.figure()
    plt.subplot(1, 2, 1)
    for agent in interp_waypoints:
        xypsi = interp_waypoints[agent]
        plt.plot(xypsi[:, 0], xypsi[:, 1], label=agent)
    plt.axis("equal")
    plt.legend()
    plt.title("X-Y of agents")

    plt.subplot(1, 2, 2)
    for agent in interp_waypoints:
        xypsi = interp_waypoints[agent]
        plt.plot(xypsi[:, 2], label=agent)
    plt.legend()
    plt.title("Heading of agents")

    plt.tight_layout()

    plt.show()
    print("done")


if __name__ == "__main__":
    main()
