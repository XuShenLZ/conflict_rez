from itertools import product
from typing import Dict
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import dill
import pickle

from pytope import Polytope

from confrez.control.compute_sets import (
    compute_initial_states,
    compute_obstacles,
    compute_parking_lines,
    compute_static_vehicles,
    compute_sets,
    interp_along_sets,
)
from confrez.control.utils import plot_car, plot_rl_agent
from confrez.pytypes import VehiclePrediction
from confrez.vehicle_types import VehicleBody

np.random.seed(0)

COLORS = {
    "vehicle_0": {"front": (255, 119, 0), "back": (128, 60, 0)},
    "vehicle_1": {"front": (0, 255, 212), "back": (0, 140, 117)},
    "vehicle_2": {"front": (164, 164, 164), "back": (64, 64, 64)},
    "vehicle_3": {"front": (255, 0, 149), "back": (128, 0, 74)},
}

static_vehicles = compute_static_vehicles()
obstacles = compute_obstacles()
parking_lines = compute_parking_lines()


def plot_continuous_scenario(rl_file_name: str = "4v_rl_traj", L: float = 2.5):
    """
    Plot scenario in the continous world
    """
    init_states = compute_initial_states(
        file_name=rl_file_name, vehicle_body=VehicleBody()
    )

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

    for i, agent in enumerate(init_states):
        state = init_states[agent]
        plot_car(state.x.x, state.x.y, state.e.psi, VehicleBody(), text=str(i))

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()


def plot_grid_scenario(rl_file_name: str = "4v_rl_traj", L: float = 2.5):
    """
    Plot grid world, obstcales, and example agents
    """
    plt.figure()
    for x, y in product(range(14), repeat=2):
        plt.axhline(y=y * L, xmin=0, xmax=14 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=14 * L, color="k", linewidth=1)
    ax = plt.gca()
    # for p in obstacles:
    #     p.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    inflated_obstacles = [
        Polytope([[1 * L, 8 * L], [1 * L, 11 * L], [6 * L, 11 * L], [6 * L, 8 * L]]),
        Polytope([[7 * L, 8 * L], [7 * L, 11 * L], [9 * L, 11 * L], [9 * L, 8 * L]]),
        Polytope(
            [[10 * L, 8 * L], [10 * L, 11 * L], [13 * L, 11 * L], [13 * L, 8 * L]]
        ),
        Polytope([[1 * L, 3 * L], [1 * L, 6 * L], [6 * L, 6 * L], [6 * L, 3 * L]]),
        Polytope([[7 * L, 3 * L], [7 * L, 6 * L], [8 * L, 6 * L], [8 * L, 3 * L]]),
        Polytope([[9 * L, 3 * L], [9 * L, 6 * L], [13 * L, 6 * L], [13 * L, 3 * L]]),
    ]
    for p in inflated_obstacles:
        p.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    with open(rl_file_name + ".pkl", "rb") as f:
        rl_states_history = pickle.load(f)

    for i, agent in enumerate(rl_states_history):
        state = rl_states_history[agent][0]
        color = COLORS[agent]
        text = {"front": f"{i}F", "back": f"{i}B"}
        plot_rl_agent(state, color, ax, text)

    ax.set_xlim(xmin=0, xmax=14 * L)
    ax.set_ylim(ymin=0, ymax=14 * L)
    plt.xticks(np.arange(0, 15 * L, L), range(0, 15))
    plt.yticks(np.arange(0, 15 * L, L), range(0, 15))
    ax.set_aspect("equal")

    plt.tight_layout()


def plot_grid_dynamics(L: float = 2.5):
    """
    Plot agent dynamis in grid world
    """
    plt.figure(figsize=(3.69, 5.44))
    ax = plt.subplot(3, 3, 1)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (0, 3), "back": (1, 2)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("FL", size="xx-large", weight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 3, 2)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (1, 3), "back": (1, 2)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("F", size="xx-large", weight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 3, 3)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (2, 3), "back": (1, 2)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("FR", size="xx-large", weight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 3, 5)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (1, 2), "back": (1, 1)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("Initial / Stop (S)", size="xx-large", weight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 3, 7)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (1, 1), "back": (0, 0)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("BL", size="xx-large", weight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 3, 8)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (1, 1), "back": (1, 0)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("B", size="xx-large", weight="bold")
    ax.axis("off")

    ax = plt.subplot(3, 3, 9)
    for x, y in product(range(4), range(5)):
        plt.axhline(y=y * L, xmin=0, xmax=3 * L, color="k", linewidth=1)
        plt.axvline(x=x * L, ymin=0, ymax=4 * L, color="k", linewidth=1)

    state_v0 = {"front": (1, 1), "back": (2, 0)}
    color_v0 = {"front": (255, 119, 0), "back": (128, 60, 0)}
    text_v0 = {"front": "0F", "back": "0B"}
    text_options = {"size": "x-large"}
    plot_rl_agent(state_v0, color_v0, ax, text_v0, text_options=text_options)
    ax.set_xlim(xmin=0, xmax=3 * L)
    ax.set_ylim(ymin=0, ymax=4 * L)
    plt.xticks(np.arange(0, 4 * L, L), range(0, 4))
    plt.yticks(np.arange(0, 5 * L, L), range(0, 5))
    ax.set_aspect("equal")
    ax.set_title("BR", size="xx-large", weight="bold")
    ax.axis("off")

    plt.tight_layout()


def plot_single_vehicle_spline(rl_file_name: str = "4v_rl_traj"):
    """
    Plot the spline interpolation of single vehicle along the sets
    """
    rl_sets = compute_sets(rl_file_name)

    interp_waypoints = interp_along_sets(
        file_name=rl_file_name, vehicle_body=VehicleBody(), N=30
    )

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)
    agent = "vehicle_0"
    for i, body_sets in enumerate(rl_sets[agent]):
        body_sets["front"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([255, 187, 127]) / 255,
        )
        body_sets["back"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([191, 157, 127]) / 255,
        )

    xypsi = interp_waypoints[agent]
    plt.plot(xypsi[:, 0], xypsi[:, 1], color="k", linewidth=2.5)
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()


def plot_single_vehicle_ws(
    rl_file_name: str = "4v_rl_traj", sol_file_name: str = "v0_zu0"
):
    """
    plot the traj of single vehicle warm start
    """
    zu0: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))

    rl_sets = compute_sets(rl_file_name)

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

    agent = "vehicle_0"
    for i, body_sets in enumerate(rl_sets[agent]):
        body_sets["front"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([255, 187, 127]) / 255,
        )
        body_sets["back"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([191, 157, 127]) / 255,
        )

    x = zu0.x
    y = zu0.y
    plt.plot(x, y, color="k", linewidth=2.5)
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()


def plot_single_vehicle_final(
    rl_file_name: str = "4v_rl_traj",
    sol_file_name: str = "4v_rl_traj_vehicle_0_zufinal",
):
    """
    plot single vehicle final traj
    """
    zu: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))

    rl_sets = compute_sets(rl_file_name)

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

    agent = "vehicle_0"
    for i, body_sets in enumerate(rl_sets[agent]):
        body_sets["front"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([255, 187, 127]) / 255,
        )
        body_sets["back"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([191, 157, 127]) / 255,
        )

    x = zu.x
    y = zu.y
    plt.plot(x, y, color="k", linewidth=2.5)
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()


def plot_single_vehicle_final_w_poses(
    rl_file_name: str = "4v_rl_traj",
    sol_file_name: str = "4v_rl_traj_vehicle_0_zufinal",
):
    """
    plot the traj of single vehicle final with a few poses
    """
    zu: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))

    rl_sets = compute_sets(rl_file_name)

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

    agent = "vehicle_0"
    for i, body_sets in enumerate(rl_sets[agent]):
        body_sets["front"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([255, 187, 127]) / 255,
        )
        body_sets["back"].plot(
            ax,
            # alpha=0.5,
            facecolor=np.array([191, 157, 127]) / 255,
        )

    x = zu.x
    y = zu.y
    plt.plot(x, y, color="k", linewidth=1, linestyle="-.")

    key_stride = 30
    for i in range(2, len(rl_sets[agent]), 2):
        k = key_stride * i

        plot_car(zu.x[k], zu.y[k], zu.psi[k], VehicleBody())
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
    ax.axis("off")
    ax.set_aspect("equal")

    plt.tight_layout()


def plot_multi_vehicle_final(
    rl_file_name: str = "4v_rl_traj",
    sol_file_name: str = "4v_rl_traj_opt",
):
    """
    plot the traj of multi vehicles final
    """
    zu: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))

    rl_sets = compute_sets(rl_file_name)

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

    for agent in rl_sets:
        x = zu[agent].x
        y = zu[agent].y
        plt.plot(
            x,
            y,
            color=np.array(COLORS[agent]["front"]) / 255,
            linewidth=2.5,
            label=agent,
        )
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
        fontsize="large",
    )

    plt.tight_layout()


def plot_multi_vehicle_final_pose_k(
    k,
    rl_file_name: str = "4v_rl_traj",
    sol_file_name: str = "4v_rl_traj_opt",
):
    """
    plot the traj of multi vehicles final with pose
    """
    zu: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))
    dt = zu["vehicle_0"].t[1] - zu["vehicle_0"].t[0]
    print(f"Current time t = {k * dt}")

    rl_sets = compute_sets(rl_file_name)

    plt.figure()
    ax = plt.gca()

    for obstacle in obstacles:
        obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))

    for obstacle in static_vehicles:
        obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")

    for line in parking_lines:
        plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

    for i, agent in enumerate(rl_sets):
        x = zu[agent].x
        y = zu[agent].y
        plt.plot(
            x,
            y,
            color=np.array(COLORS[agent]["front"]) / 255,
            linewidth=2.5,
            label=agent,
            zorder=i,
        )

        plot_car(
            zu[agent].x[k],
            zu[agent].y[k],
            zu[agent].psi[k],
            VehicleBody(),
            text=i,
            zorder=10 + i,
        )
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.96, 0.97),
        fontsize="large",
    )

    plt.tight_layout()


def plot_multi_vehicle_states(sol_file_name: str = "4v_rl_traj_opt"):
    """
    plot the state & input profile of multi vehicles final
    """
    zu: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))
    print(zu["vehicle_0"].t[-1] / 9)

    fig = plt.figure(figsize=(14, 8))

    for agent in sorted(zu):
        ax = plt.subplot(2, 2, 1)
        t = zu[agent].t
        v = zu[agent].v
        ax.plot(
            t,
            v,
            color=np.array(COLORS[agent]["front"]) / 255,
            linewidth=2,
            label=agent,
        )
        ax.set_ylabel(
            "Speed (m/s)", fontname="Times New Roman", fontsize=20, fontweight="bold"
        )
        ax.tick_params(axis="y", labelsize="x-large")
        ax.get_xaxis().set_visible(False)
        ax.legend(fontsize="x-large")

        ax = plt.subplot(2, 2, 2)
        t = zu[agent].t
        u_steer = zu[agent].u_steer
        ax.plot(
            t,
            u_steer,
            color=np.array(COLORS[agent]["front"]) / 255,
            linewidth=2,
            label=agent,
        )
        ax.set_ylabel(
            "Steering Angle (rad)",
            fontname="Times New Roman",
            fontsize=20,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize="x-large")
        ax.get_xaxis().set_visible(False)
        ax.legend(fontsize="x-large")

        ax = plt.subplot(2, 2, 3)
        t = zu[agent].t
        a = zu[agent].u_a
        ax.plot(
            t,
            a,
            color=np.array(COLORS[agent]["front"]) / 255,
            linewidth=2,
            label=agent,
        )
        ax.set_ylabel(
            "Acceleration ($m/s^2$)",
            fontname="Times New Roman",
            fontsize=20,
            fontweight="bold",
            math_fontfamily="cm",
        )
        ax.set_xlabel(
            "Time (s)", fontname="Times New Roman", fontsize=20, fontweight="bold"
        )
        ax.tick_params(axis="x", labelsize="x-large")
        ax.tick_params(axis="y", labelsize="x-large")
        ax.legend(fontsize="x-large")

        ax = plt.subplot(2, 2, 4)
        t = zu[agent].t
        u_steer_dot = zu[agent].u_steer_dot
        ax.plot(
            t,
            u_steer_dot,
            color=np.array(COLORS[agent]["front"]) / 255,
            linewidth=2,
            label=agent,
        )
        ax.set_ylabel(
            "Steering Rate (rad/s)",
            fontname="Times New Roman",
            fontsize=20,
            fontweight="bold",
        )
        ax.set_xlabel(
            "Time (s)", fontname="Times New Roman", fontsize=20, fontweight="bold"
        )
        ax.tick_params(axis="x", labelsize="x-large")
        ax.tick_params(axis="y", labelsize="x-large")
        ax.legend(fontsize="x-large")
    # ax.set_xlim(xmin=0, xmax=13 * 2.5)
    # ax.set_ylim(ymin=3 * 2.5, ymax=11 * 2.5)

    # plt.tight_layout()
    fig.subplots_adjust(wspace=0.22, hspace=0)
    fig.savefig("state_input_profile.pdf", bbox_inches="tight")


def generate_animation(sol_file_name: str = "4v_rl_traj_opt", interval: int = None):
    """
    generate animation
    """
    zu: Dict[str, VehiclePrediction] = dill.load(open(f"{sol_file_name}.pkl", "rb"))

    if interval is None:
        interval = int((zu["vehicle_0"].t[1] - zu["vehicle_0"].t[0]) * 1000)

    fig = plt.figure()
    ax = plt.gca()

    def plot_frame(i):
        ax.clear()
        for obstacle in obstacles:
            obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))
        for obstacle in static_vehicles:
            obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")
        for line in parking_lines:
            plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

        for j, agent in enumerate(sorted(zu)):
            ax.plot(
                zu[agent].x,
                zu[agent].y,
                color=np.array(COLORS[agent]["front"]) / 255.0,
                label=agent,
                zorder=j,
            )
            plot_car(
                zu[agent].x[i],
                zu[agent].y[i],
                zu[agent].psi[i],
                VehicleBody(),
                text=j,
                zorder=10 + j,
            )
        ax.axis("off")
        ax.set_aspect("equal")
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.96, 0.97),
            fontsize="large",
        )
        plt.tight_layout()

    ani = FuncAnimation(
        fig,
        plot_frame,
        frames=len(zu["vehicle_0"].t),
        interval=interval,
        repeat=True,
    )

    fps = int(1000 / interval)
    writer = FFMpegWriter(fps=fps)
    ani.save(f"{sol_file_name}_{fps}fps_animation.mp4", writer=writer)


def plot_training_rewards(csv_file_name: str, smoothing_factor: float = 0.92):
    """
    plot the training rewards
    """
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    df = pd.read_csv(csv_file_name)

    smoothed_df = df.ewm(alpha=(1 - smoothing_factor)).mean()

    fig = plt.figure(figsize=(14, 4))

    sns.lineplot(x="Step", y="Value", data=smoothed_df, linewidth=3)

    ax = plt.gca()
    ax.set_ylabel(
        "Reward",
        fontname="Times New Roman",
        fontsize=25,
        fontweight="bold",
    )
    ax.set_xlabel("Step", fontname="Times New Roman", fontsize=25, fontweight="bold")
    ax.tick_params(axis="x", labelsize="x-large")
    ax.tick_params(axis="y", labelsize="x-large")
    ax.ticklabel_format(axis="y", style="sci")

    # plt.plot(smoothed_df["Value"])

    # plt.tight_layout()
    fig.savefig("training_rewards.pdf", bbox_inches="tight")


def main():
    """
    main function
    """
    rl_file_name = "4v_rl_traj"
    # plot_grid_scenario(rl_file_name=rl_file_name)
    # plot_single_vehicle_spline(rl_file_name=rl_file_name)
    sv_ws_file_name = "4v_rl_traj_vehicle_0_zu0"
    # plot_single_vehicle_ws(rl_file_name=rl_file_name, sol_file_name=sv_ws_file_name)
    sv_sol_file_name = "4v_rl_traj_vehicle_0_zufinal"
    # plot_single_vehicle_final(rl_file_name=rl_file_name, sol_file_name=sv_sol_file_name)
    # plot_single_vehicle_final_w_poses(
    # rl_file_name=rl_file_name, sol_file_name=sv_sol_file_name
    # )

    mv_sol_file_name = "4v_rl_traj_opt"
    # plot_multi_vehicle_final(rl_file_name=rl_file_name, sol_file_name=mv_sol_file_name)

    # plot_multi_vehicle_final_pose_k(30, rl_file_name=rl_file_name, sol_file_name=mv_sol_file_name)
    # plot_multi_vehicle_final_pose_k(85, rl_file_name=rl_file_name, sol_file_name=mv_sol_file_name)
    # plot_multi_vehicle_final_pose_k(150, rl_file_name=rl_file_name, sol_file_name=mv_sol_file_name)
    # plot_multi_vehicle_final_pose_k(240, rl_file_name=rl_file_name, sol_file_name=mv_sol_file_name)

    # plot_multi_vehicle_states(sol_file_name=mv_sol_file_name)

    # generate_animation(sol_file_name=mv_sol_file_name, interval=40)

    csv_file_name = (
        "run-DQN-CNN-4v-fixed_11-01-2022_11-03-10_1-tag-eval_mean_epi_rewards.csv"
    )
    plot_training_rewards(csv_file_name=csv_file_name)

    plt.show()


if __name__ == "__main__":
    main()
