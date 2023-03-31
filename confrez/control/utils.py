from typing import Dict, Tuple
import matplotlib.pyplot as plt
from math import cos, sin, pi
import numpy as np
from pytope import Polytope

from scipy.spatial.transform import Rotation as Rot

from confrez.vehicle_types import VehicleBody


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle
    Parameters
    ----------
    angle :
    Returns
    -------
    A 2D rotation matrix
    Examples
    --------
    >>> angle_mod(-4.0)
    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def plot_car(
    x: float,
    y: float,
    yaw: float,
    vehicle_body: VehicleBody,
    text=None,
    zorder=10,
    car_color="k",
    fill_color=None,
):
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(vehicle_body.xy[:, 0], vehicle_body.xy[:, 1]):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    if fill_color is not None:
        plt.fill(car_outline_x, car_outline_y, color=fill_color, zorder=zorder)

    plt.plot(car_outline_x, car_outline_y, color=car_color, zorder=zorder)

    plt.plot(
        [x, x + cos(yaw) * vehicle_body.wb],
        [y, y + sin(yaw) * vehicle_body.wb],
        color=car_color,
        linestyle="None",
        marker="D",
        markersize=2.5,
        zorder=zorder,
    )

    plt.arrow(
        x,
        y,
        0.2 * vehicle_body.wb * np.cos(yaw),
        0.2 * vehicle_body.wb * np.sin(yaw),
        color=car_color,
        width=0.04,
        head_width=0.5,
        zorder=zorder,
    )

    if text is not None:
        plt.annotate(
            text,
            xy=(
                x + cos(yaw) * 0.7 * vehicle_body.wb,
                y + sin(yaw) * 0.7 * vehicle_body.wb,
            ),
            ha="center",
            va="center",
            zorder=zorder,
        )


def plot_rl_agent(
    state: Dict[str, Tuple[int, int]],
    color: Dict[str, Tuple[int, int]],
    ax,
    text: Dict[str, str] = None,
    text_options: Dict = {},
    L: float = 2.5,
):
    for i in state:
        x, y = state[i]
        c = np.array(color[i]) / 255
        p = Polytope(
            [
                [x * L, y * L],
                [x * L, (y + 1) * L],
                [(x + 1) * L, (y + 1) * L],
                [(x + 1) * L, y * L],
            ]
        )
        p.plot(ax, facecolor=c)

        if text is not None:
            t = text[i]
            if i == "back":
                plt.annotate(
                    text=t,
                    xy=((x + 0.5) * L, (y + 0.5) * L),
                    ha="center",
                    va="center",
                    color=(1, 1, 1),
                    **text_options
                )
            else:
                plt.annotate(
                    text=t,
                    xy=((x + 0.5) * L, (y + 0.5) * L),
                    ha="center",
                    va="center",
                    color=(0, 0, 0),
                    **text_options
                )
