import matplotlib.pyplot as plt
from math import cos, sin, tan, pi
import numpy as np

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


def plot_car(x: float, y: float, yaw: float, vehicle_body: VehicleBody):
    car_color = "-k"
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(vehicle_body.xy[:, 0], vehicle_body.xy[:, 1]):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0] + x)
        car_outline_y.append(converted_xy[1] + y)

    plt.plot(car_outline_x, car_outline_y, car_color)
    plt.plot(
        [x, x + cos(yaw) * vehicle_body.wb],
        [y, y + sin(yaw) * vehicle_body.wb],
        "kD",
        markersize=2.5,
    )
