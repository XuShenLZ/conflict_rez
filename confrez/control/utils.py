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


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi
