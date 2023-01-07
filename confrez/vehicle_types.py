from dataclasses import dataclass, field
import numpy as np

from confrez.pytypes import *
from confrez.obstacle_types import BasePolytopeObstacle


@dataclass
class VehicleBody(BasePolytopeObstacle):
    """
    Class to represent the body of a rectangular vehicle in the body frame
    verticies everywhere are computed with the assumption that 0 degrees has the vehicle pointing east.
    matrices are computed for 0 degrees, with the assumption that they are rotated by separate code.
    """

    # Wheelbase
    hf: float = field(default=0.8)  # Front hang length
    wb: float = field(default=2.5)  # Wheelbase
    hr: float = field(default=0.6)  # Rear hang length

    offset: float = field(default=0)  # Offset from rear axis center to vehicle center
    lf: float = field(default=0)  # From rear axis center to front bumper
    lr: float = field(default=0)  # From rear axis center to rear bumper

    # Total Length and width
    l: float = field(default=0)
    w: float = field(default=1.8)

    # Circle Approximation
    cr: float = field(default=0)  # Offset of the first circle center in front
    cf: float = field(default=0)  # Offset of the first circle center at rear
    num_circles: int = field(default=3)

    def __post_init__(self):
        self.offset = self.wb / 2
        self.lf = self.wb + self.hf
        self.lr = self.hr
        self.l = self.lf + self.lr

        self.cf = 2.45
        self.cr = -0.2
        self.num_circles = 4

        self.__calc_V__()
        self.__calc_A_b__()
        return

    def __calc_V__(self):
        xy = np.array(
            [
                [self.lf, self.w / 2],
                [-self.lr, self.w / 2],
                [-self.lr, -self.w / 2],
                [self.lf, -self.w / 2],
                [self.lf, self.w / 2],
            ]
        )

        V = xy[:-1, :]

        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "V", V)
        return

    def __calc_A_b__(self):
        A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

        b = np.array([self.lf, self.w / 2, self.lr, self.w / 2])
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)
        return


@dataclass
class VehicleConfig(PythonMsg):
    """
    vehicle configuration class
    """

    # Vehicle Limits
    v_max: float = field(default=2.5)  # maximum velocity
    v_min: float = field(default=-2.5)  # minimum velocity
    a_max: float = field(default=1.5)  # maximum acceleration
    a_min: float = field(default=-1.5)  # minimum acceleration
    delta_max: float = field(default=0.85)  # maximum steering angle
    delta_min: float = field(default=-0.85)  # minimum steering angle
    w_delta_max: float = field(default=1)  # maximum angular velocity of steering angles
    w_delta_min: float = field(
        default=-1
    )  # minimum angular velocity of steering angles
