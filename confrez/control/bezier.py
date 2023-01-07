"""
Path planning with Bezier curve.
author: Atsushi Sakai(@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from confrez.control.utils import pi_2_pi

from confrez.vehicle_types import VehicleBody, VehicleState


class BezierPlanner(object):
    """
    Bezier curve path planner
    """

    def __init__(self, offset: float):
        self.offset = offset

    def interpolate(self, start_state: VehicleState, end_state: VehicleState, N):
        """
        interpolate N states between start and end state. Not including the end state
        """
        sx = start_state.x.x
        sy = start_state.x.y
        syaw = start_state.e.psi

        ex = end_state.x.x
        ey = end_state.x.y
        eyaw = end_state.e.psi

        dist = np.hypot(sx - ex, sy - ey) / self.offset
        control_points = np.array(
            [
                [sx, sy],
                [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
                [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
                [ex, ey],
            ]
        )

        xy_path = self.calc_bezier_path(control_points, n_points=N)

        derivatives_cp = self.bezier_derivatives_control_points(control_points, 2)
        yaws = []
        for t in np.linspace(0, 1, N, endpoint=False):
            dt = self.bezier(t, derivatives_cp[1])
            yaws.append(np.arctan2(dt[1], dt[0]))

        yaws = np.array(yaws)

        path = np.insert(xy_path, 2, yaws, axis=1)

        return path

    def calc_4points_bezier_path(self, sx, sy, syaw, ex, ey, eyaw, offset):
        """
        Compute control points and path given start and end position.
        :param sx: (float) x-coordinate of the starting point
        :param sy: (float) y-coordinate of the starting point
        :param syaw: (float) yaw angle at start
        :param ex: (float) x-coordinate of the ending point
        :param ey: (float) y-coordinate of the ending point
        :param eyaw: (float) yaw angle at the end
        :param offset: (float)
        :return: (numpy array, numpy array)
        """
        dist = np.hypot(sx - ex, sy - ey) / offset
        control_points = np.array(
            [
                [sx, sy],
                [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
                [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
                [ex, ey],
            ]
        )

        path = self.calc_bezier_path(control_points, n_points=self.N)

        return path, control_points

    def calc_bezier_path(self, control_points, n_points=100):
        """
        Compute bezier path (trajectory) given control points.
        :param control_points: (numpy array)
        :param n_points: (int) number of points in the trajectory
        :return: (numpy array)
        """
        traj = []
        for t in np.linspace(0, 1, n_points, endpoint=False):
            traj.append(self.bezier(t, control_points))

        return np.array(traj)

    def bernstein_poly(self, n, i, t):
        """
        Bernstein polynom.
        :param n: (int) polynom degree
        :param i: (int)
        :param t: (float)
        :return: (float)
        """
        return scipy.special.comb(n, i) * t**i * (1 - t) ** (n - i)

    def bezier(self, t, control_points):
        """
        Return one point on the bezier curve.
        :param t: (float) number in [0, 1]
        :param control_points: (numpy array)
        :return: (numpy array) Coordinates of the point
        """
        n = len(control_points) - 1
        return np.sum(
            [self.bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)],
            axis=0,
        )

    def bezier_derivatives_control_points(self, control_points, n_derivatives):
        """
        Compute control points of the successive derivatives of a given bezier curve.
        A derivative of a bezier curve is a bezier curve.
        See https://pomax.github.io/bezierinfo/#derivatives
        for detailed explanations
        :param control_points: (numpy array)
        :param n_derivatives: (int)
        e.g., n_derivatives=2 -> compute control points for first and second derivatives
        :return: ([numpy array])
        """
        w = {0: control_points}
        for i in range(n_derivatives):
            n = len(w[i])
            w[i + 1] = np.array(
                [(n - 1) * (w[i][j + 1] - w[i][j]) for j in range(n - 1)]
            )
        return w

    def curvature(self, dx, dy, ddx, ddy):
        """
        Compute curvature at one point given first and second derivatives.
        :param dx: (float) First derivative along x axis
        :param dy: (float)
        :param ddx: (float) Second derivative along x axis
        :param ddy: (float)
        :return: (float)
        """
        return (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** (3 / 2)


def main():

    from confrez.control.compute_sets import convert_rl_states

    rl_states = [
        {"front": (6, 8), "back": (6, 7)},
        {"front": (6, 7), "back": (7, 6)},
        {"front": (7, 6), "back": (8, 6)},
    ]

    planner = BezierPlanner(offset=2.5)

    plt.figure()

    for i in range(len(rl_states) - 1):
        start_state = convert_rl_states(states=rl_states[i], vehicle_body=VehicleBody())
        end_state = convert_rl_states(
            states=rl_states[i + 1], vehicle_body=VehicleBody()
        )

        if rl_states[i + 1]["front"] == rl_states[i]["back"]:
            # Vehicle goes backward
            angle_offset = np.pi
        else:
            angle_offset = 0

        start_state.e.psi = pi_2_pi(start_state.e.psi + angle_offset)
        end_state.e.psi = pi_2_pi(end_state.e.psi + angle_offset)

        path = planner.interpolate(start_state=start_state, end_state=end_state, N=30)

        path[:, 2] = pi_2_pi(path[:, 2] - angle_offset)

        plt.subplot(1, 2, 1)
        plt.plot(path[:, 0], path[:, 1])
        plt.axis("equal")

        plt.subplot(1, 2, 2)
        plt.plot(path[:, 2])

    plt.show()


if __name__ == "__main__":
    main()
    #  main2()
