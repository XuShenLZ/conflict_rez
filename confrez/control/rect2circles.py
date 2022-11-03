import numpy as np

import matplotlib.pyplot as plt
from confrez.control.utils import plot_car

from confrez.pytypes import VehicleState
from confrez.vehicle_types import VehicleBody
from confrez.obstacle_types import RectangleObstacle

import casadi as ca


def v2c_ca(
    x: ca.Opti.variable,
    y: ca.Opti.variable,
    psi: ca.Opti.variable,
    vehicle_body: VehicleBody,
):
    """
    Use a few circles to approximate vehicle body rectangle
    num_circles: the number of circles to approximate
    """
    radius = vehicle_body.w / 2

    start_offset = vehicle_body.cr
    start_xc = x - start_offset * ca.cos(psi)
    start_yc = y - start_offset * ca.sin(psi)

    end_offset = vehicle_body.cf
    end_xc = x + end_offset * ca.cos(psi)
    end_yc = y + end_offset * ca.sin(psi)

    xcs = ca.linspace(start_xc, end_xc, vehicle_body.num_circles)
    ycs = ca.linspace(start_yc, end_yc, vehicle_body.num_circles)
    cs = ca.horzcat(xcs, ycs)

    return xcs, ycs


def v2c(state: VehicleState, vehicle_body: VehicleBody):
    """
    Use a few circles to approximate vehicle body rectangle
    num_circles: the number of circles to approximate
    """
    radius = vehicle_body.w / 2

    start_offset = vehicle_body.cr
    start_xc = state.x.x - start_offset * np.cos(state.e.psi)
    start_yc = state.x.y - start_offset * np.sin(state.e.psi)

    end_offset = vehicle_body.cf
    end_xc = state.x.x + end_offset * np.cos(state.e.psi)
    end_yc = state.x.y + end_offset * np.sin(state.e.psi)

    xcs = np.linspace(start_xc, end_xc, vehicle_body.num_circles, endpoint=True)
    ycs = np.linspace(start_yc, end_yc, vehicle_body.num_circles, endpoint=True)

    circles = []
    for xc, yc in zip(xcs, ycs):
        circles.append((xc, yc, radius))

    return circles


def main():
    state = VehicleState()
    state.x.x = 1
    state.x.y = 2
    state.e.psi = np.pi / 4
    vehicle_body = VehicleBody()

    fig, ax = plt.subplots(1)

    plot_car(state.x.x, state.x.y, state.e.psi, vehicle_body)

    circles = v2c(state, vehicle_body)

    for circle in circles:
        cir = plt.Circle((circle[0], circle[1]), circle[2], color="b")
        ax.add_patch(cir)

    plt.xlim(-2, 5)
    plt.ylim(-1, 5)
    plt.show()


def main2():
    state = VehicleState()
    state.x.x = 1
    state.x.y = 2
    state.e.psi = np.pi / 4
    vehicle_body = VehicleBody()

    fig, ax = plt.subplots(1)

    plot_car(state.x.x, state.x.y, state.e.psi, vehicle_body)

    xcs, ycs = v2c_ca(state.x.x, state.x.y, state.e.psi, vehicle_body)

    for i in range(vehicle_body.num_circles):
        cir = plt.Circle(
            (xcs[i].__float__(), ycs[i].__float__()), vehicle_body.w / 2, color="b"
        )
        ax.add_patch(cir)

    plt.xlim(-2, 5)
    plt.ylim(-1, 5)
    plt.show()


if __name__ == "__main__":
    # main()
    main2()
