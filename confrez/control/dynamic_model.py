import casadi as ca
from confrez.vehicle_types import VehicleBody


def kinematic_bicycle_ct(vehicle_body: VehicleBody):
    """
    return the continuous time kinematic bicycle model
    """
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    v = ca.SX.sym("v")
    yaw = ca.SX.sym("yaw")
    delta = ca.SX.sym("delta")
    state = ca.vertcat(x, y, yaw, v, delta)

    a = ca.SX.sym("a")
    w = ca.SX.sym("w")  # Angular velocity of steering angle
    input = ca.vertcat(a, w)

    xdot = v * ca.cos(yaw)
    ydot = v * ca.sin(yaw)
    vdot = a
    yawdot = v / vehicle_body.wb * ca.tan(delta)
    deltadot = w
    output = ca.vertcat(xdot, ydot, yawdot, vdot, deltadot)

    return ca.Function("f_ct", [state, input], [output])
