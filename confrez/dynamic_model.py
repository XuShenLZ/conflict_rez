import casadi as ca
from confrez.vehicle_types import VehicleBody
import pymunk


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


def kinematic_bicycle_rk(dt: float, vehicle_body: VehicleBody, M=4):
    """
    rk discrete time model for the kinematic bicycle dynamics
    """
    h = dt / M

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    v = ca.SX.sym("v")
    yaw = ca.SX.sym("yaw")
    delta = ca.SX.sym("delta")
    state = ca.vertcat(x, y, yaw, v, delta)

    a = ca.SX.sym("a")
    w = ca.SX.sym("w")  # Angular velocity of steering angle
    input = ca.vertcat(a, w)

    f_ct = kinematic_bicycle_ct(vehicle_body=vehicle_body)

    zkp = state
    for _ in range(M):
        a1 = f_ct(zkp, input)
        a2 = f_ct(zkp + h * a1 / 2, input)
        a3 = f_ct(zkp + h * a2 / 2, input)
        a4 = f_ct(zkp + h * a3, input)

        zkp = zkp + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)

    return ca.Function("f_dt", [state, input], [zkp])


def kinematic_bicycle_simulator(dt: float, vehicle_body: VehicleBody):
    """
    kinematic_bicycle simulator with ODE integrator
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

    prob = {"x": state, "p": input, "ode": output}
    setup = {"t0": 0, "tf": dt}

    ode_solver = ca.integrator("int", "idas", prob, setup)

    state_mx = ca.MX.sym("state", state.size())
    input_mx = ca.MX.sym("input", input.size())

    zf = ode_solver.call([state_mx, input_mx, 0, 0, 0, 0])[0]

    return ca.Function("zf", [state_mx, input_mx], [zf])


def unicycle_simulator(dt: float) -> ca.Function:
    """
    unicycle model simulator
    """
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    yaw = ca.SX.sym("yaw")
    state = ca.vertcat(x, y, yaw)

    v = ca.SX.sym("v")  # Velocity / speed
    w = ca.SX.sym("w")  # Rotation rate
    input = ca.vertcat(v, w)

    xdot = v * ca.cos(yaw)
    ydot = v * ca.sin(yaw)
    yawdot = w
    output = ca.vertcat(xdot, ydot, yawdot)

    prob = {"x": state, "p": input, "ode": output}
    setup = {"t0": 0, "tf": dt}

    ode_solver = ca.integrator("int", "idas", prob, setup)

    state_mx = ca.MX.sym("state", state.size())
    input_mx = ca.MX.sym("input", input.size())

    zf = ode_solver.call([state_mx, input_mx, 0, 0, 0, 0])[0]

    return ca.Function("zf", [state_mx, input_mx], [zf])
