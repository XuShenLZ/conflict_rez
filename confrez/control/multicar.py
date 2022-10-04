from itertools import product
from math import ceil
from re import L
from typing import Dict, List, Tuple
import casadi as ca
import numpy as np
from pytope import Polytope

from scipy.interpolate import interp1d
from confrez.control.rect2circles import v2c_ca
from confrez.control.utils import plot_car
from confrez.obstacle_types import GeofenceRegion

from confrez.pytypes import VehiclePrediction, VehicleState
from confrez.vehicle_types import VehicleBody, VehicleConfig
from confrez.control.dynamic_model import kinematic_bicycle_ct
from confrez.control.compute_sets import (
    compute_initial_states,
    compute_obstacles,
    compute_sets,
    interp_along_sets,
)
from confrez.control.planner import SingleVehiclePlanner

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


class Vehicle(object):
    """
    Vehicle class containing the states and transformation for final optimization
    """

    def __init__(
        self,
        opti: ca.Opti,
        dt: ca.Opti.variable,
        N: int,
        K: int,
        init_state: VehicleState,
        zu0: VehiclePrediction,
        obstacles: List[Polytope],
        rl_sets: List[Dict[str, Polytope]],
        init_offset: VehicleState,
        N_per_set: int = 5,
        dmin: float = 0.05,
        shrink_tube: float = 0.8,
        vehicle_config: VehicleConfig = VehicleConfig(),
        region: GeofenceRegion = GeofenceRegion(),
        vehicle_body: VehicleBody = VehicleBody(),
    ) -> None:
        self.opti = opti
        self.dt = dt
        self.vehicle_body = vehicle_body
        self.region = region
        self.vehicle_config = vehicle_config
        self.N = N
        self.K = K
        self.zu0 = zu0

        n_obs = len(obstacles)

        n_hps = []
        for obs in obstacles:
            n_hps.append(len(obs.b))

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        self.x = opti.variable(N, K + 1)
        self.y = opti.variable(N, K + 1)
        self.psi = opti.variable(N, K + 1)
        self.v = opti.variable(N, K + 1)
        self.delta = opti.variable(N, K + 1)

        self.a = opti.variable(N, K + 1)
        self.w = opti.variable(N, K + 1)

        self.l = [[opti.variable(sum(n_hps)) for _ in range(K + 1)] for _ in range(N)]
        self.m = [[opti.variable(4 * n_obs) for _ in range(K + 1)] for _ in range(N)]

        self.J = 0

        A, B, D = self.collocation_coefficients(K=K)

        f_ct = kinematic_bicycle_ct(vehicle_body=self.vehicle_body)

        opti.subject_to(self.x[0, 0] == init_state.x.x + init_offset.x.x)
        opti.subject_to(self.y[0, 0] == init_state.x.y + init_offset.x.y)
        opti.subject_to(self.psi[0, 0] == init_state.e.psi + init_offset.e.psi)

        opti.subject_to(self.v[0, 0] == 0)
        opti.subject_to(self.delta[0, 0] == 0)

        opti.subject_to(self.a[0, 0] == 0)
        opti.subject_to(self.w[0, 0] == 0)

        for i in range(N):
            for k in range(K + 1):
                # collocation point variables
                opti.subject_to(
                    opti.bounded(self.region.x_min, self.x[i, k], self.region.x_max)
                )
                opti.subject_to(
                    opti.bounded(self.region.y_min, self.y[i, k], self.region.y_max)
                )

                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.v_min,
                        self.v[i, k],
                        self.vehicle_config.v_max,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.delta_min,
                        self.delta[i, k],
                        self.vehicle_config.delta_max,
                    )
                )

                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.a_min,
                        self.a[i, k],
                        self.vehicle_config.a_max,
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.w_delta_min,
                        self.w[i, k],
                        self.vehicle_config.w_delta_max,
                    )
                )

                # Dual multipliers. Note here that the l and m are lists of opti.variable
                opti.subject_to(self.l[i][k] >= 0)
                opti.set_initial(self.l[i][k], zu0.l[i][k])

                opti.subject_to(self.m[i][k] >= 0)
                opti.set_initial(self.m[i][k], zu0.m[i][k])

                # Collocation constraints
                state = ca.vertcat(
                    self.x[i, k],
                    self.y[i, k],
                    self.psi[i, k],
                    self.v[i, k],
                    self.delta[i, k],
                )
                input = ca.vertcat(self.a[i, k], self.w[i, k])
                func_ode = f_ct(state, input)
                poly_ode = 0

                for j in range(K + 1):
                    zij = ca.vertcat(
                        self.x[i, j],
                        self.y[i, j],
                        self.psi[i, j],
                        self.v[i, j],
                        self.delta[i, j],
                    )
                    poly_ode += A[j, k] * zij / dt

                opti.subject_to(poly_ode == func_ode)

                # Cost
                error = (
                    0
                    # + (x[i, k] - x_interp[i * K + k]) ** 2
                    # + (y[i, k] - y_interp[i * K + k]) ** 2
                    # + (psi[i, k] - psi_interp[i * K + k]) ** 2
                    + self.a[i, k] ** 2
                    + (self.v[i, k] ** 2) * (self.w[i, k] ** 2)
                    + self.delta[i, k] ** 2
                )
                self.J += B[k] * error * dt

                # OBCA constraints
                t = ca.vertcat(self.x[i, k], self.y[i, k])
                R = ca.vertcat(
                    ca.horzcat(ca.cos(self.psi[i, k]), -ca.sin(self.psi[i, k])),
                    ca.horzcat(ca.sin(self.psi[i, k]), ca.cos(self.psi[i, k])),
                )
                for j, obs in enumerate(obstacles):
                    idx0 = sum(n_hps[:j])
                    idx1 = sum(n_hps[: j + 1])
                    lj = self.l[i][k][idx0:idx1]
                    mj = self.m[i][k][4 * j : 4 * (j + 1)]

                    opti.subject_to(
                        ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj) >= dmin
                    )
                    opti.subject_to(veh_G.T @ mj + R.T @ obs.A.T @ lj == np.zeros(2))
                    opti.subject_to(ca.dot(obs.A.T @ lj, obs.A.T @ lj) == 1)

            # Continuity constraints
            if i >= 1:
                poly_prev = 0
                input_prev = 0
                for j in range(K + 1):
                    zimj = ca.vertcat(
                        self.x[i - 1, j],
                        self.y[i - 1, j],
                        self.psi[i - 1, j],
                        self.v[i - 1, j],
                        self.delta[i - 1, j],
                    )
                    uimj = ca.vertcat(self.a[i - 1, j], self.w[i - 1, j])
                    poly_prev += D[j] * zimj
                    input_prev += D[j] * uimj

                zi0 = ca.vertcat(
                    self.x[i, 0],
                    self.y[i, 0],
                    self.psi[i, 0],
                    self.v[i, 0],
                    self.delta[i, 0],
                )
                ui0 = ca.vertcat(self.a[i, 0], self.w[i, 0])
                opti.subject_to(poly_prev == zi0)
                opti.subject_to(input_prev == ui0)

                # RL set constraints
                q, r = divmod(i, N_per_set)
                if r == 0 and i > 0:
                    back = ca.vertcat(self.x[i, 0], self.y[i, 0])
                    tA = rl_sets[q]["back"].A
                    tb = rl_sets[q]["back"].b
                    opti.subject_to(tA @ back <= tb - shrink_tube)

                    front = ca.vertcat(
                        self.x[i, 0] + self.vehicle_body.wb * ca.cos(self.psi[i, 0]),
                        self.y[i, 0] + self.vehicle_body.wb * ca.sin(self.psi[i, 0]),
                    )
                    tA = rl_sets[q]["front"].A
                    tb = rl_sets[q]["front"].b
                    opti.subject_to(tA @ front <= tb - shrink_tube)

        # Final constraints
        self.zF = 0
        self.uF = 0
        for j in range(K + 1):
            zimj = ca.vertcat(
                self.x[N - 1, j],
                self.y[N - 1, j],
                self.psi[N - 1, j],
                self.v[N - 1, j],
                self.delta[N - 1, j],
            )
            uimj = ca.vertcat(self.a[N - 1, j], self.w[N - 1, j])
            self.zF += D[j] * zimj
            self.uF += D[j] * uimj

        # opti.subject_to(self.zF[0] == x_interp[-1])
        # opti.subject_to(self.zF[1] == y_interp[-1])
        # opti.subject_to(self.zF[2] == psi_interp[-1])

        # RL set constraints
        back = ca.vertcat(self.zF[0], self.zF[1])
        tA = rl_sets[-1]["back"].A
        tb = rl_sets[-1]["back"].b
        opti.subject_to(tA @ back <= tb - shrink_tube)

        front = ca.vertcat(
            self.zF[0] + self.vehicle_body.wb * ca.cos(self.zF[2]),
            self.zF[1] + self.vehicle_body.wb * ca.sin(self.zF[2]),
        )
        tA = rl_sets[-1]["front"].A
        tb = rl_sets[-1]["front"].b
        opti.subject_to(tA @ front <= tb - shrink_tube)

        opti.subject_to(self.zF[3] == 0)
        opti.subject_to(self.zF[4] == 0)

        opti.subject_to(self.uF[0] == 0)
        opti.subject_to(self.uF[1] == 0)

        # Initial guess
        opti.set_initial(self.x, np.array(zu0.x[:-1]).reshape(N, K + 1))
        opti.set_initial(self.y, np.array(zu0.y[:-1]).reshape(N, K + 1))
        opti.set_initial(self.psi, np.array(zu0.psi[:-1]).reshape(N, K + 1))
        opti.set_initial(self.v, np.array(zu0.v[:-1]).reshape(N, K + 1))
        opti.set_initial(self.delta, np.array(zu0.u_steer[:-1]).reshape(N, K + 1))

        opti.set_initial(self.a, np.array(zu0.u_a[:-1]).reshape(N, K + 1))
        opti.set_initial(self.w, np.array(zu0.u_steer_dot[:-1]).reshape(N, K + 1))

        self.J += (N * dt) ** 2

    def cost(self):
        return self.J

    def collocation_coefficients(
        self, K: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        computing the coefficients of the collocation polynominal
        `K`: (int) degree of the polynominal
        Return:
        `A`: Coefficients of the collocation equation
        `B`: Coefficients of the quadrature function for cost integral
        `D`: Coefficients of the continuity equation
        """
        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(K, "radau"))

        # Coefficients of the collocation equation
        A = np.zeros((K + 1, K + 1))

        # Coefficients of the continuity equation
        D = np.zeros(K + 1)

        # Coefficients of the quadrature function
        B = np.zeros(K + 1)

        # Construct polynomial basis
        for j in range(K + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for k in range(K + 1):
                if k != j:
                    p *= np.poly1d([1, -tau_root[k]]) / (tau_root[j] - tau_root[k])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for k in range(K + 1):
                A[j, k] = pder(tau_root[k])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return A, B, D

    def get_sol(self, sol, dt):
        """
        get the solution from the optimization problem
        """
        x_opt = np.array(sol.value(self.x)).flatten()
        y_opt = np.array(sol.value(self.y)).flatten()
        psi_opt = np.array(sol.value(self.psi)).flatten()
        v_opt = np.array(sol.value(self.v)).flatten()
        delta_opt = np.array(sol.value(self.delta)).flatten()

        a_opt = np.array(sol.value(self.a)).flatten()
        w_opt = np.array(sol.value(self.w)).flatten()

        result = VehiclePrediction()

        result.dt = sol.value(dt)
        result.t = self.zu0.t / self.zu0.dt * sol.value(dt)
        result.x = np.append(x_opt, sol.value(self.zF[0]))
        result.y = np.append(y_opt, sol.value(self.zF[1]))
        result.psi = np.append(psi_opt, sol.value(self.zF[2]))

        result.v = np.append(v_opt, sol.value(self.zF[3]))
        result.u_steer = np.append(delta_opt, sol.value(self.zF[4]))

        result.u_a = np.append(a_opt, sol.value(self.uF[0]))
        result.u_steer_dot = np.append(w_opt, sol.value(self.uF[1]))

        result.l = [[None for _ in range(self.K + 1)] for _ in range(self.N)]
        result.m = [[None for _ in range(self.K + 1)] for _ in range(self.N)]
        for i in range(self.N):
            for k in range(self.K + 1):
                result.l[i][k] = np.array(sol.value(self.l[i][k]))
                result.m[i][k] = np.array(sol.value(self.m[i][k]))

        self.get_interpolator(K=self.K, N=self.N, dt=sol.value(dt), opt=result)

        return result

    def get_interpolator(self, K: int, N: int, dt: float, opt: VehiclePrediction):
        """
        Get the interpolation functions with collocation polynomial
        `K`: (int) The degree of collocation polinomial
        `N`: (int) The number of collocation intervals
        `dt`: (float) The optimal interval length
        `opt`: (VehiclePrediction) The optimal state-input solution
        """
        x_opt = np.reshape(opt.x[:-1], (N, K + 1))
        y_opt = np.reshape(opt.y[:-1], (N, K + 1))
        psi_opt = np.reshape(opt.psi[:-1], (N, K + 1))
        v_opt = np.reshape(opt.v[:-1], (N, K + 1))
        delta_opt = np.reshape(opt.u_steer[:-1], (N, K + 1))

        X = np.stack([x_opt, y_opt, psi_opt, v_opt, delta_opt], axis=-1)

        tau_root = np.append(0, ca.collocation_points(K, "radau"))
        _, _, D = self.collocation_coefficients(K=K)

        t = ca.SX.sym("t")
        tgrid = np.linspace(
            start=0, stop=N * dt, num=N + 1, endpoint=True
        )  # The time steps at the interval transitions

        t_i = ca.pw_const(t, tgrid[1:], tgrid)  # the starting time of each interval

        n = X.shape[2]  # dim of the state space

        collocation_states = np.resize(
            np.array([], dtype=ca.SX), (K + 1, n)
        )  # Interpolator of collocation states

        for l in range(n):

            lf = 0
            for k in range(K + 1):
                lf += X[-1, k, l] * D[k]

            for k in range(K + 1):
                xp = ca.horzcat(*[X[i, k, l] for i in range(N)], lf)
                collocation_states[k, l] = ca.pw_const(t, tgrid[1:], xp.T)

        rel_t = (t - t_i) / dt
        interpolated_state = [0] * n

        for l in range(n):
            xl = 0
            for j in range(K + 1):
                Lj = 1
                for k in range(K + 1):
                    if k != j:
                        Lj *= (rel_t - tau_root[k]) / (tau_root[j] - tau_root[k])

                xl += Lj * collocation_states[j, l]

            interpolated_state[l] = xl

        self.state_interpolator = ca.Function(
            "state_interpolator", [t], [ca.vertcat(*interpolated_state)]
        )

        pw_a = ca.pw_const(t, opt.t[1:], opt.u_a)
        pw_w = ca.pw_const(t, opt.t[1:], opt.u_steer_dot)

        self.input_interpolator = ca.Function("input_interpolator", [t], [pw_a, pw_w])

    def interpolate_states(self, time: np.ndarray) -> VehiclePrediction:
        """
        interpolate states using specified time sequence
        `time`: 1-D array like object containing the new time sequence
        `state_interpolator`: State interpolator as a casadi function
        `input_interpolator`: Input interpolator as a casadi function
        """
        x = []
        y = []
        psi = []
        v = []

        a = []
        delta = []
        w = []
        for t in time:
            interp_state = self.state_interpolator(t)
            interp_input = self.input_interpolator(t)

            x.append(interp_state[0].__float__())
            y.append(interp_state[1].__float__())
            psi.append(interp_state[2].__float__())
            v.append(interp_state[3].__float__())
            delta.append(interp_state[4].__float__())

            a.append(interp_input[0].__float__())
            w.append(interp_input[1].__float__())

        result = VehiclePrediction()

        result.t = np.array(time)

        result.x = np.array(x)
        result.y = np.array(y)
        result.psi = np.array(psi)
        result.v = np.array(v)

        result.u_a = np.array(a)
        result.u_steer = np.array(delta)
        result.u_steer_dot = np.array(w)

        return result


class MultiVehiclePlanner(object):
    """
    Conflict resolution of multiple vehicle
    """

    def __init__(
        self,
        rl_file_name: str,
        ws_config: Dict[str, bool],
        init_offsets: Dict[str, VehicleState],
        vehicle_body: VehicleBody = VehicleBody(),
        vehicle_config: VehicleConfig = VehicleConfig(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        self.rl_file_name = rl_file_name
        self.vehicle_body = vehicle_body
        self.init_offsets = init_offsets
        self.vehicle_config = vehicle_config
        self.region = region
        self.ws_config = ws_config

        self.rl_sets = compute_sets(rl_file_name)
        self.init_states = compute_initial_states(rl_file_name, vehicle_body)
        self.obstacles = compute_obstacles()

        self.agents = set(self.rl_sets.keys())

        self.single_planners = {
            agent: SingleVehiclePlanner(
                rl_file_name=rl_file_name,
                agent=agent,
                init_offset=self.init_offsets[agent],
                vehicle_body=self.vehicle_body,
                vehicle_config=self.vehicle_config,
                region=self.region,
                verbose=0,
            )
            for agent in self.agents
        }

        self.num_sets = {agent: len(self.rl_sets[agent]) for agent in self.agents}

    def solve_single_problems(self):
        self.single_result = {agent: VehiclePrediction() for agent in self.agents}

        for agent in self.agents:
            print("====Solving single vehicle problem=======")
            print(agent)
            zu0 = self.single_planners[agent].solve_ws(
                N=30, shrink_tube=0.5, spline_ws=self.ws_config[agent]
            )
            l0, m0 = self.single_planners[agent].dual_ws(
                zu0=zu0, obstacles=self.single_planners[agent].obstacles
            )

            self.single_result[agent] = self.single_planners[agent].solve(
                zu0=zu0,
                l0=l0,
                m0=m0,
                obstacles=self.single_planners[agent].obstacles,
                K=5,
                N_per_set=5,
                shrink_tube=0.5,
            )

    def solve_final_problem(
        self,
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.8,
        dmin: float = 0.05,
    ):
        """
        solve the final problem for multiple vehicles
        """
        print("Solving final opt...")

        N = {agent: N_per_set * (self.num_sets[agent] - 1) for agent in self.agents}

        # Collocation time steps
        tau_root = np.append(0, ca.collocation_points(K, "radau"))

        # Initial guess for time step
        dt0 = np.mean([self.single_result[agent].dt for agent in self.agents])

        # A, B, D = self.collocation_coefficients(K=K)

        # f_ct = kinematic_bicycle_ct(vehicle_body=self.vehicle_body)

        opti = ca.Opti()
        dt = opti.variable()
        opti.set_initial(dt, dt0)

        vehicles = {
            agent: Vehicle(
                opti=opti,
                dt=dt,
                N=N[agent],
                K=K,
                dmin=dmin,
                init_offset=self.init_offsets[agent],
                shrink_tube=shrink_tube,
                init_state=self.init_states[agent],
                zu0=self.single_result[agent],
                obstacles=self.obstacles,
                rl_sets=self.rl_sets[agent],
            )
            for agent in self.agents
        }

        for agent in self.agents:
            others = self.agents - {agent}
            for other in others:
                N_min = min(N[agent], N[other])
                for i, k in product(range(N_min), range(K)):
                    self_xcs, self_ycs = v2c_ca(
                        vehicles[agent].x[i, k],
                        vehicles[agent].y[i, k],
                        vehicles[agent].psi[i, k],
                        self.vehicle_body,
                    )
                    other_xcs, other_ycs = v2c_ca(
                        vehicles[other].x[i, k],
                        vehicles[other].y[i, k],
                        vehicles[other].psi[i, k],
                        self.vehicle_body,
                    )
                    for j1, j2 in product(
                        range(self.vehicle_body.num_circles), repeat=2
                    ):
                        vec = ca.vertcat(
                            self_xcs[j1] - other_xcs[j2], self_ycs[j1] - other_ycs[j2]
                        )

                        # opti.subject_to(ca.norm_2(vec) >= self.vehicle_body.w / 2)
                        opti.subject_to(
                            ca.bilin(ca.MX.eye(2), vec, vec)
                            >= (self.vehicle_body.w + 0.2) ** 2
                        )

        J = 0
        for agent in self.agents:
            J += vehicles[agent].J

        opti.minimize(J)

        p_opts = {"expand": True}
        s_opts = {
            "print_level": 0,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            # "max_iter": 300,
            # "mumps_mem_percent": 64000,
            "linear_solver": "ma97",
        }
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()
        print(sol.stats()["return_status"])

        for agent in self.agents:
            vehicles[agent].get_sol(sol=sol, dt=dt)

        N_max = np.max([N[agent] for agent in self.agents])

        final_t = np.linspace(
            0, N_max * sol.value(dt), N_max * (K + 1) + 1, endpoint=True
        )

        final_sol = {
            agent: vehicles[agent].interpolate_states(final_t) for agent in self.agents
        }

        plt.figure()
        for agent in self.agents:
            plt.plot(
                np.array(sol.value(vehicles[agent].x)).flatten(),
                label=agent + "_final_sol",
            )
            plt.plot(
                sol.value(self.single_result[agent].x), label=agent + "_individual_sol"
            )

        plt.legend()

        plt.figure()
        for agent in self.agents:
            plt.plot(
                np.array(sol.value(vehicles[agent].x)).flatten(),
                np.array(sol.value(vehicles[agent].y)).flatten(),
                label=agent,
            )
        plt.axis("equal")

        fig = plt.figure()
        ax = plt.gca()

        def plot_frame(i):
            ax.clear()
            for obstacle in self.obstacles:
                obstacle.plot(ax, facecolor="b", alpha=0.5)
            for agent in self.agents:
                ax.plot(
                    np.array(sol.value(vehicles[agent].x)).flatten(),
                    np.array(sol.value(vehicles[agent].y)).flatten(),
                    label=agent,
                )
                plot_car(
                    final_sol[agent].x[i],
                    final_sol[agent].y[i],
                    final_sol[agent].psi[i],
                    vehicle_body=self.vehicle_body,
                )
            ax.set_aspect("equal")

        ani = FuncAnimation(
            fig, plot_frame, frames=len(final_t), interval=40, repeat=True
        )

        writer = FFMpegWriter(fps=int(1000 / 40))
        ani.save(self.rl_file_name + ".mp4", writer=writer)

        plt.show()


def main():
    """
    main
    """
    init_offsets = {
        "vehicle_0": VehicleState(),
        "vehicle_1": VehicleState(),
        "vehicle_2": VehicleState(),
        "vehicle_3": VehicleState(),
    }
    # init_offsets["vehicle_0"].x.x = 0.1
    # init_offsets["vehicle_0"].x.y = 0.1
    # init_offsets["vehicle_0"].e.psi = np.pi / 20
    planner = MultiVehiclePlanner(
        rl_file_name="4v_rl_traj",
        ws_config={"vehicle_0": False, "vehicle_1": True, "vehicle_2": True, "vehicle_3": True},
        init_offsets=init_offsets,
    )
    planner.solve_single_problems()
    planner.solve_final_problem(shrink_tube=0.3)


if __name__ == "__main__":
    main()
