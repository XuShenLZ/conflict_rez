from math import ceil
from typing import Dict, Tuple
import numpy as np
import casadi as ca
import dill

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from confrez.control.compute_sets import (
    compute_initial_states,
    compute_obstacles,
    compute_sets,
    interp_along_sets,
)
from confrez.control.dynamic_model import kinematic_bicycle_ct
from confrez.control.utils import plot_car
from confrez.obstacle_types import GeofenceRegion

from confrez.pytypes import VehiclePrediction, VehicleState
from confrez.vehicle_types import VehicleBody, VehicleConfig


class Vehicle(object):
    """
    Single vehicle class containing the optimization variables and constraints formulation
    """

    def __init__(
        self,
        rl_file_name: str,
        agent: str,
        color: Dict[str, Tuple[float, float, float]],
        vehicle_config: VehicleConfig = VehicleConfig(),
        vehicle_body: VehicleBody = VehicleBody(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        self.rl_file_name = rl_file_name
        self.agent = agent
        self.color = color

        self.vehicle_config = vehicle_config
        self.vehicle_body = vehicle_body
        self.region = region

        self.init_state = compute_initial_states(self.rl_file_name, self.vehicle_body)[
            self.agent
        ]
        self.obstacles = compute_obstacles()
        self.rl_tube = compute_sets(self.rl_file_name)[self.agent]

        self.num_sets = len(self.rl_tube)

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

    def state_ws(
        self,
        N: int = 30,
        dt: float = 0.1,
        init_offset: VehicleState = VehicleState(),
        bounded_input: bool = False,
        shrink_tube: float = 0.8,
        spline_ws: bool = False,
        verbose: int = 0,
    ) -> VehiclePrediction:
        """
        setup the warm start optimization problem
        """
        print("Solving state ws...")
        f_ct = kinematic_bicycle_ct(vehicle_body=self.vehicle_body)

        M = self.num_sets - 1

        opti = ca.Opti()

        x = opti.variable(N * M + 1)
        y = opti.variable(N * M + 1)
        psi = opti.variable(N * M + 1)
        v = opti.variable(N * M + 1)
        delta = opti.variable(N * M + 1)

        a = opti.variable(N * M)
        w = opti.variable(N * M)

        J = 0

        opti.subject_to(x[0] == self.init_state.x.x + init_offset.x.x)
        opti.subject_to(y[0] == self.init_state.x.y + init_offset.x.y)
        opti.subject_to(psi[0] == self.init_state.e.psi + init_offset.e.psi)
        opti.subject_to(v[0] == 0)
        opti.subject_to(delta[0] == 0)

        opti.subject_to(a[0] == 0)
        opti.subject_to(w[0] == 0)

        for k in range(N * M):
            opti.subject_to(opti.bounded(self.region.x_min, x[k], self.region.x_max))
            opti.subject_to(opti.bounded(self.region.y_min, y[k], self.region.y_max))

            opti.subject_to(
                opti.bounded(self.vehicle_config.v_min, v[k], self.vehicle_config.v_max)
            )
            opti.subject_to(
                opti.bounded(
                    self.vehicle_config.delta_min,
                    delta[k],
                    self.vehicle_config.delta_max,
                )
            )

            if bounded_input:
                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.a_min, a[k], self.vehicle_config.a_max
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.w_delta_min,
                        w[k],
                        self.vehicle_config.w_delta_max,
                    )
                )

            state = ca.vertcat(x[k], y[k], psi[k], v[k], delta[k])
            input = ca.vertcat(a[k], w[k])

            state_p = ca.vertcat(x[k + 1], y[k + 1], psi[k + 1], v[k + 1], delta[k + 1])
            opti.subject_to(state_p == state + dt * f_ct(state, input))

            error = a[k] ** 2 + w[k] ** 2
            J += error

        for i in range(1, self.num_sets):
            k = N * i

            back = ca.vertcat(x[k], y[k])
            A = self.rl_tube[i]["back"].A
            b = self.rl_tube[i]["back"].b
            opti.subject_to(A @ back <= b - shrink_tube)

            front = ca.vertcat(
                x[k] + self.vehicle_body.wb * ca.cos(psi[k]),
                y[k] + self.vehicle_body.wb * ca.sin(psi[k]),
            )
            A = self.rl_tube[i]["front"].A
            b = self.rl_tube[i]["front"].b
            opti.subject_to(A @ front <= b - shrink_tube)

        opti.minimize(J)

        if spline_ws:
            interp_waypoints = interp_along_sets(
                file_name=self.rl_file_name, vehicle_body=self.vehicle_body, N=N
            )[self.agent]
            opti.set_initial(x, interp_waypoints[:, 0])
            opti.set_initial(y, interp_waypoints[:, 1])
            opti.set_initial(psi, interp_waypoints[:, 2])

        p_opts = {"expand": True}
        s_opts = {
            "print_level": verbose,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            "max_iter": 500,
            # "mumps_mem_percent": 64000,
        }
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()
        print(sol.stats()["return_status"])

        result = VehiclePrediction()

        result.t = np.linspace(0, (N * M) * dt, N * M + 1, endpoint=True)
        result.x = sol.value(x)
        result.y = sol.value(y)
        result.psi = sol.value(psi)
        result.v = sol.value(v)

        result.u_a = np.append(sol.value(a), sol.value(a)[-1])
        result.u_steer = sol.value(delta)
        result.u_steer_dot = np.append(sol.value(w), sol.value(w)[-1])

        return result

    def dual_ws(self, zu0: VehiclePrediction, verbose: int = 0) -> VehiclePrediction:
        """
        Warm start the dual variables
        `zu0`: VehiclePrediction object including the state-input warm start
        """
        print("Solving Dual WS Problem...")
        N = len(zu0.x)

        n_obs = len(self.obstacles)

        n_hps = []
        for obs in self.obstacles:
            n_hps.append(len(obs.b))

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        opti = ca.Opti()

        l = opti.variable(sum(n_hps), N)
        m = opti.variable(4 * n_obs, N)
        d = opti.variable(n_obs, N)

        obj = 0

        opti.subject_to(ca.vec(l) >= 0)
        opti.subject_to(ca.vec(m) >= 0)

        for k in range(N):
            t = ca.vertcat(zu0.x[k], zu0.y[k])
            R = np.array(
                [
                    [np.cos(zu0.psi[k]), -np.sin(zu0.psi[k])],
                    [np.sin(zu0.psi[k]), np.cos(zu0.psi[k])],
                ]
            )

            for j, obs in enumerate(self.obstacles):
                idx0 = sum(n_hps[:j])
                idx1 = sum(n_hps[: j + 1])
                lj = l[idx0:idx1, k]
                mj = m[4 * j : 4 * (j + 1), k]

                opti.subject_to(
                    ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj) == d[j, k]
                )
                opti.subject_to(veh_G.T @ mj + R.T @ obs.A.T @ lj == np.zeros(2))
                opti.subject_to(ca.dot(obs.A.T @ lj, obs.A.T @ lj) <= 1)

                obj -= d[j, k]

        opti.minimize(obj)

        p_opts = {"expand": True}
        s_opts = {"print_level": verbose}
        opti.solver("ipopt", p_opts, s_opts)

        sol = opti.solve()
        print(sol.stats()["return_status"])

        zu0.l = sol.value(l)
        zu0.m = sol.value(m)

        return zu0

    def interp_ws_for_collocation(
        self,
        zu0: VehiclePrediction,
        K: int = 5,
        N_per_set: int = 5,
    ):
        """
        interpolate warm-starting solution with normal timestamps into collocation format
        `zu0`: vehicle initial guess with states, inputs, and dual multipliers
        """
        # Interpolate for collocation
        fx0 = interp1d(zu0.t, zu0.x)
        fy0 = interp1d(zu0.t, zu0.y)
        fpsi0 = interp1d(zu0.t, zu0.psi)
        fv0 = interp1d(zu0.t, zu0.v)
        fdelta0 = interp1d(zu0.t, zu0.u_steer)

        fa0 = interp1d(zu0.t, zu0.u_a)
        fw0 = interp1d(zu0.t, zu0.u_steer_dot)

        fl0 = interp1d(zu0.t, zu0.l, axis=1)
        fm0 = interp1d(zu0.t, zu0.m, axis=1)

        # Collocation intervals
        N = N_per_set * (self.num_sets - 1)

        # Collocation time steps
        tau_root = np.append(0, ca.collocation_points(K, "radau"))

        t_interp = np.array([])
        for i in range(N):
            t_interp = np.append(t_interp, i + tau_root)
        t_interp = np.append(t_interp, N)
        t_interp = t_interp / N * zu0.t[-1]

        output = VehiclePrediction()

        # Interpolated states, input, and dual variables at collocation points
        output.t = t_interp
        output.x = fx0(t_interp)
        output.y = fy0(t_interp)
        output.psi = fpsi0(t_interp)
        output.v = fv0(t_interp)
        output.u_steer = fdelta0(t_interp)

        output.u_a = fa0(t_interp)
        output.u_steer_dot = fw0(t_interp)

        l_interp = fl0(t_interp)
        m_interp = fm0(t_interp)

        output.l = [[None for _ in range(K + 1)] for _ in range(N)]
        output.m = [[None for _ in range(K + 1)] for _ in range(N)]
        j = 0
        for i in range(N):
            for k in range(K + 1):
                output.l[i][k] = l_interp[:, j]
                output.m[i][k] = m_interp[:, j]
                j += 1

        return output

    def setup_single_final_problem(
        self,
        zu0: VehiclePrediction,
        init_offset: VehicleState = VehicleState(),
        opti: ca.Opti = None,
        dt: ca.Opti.variable = None,
        K: int = 5,
        N_per_set: int = 5,
        dmin: float = 0.05,
        shrink_tube: float = 0.8,
    ):
        """
        setup the final problem of a single vehicle
        `zu0`: initial guess for states, inputs, and dual multipliers at collocation points. If the initial guess was not solved with collocation, it is necessary to call `self.interp_ws_for_collocation()` first
        """
        if opti is None:
            self.opti = ca.Opti()
        else:
            self.opti = opti

        N = N_per_set * (self.num_sets - 1)
        self.N = N
        self.K = K

        if dt is None:
            self.dt = self.opti.variable()
            # Initial guess for time step
            dt0 = zu0.t[-1] / N
            self.opti.set_initial(self.dt, dt0)
        else:
            self.dt = dt

        n_obs = len(self.obstacles)

        n_hps = []
        for obs in self.obstacles:
            n_hps.append(len(obs.b))

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        self.x = self.opti.variable(N, K + 1)
        self.y = self.opti.variable(N, K + 1)
        self.psi = self.opti.variable(N, K + 1)
        self.v = self.opti.variable(N, K + 1)
        self.delta = self.opti.variable(N, K + 1)

        self.a = self.opti.variable(N, K + 1)
        self.w = self.opti.variable(N, K + 1)

        self.l = [
            [self.opti.variable(sum(n_hps)) for _ in range(K + 1)] for _ in range(N)
        ]
        self.m = [
            [self.opti.variable(4 * n_obs) for _ in range(K + 1)] for _ in range(N)
        ]

        self.J = 0

        A, B, D = self.collocation_coefficients(K=K)

        f_ct = kinematic_bicycle_ct(vehicle_body=self.vehicle_body)

        self.opti.subject_to(self.x[0, 0] == self.init_state.x.x + init_offset.x.x)
        self.opti.subject_to(self.y[0, 0] == self.init_state.x.y + init_offset.x.y)
        self.opti.subject_to(
            self.psi[0, 0] == self.init_state.e.psi + init_offset.e.psi
        )

        self.opti.subject_to(self.v[0, 0] == 0)
        self.opti.subject_to(self.delta[0, 0] == 0)

        self.opti.subject_to(self.a[0, 0] == 0)
        self.opti.subject_to(self.w[0, 0] == 0)

        for i in range(N):
            for k in range(K + 1):
                # collocation point variables
                self.opti.subject_to(
                    self.opti.bounded(
                        self.region.x_min, self.x[i, k], self.region.x_max
                    )
                )
                self.opti.subject_to(
                    self.opti.bounded(
                        self.region.y_min, self.y[i, k], self.region.y_max
                    )
                )

                self.opti.subject_to(
                    self.opti.bounded(
                        self.vehicle_config.v_min,
                        self.v[i, k],
                        self.vehicle_config.v_max,
                    )
                )
                self.opti.subject_to(
                    self.opti.bounded(
                        self.vehicle_config.delta_min,
                        self.delta[i, k],
                        self.vehicle_config.delta_max,
                    )
                )

                self.opti.subject_to(
                    self.opti.bounded(
                        self.vehicle_config.a_min,
                        self.a[i, k],
                        self.vehicle_config.a_max,
                    )
                )
                self.opti.subject_to(
                    self.opti.bounded(
                        self.vehicle_config.w_delta_min,
                        self.w[i, k],
                        self.vehicle_config.w_delta_max,
                    )
                )

                # Dual multipliers. Note here that the l and m are lists of self.opti.variable
                self.opti.subject_to(self.l[i][k] >= 0)
                self.opti.set_initial(self.l[i][k], zu0.l[i][k])

                self.opti.subject_to(self.m[i][k] >= 0)
                self.opti.set_initial(self.m[i][k], zu0.m[i][k])

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
                    poly_ode += A[j, k] * zij / self.dt

                self.opti.subject_to(poly_ode == func_ode)

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
                self.J += B[k] * error * self.dt

                # OBCA constraints
                t = ca.vertcat(self.x[i, k], self.y[i, k])
                R = ca.vertcat(
                    ca.horzcat(ca.cos(self.psi[i, k]), -ca.sin(self.psi[i, k])),
                    ca.horzcat(ca.sin(self.psi[i, k]), ca.cos(self.psi[i, k])),
                )
                for j, obs in enumerate(self.obstacles):
                    idx0 = sum(n_hps[:j])
                    idx1 = sum(n_hps[: j + 1])
                    lj = self.l[i][k][idx0:idx1]
                    mj = self.m[i][k][4 * j : 4 * (j + 1)]

                    self.opti.subject_to(
                        ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj) >= dmin
                    )
                    self.opti.subject_to(
                        veh_G.T @ mj + R.T @ obs.A.T @ lj == np.zeros(2)
                    )
                    self.opti.subject_to(ca.dot(obs.A.T @ lj, obs.A.T @ lj) == 1)

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
                self.opti.subject_to(poly_prev == zi0)
                self.opti.subject_to(input_prev == ui0)

                # RL set constraints
                q, r = divmod(i, N_per_set)
                if r == 0 and i > 0:
                    back = ca.vertcat(self.x[i, 0], self.y[i, 0])
                    tA = self.rl_tube[q]["back"].A
                    tb = self.rl_tube[q]["back"].b
                    self.opti.subject_to(tA @ back <= tb - shrink_tube)

                    front = ca.vertcat(
                        self.x[i, 0] + self.vehicle_body.wb * ca.cos(self.psi[i, 0]),
                        self.y[i, 0] + self.vehicle_body.wb * ca.sin(self.psi[i, 0]),
                    )
                    tA = self.rl_tube[q]["front"].A
                    tb = self.rl_tube[q]["front"].b
                    self.opti.subject_to(tA @ front <= tb - shrink_tube)

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

        # self.opti.subject_to(self.zF[0] == x_interp[-1])
        # self.opti.subject_to(self.zF[1] == y_interp[-1])
        # self.opti.subject_to(self.zF[2] == psi_interp[-1])

        # RL set constraints
        back = ca.vertcat(self.zF[0], self.zF[1])
        tA = self.rl_tube[-1]["back"].A
        tb = self.rl_tube[-1]["back"].b
        self.opti.subject_to(tA @ back <= tb - shrink_tube)

        front = ca.vertcat(
            self.zF[0] + self.vehicle_body.wb * ca.cos(self.zF[2]),
            self.zF[1] + self.vehicle_body.wb * ca.sin(self.zF[2]),
        )
        tA = self.rl_tube[-1]["front"].A
        tb = self.rl_tube[-1]["front"].b
        self.opti.subject_to(tA @ front <= tb - shrink_tube)

        self.opti.subject_to(self.zF[3] == 0)
        self.opti.subject_to(self.zF[4] == 0)

        self.opti.subject_to(self.uF[0] == 0)
        self.opti.subject_to(self.uF[1] == 0)

        # Initial guess
        self.opti.set_initial(self.x, zu0.x[:-1].reshape(N, K + 1))
        self.opti.set_initial(self.y, zu0.y[:-1].reshape(N, K + 1))
        self.opti.set_initial(self.psi, zu0.psi[:-1].reshape(N, K + 1))
        self.opti.set_initial(self.v, zu0.v[:-1].reshape(N, K + 1))
        self.opti.set_initial(self.delta, zu0.u_steer[:-1].reshape(N, K + 1))

        self.opti.set_initial(self.a, zu0.u_a[:-1].reshape(N, K + 1))
        self.opti.set_initial(self.w, zu0.u_steer_dot[:-1].reshape(N, K + 1))

        self.J += (N * self.dt) ** 2

        return self.opti

    def solve_single_final_problem(self, verbose: int = 0):
        """
        solve the trajectory of single vehicle problem
        """
        print("Solving single vehicle final trajectory...")
        self.opti.minimize(self.J)
        p_opts = {"expand": True}
        s_opts = {
            "print_level": verbose,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            # "max_iter": 300,
            # "mumps_mem_percent": 64000,
            "linear_solver": "ma97",
        }
        self.opti.solver("ipopt", p_opts, s_opts)
        sol = self.opti.solve()
        print(sol.stats()["return_status"])

        return sol

    def get_solution(self, sol: ca.OptiSol) -> VehiclePrediction:
        """
        get solution of this vehicle
        """

        x_opt = np.array(sol.value(self.x)).flatten()
        y_opt = np.array(sol.value(self.y)).flatten()
        psi_opt = np.array(sol.value(self.psi)).flatten()
        v_opt = np.array(sol.value(self.v)).flatten()
        delta_opt = np.array(sol.value(self.delta)).flatten()

        a_opt = np.array(sol.value(self.a)).flatten()
        w_opt = np.array(sol.value(self.w)).flatten()

        result = VehiclePrediction()

        result.dt = sol.value(self.dt)

        # Collocation time steps
        tau_root = np.append(0, ca.collocation_points(self.K, "radau"))
        t_collo = np.array([])
        for i in range(self.N):
            t_collo = np.append(t_collo, i + tau_root)
        t_collo = np.append(t_collo, self.N)
        result.t = t_collo * sol.value(self.dt)

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

        self.get_interpolator(K=self.K, N=self.N, dt=sol.value(self.dt), opt=result)

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

    def plot_result(self, result: VehiclePrediction, key_stride: int = 6):
        """
        plot the sets and result
        """
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 2, 1)
        for obstacle in self.obstacles:
            obstacle.plot(ax, facecolor="b", alpha=0.5)
        for body_sets in self.rl_tube:
            body_sets["front"].plot(ax, facecolor=self.color["front"], alpha=0.5)
            body_sets["back"].plot(ax, facecolor=self.color["back"], alpha=0.5)

        for i in range(self.num_sets):
            k = key_stride * i

            plot_car(result.x[k], result.y[k], result.psi[k], self.vehicle_body)

        ax.plot(result.x, result.y)
        ax.set_aspect("equal")
        ax = plt.subplot(2, 4, 3)
        ax.plot(result.t, result.v, label="v")
        ax.legend()
        ax = plt.subplot(2, 4, 4)
        ax.plot(result.t, result.u_a, label="u_a")
        ax.legend()
        ax = plt.subplot(2, 4, 7)
        ax.plot(result.t, result.u_steer, label="u_steer")
        ax.legend()
        ax = plt.subplot(2, 4, 8)
        ax.plot(result.t, result.u_steer_dot, label="u_steer_dot")
        ax.legend()

        ncol = 4
        nrow = ceil(self.num_sets / ncol)
        plt.figure(figsize=(2.5 * ncol, 2.5 * nrow))
        for i, body_sets in enumerate(self.rl_tube):
            ax = plt.subplot(nrow, ncol, i + 1)
            body_sets["front"].plot(ax, facecolor=self.color["front"], alpha=0.5)
            body_sets["back"].plot(ax, facecolor=self.color["back"], alpha=0.5)

            k = key_stride * i
            plot_car(result.x[k], result.y[k], result.psi[k], self.vehicle_body)

            # ax.set_xlim(xmin=-2.5, xmax=15 * 2.5)
            # ax.set_ylim(ymin=-2.5, ymax=15 * 2.5)
            ax.set_aspect("equal")

        plt.show()


def main():
    """
    main
    """
    rl_file_name = "4v_rl_traj"
    agent = "vehicle_0"
    vehicle = Vehicle(
        rl_file_name=rl_file_name,
        agent=agent,
        color={"front": (1, 0, 0), "back": (0, 1, 0)},
    )
    init_offset = VehicleState()

    spline_ws_config = {
        "vehicle_0": False,
        "vehicle_1": True,
        "vehicle_2": True,
        "vehicle_3": True,
    }

    zu0 = vehicle.state_ws(
        N=30,
        dt=0.1,
        init_offset=init_offset,
        shrink_tube=0.5,
        spline_ws=spline_ws_config[agent],
    )

    zu0 = vehicle.dual_ws(zu0)
    zu0 = vehicle.interp_ws_for_collocation(zu0, K=5, N_per_set=5)
    vehicle.setup_single_final_problem(
        zu0=zu0, init_offset=init_offset, K=5, N_per_set=5, shrink_tube=0.5
    )

    sol = vehicle.solve_single_final_problem()
    result = vehicle.get_solution(sol=sol)
    vehicle.plot_result(result, key_stride=30)

    dill.dump(zu0, open(f"{rl_file_name}_{agent}_zu0.pkl", "wb"))
    dill.dump(result, open(f"{rl_file_name}_{agent}_zufinal.pkl", "wb"))


if __name__ == "__main__":
    main()
