from typing import Tuple
import casadi as ca
import numpy as np

from scipy.interpolate import interp1d
from confrez.control.utils import plot_car
from confrez.obstacle_types import GeofenceRegion

from confrez.pytypes import VehiclePrediction
from confrez.vehicle_types import VehicleBody, VehicleConfig
from confrez.control.dynamic_model import kinematic_bicycle_ct
from confrez.control.compute_sets import compute_initial_states, compute_sets

import matplotlib.pyplot as plt


class ConflictPlanner(object):
    """
    Planner for conflict resolution
    """

    def __init__(
        self,
        rl_file_name: str = "1v_rl_traj",
        vehicle_body: VehicleBody = VehicleBody(),
        vehicle_config: VehicleConfig = VehicleConfig(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        """
        `vehicle_body`: VehicleBody object
        `vehicle_config`: VehicleConfig object
        """
        self.vehicle_body = vehicle_body
        self.vehicle_config = vehicle_config
        self.region = region
        self.rl_sets = compute_sets(rl_file_name)
        self.init_states = compute_initial_states(rl_file_name, vehicle_body)

        self.num_sets = len(self.rl_sets["vehicle_0"])

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

    def solve_ws(
        self,
        N: int = 30,
        dt: float = 0.1,
        bounded_input: bool = False,
        shrink_tube: float = 0.8,
    ):
        """
        setup the warm start optimization problem
        """
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

        opti.subject_to(x[0] == self.init_states["vehicle_0"].x.x)
        opti.subject_to(y[0] == self.init_states["vehicle_0"].x.y)
        opti.subject_to(psi[0] == self.init_states["vehicle_0"].e.psi)
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

        for i in range(self.num_sets):
            k = N * i

            back = ca.vertcat(x[k], y[k])
            A = self.rl_sets["vehicle_0"][i]["back"].A
            b = self.rl_sets["vehicle_0"][i]["back"].b
            opti.subject_to(A @ back <= b - shrink_tube)

            front = ca.vertcat(
                x[k] + self.vehicle_body.wb * ca.cos(psi[k]),
                y[k] + self.vehicle_body.wb * ca.sin(psi[k]),
            )
            A = self.rl_sets["vehicle_0"][i]["front"].A
            b = self.rl_sets["vehicle_0"][i]["front"].b
            opti.subject_to(A @ front <= b - shrink_tube)

        opti.minimize(J)

        p_opts = {"expand": True}
        s_opts = {
            "print_level": 5,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            "max_iter": 500,
            # "mumps_mem_percent": 64000,
        }
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

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

    def solve(
        self,
        zu0: VehiclePrediction,
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.8,
        bound_dt: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.5,
    ) -> VehiclePrediction:
        """
        `zu0`: state-input warm start
        `N_per_set`: number of intervals to arrive at each rl set
        `K`: order of collocation polynomial
        Return: VehiclePrediction object for states and inputs
        """
        # Interpolate for collocation
        fx0 = interp1d(zu0.t, zu0.x)
        fy0 = interp1d(zu0.t, zu0.y)
        fpsi0 = interp1d(zu0.t, zu0.psi)
        fv0 = interp1d(zu0.t, zu0.v)
        fdelta0 = interp1d(zu0.t, zu0.u_steer)

        fa0 = interp1d(zu0.t, zu0.u_a)
        fw0 = interp1d(zu0.t, zu0.u_steer_dot)

        # Collocation intervals
        N = N_per_set * (self.num_sets - 1)

        # Collocation time steps
        tau_root = np.append(0, ca.collocation_points(K, "radau"))

        t_interp = np.array([])
        for i in range(N):
            t_interp = np.append(t_interp, i + tau_root)
        t_interp = np.append(t_interp, N)
        t_interp = t_interp / N * zu0.t[-1]

        # Interpolated states, input, and dual variables at collocation points
        x_interp = fx0(t_interp)
        y_interp = fy0(t_interp)
        psi_interp = fpsi0(t_interp)
        v_interp = fv0(t_interp)
        delta_interp = fdelta0(t_interp)

        a_interp = fa0(t_interp)
        w_interp = fw0(t_interp)

        # Initial guess for time step
        dt0 = zu0.t[-1] / N

        A, B, D = self.collocation_coefficients(K=K)

        f_ct = kinematic_bicycle_ct(vehicle_body=self.vehicle_body)

        opti = ca.Opti()

        x = opti.variable(N, K + 1)
        y = opti.variable(N, K + 1)
        psi = opti.variable(N, K + 1)
        v = opti.variable(N, K + 1)
        delta = opti.variable(N, K + 1)

        dt = opti.variable()

        a = opti.variable(N, K + 1)
        w = opti.variable(N, K + 1)

        J = 0

        opti.subject_to(x[0, 0] == self.init_states["vehicle_0"].x.x)
        opti.subject_to(y[0, 0] == self.init_states["vehicle_0"].x.y)
        opti.subject_to(psi[0, 0] == self.init_states["vehicle_0"].e.psi)

        opti.subject_to(v[0, 0] == 0)
        opti.subject_to(delta[0, 0] == 0)

        opti.subject_to(a[0, 0] == 0)
        opti.subject_to(w[0, 0] == 0)

        if bound_dt:
            opti.subject_to(opti.bounded(dt_min, dt, dt_max))

        for i in range(N):
            for k in range(K + 1):
                # collocation point variables
                opti.subject_to(
                    opti.bounded(self.region.x_min, x[i, k], self.region.x_max)
                )
                opti.subject_to(
                    opti.bounded(self.region.y_min, y[i, k], self.region.y_max)
                )

                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.v_min, v[i, k], self.vehicle_config.v_max
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.delta_min,
                        delta[i, k],
                        self.vehicle_config.delta_max,
                    )
                )

                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.a_min, a[i, k], self.vehicle_config.a_max
                    )
                )
                opti.subject_to(
                    opti.bounded(
                        self.vehicle_config.w_delta_min,
                        w[i, k],
                        self.vehicle_config.w_delta_max,
                    )
                )

                # Collocation constraints
                state = ca.vertcat(x[i, k], y[i, k], psi[i, k], v[i, k], delta[i, k])
                input = ca.vertcat(a[i, k], w[i, k])
                func_ode = f_ct(state, input)
                poly_ode = 0

                for j in range(K + 1):
                    zij = ca.vertcat(x[i, j], y[i, j], psi[i, j], v[i, j], delta[i, j])
                    poly_ode += A[j, k] * zij / dt

                opti.subject_to(poly_ode == func_ode)

                # Cost
                error = (
                    0
                    # + (x[i, k] - x_interp[i * K + k]) ** 2
                    # + (y[i, k] - y_interp[i * K + k]) ** 2
                    # + (psi[i, k] - psi_interp[i * K + k]) ** 2
                    + a[i, k] ** 2
                    + (v[i, k] ** 2) * (w[i, k] ** 2)
                    + delta[i, k] ** 2
                )
                J += B[k] * error * dt

            # Continuity constraints
            if i >= 1:
                poly_prev = 0
                input_prev = 0
                for j in range(K + 1):
                    zimj = ca.vertcat(
                        x[i - 1, j],
                        y[i - 1, j],
                        psi[i - 1, j],
                        v[i - 1, j],
                        delta[i - 1, j],
                    )
                    uimj = ca.vertcat(a[i - 1, j], w[i - 1, j])
                    poly_prev += D[j] * zimj
                    input_prev += D[j] * uimj

                zi0 = ca.vertcat(x[i, 0], y[i, 0], psi[i, 0], v[i, 0], delta[i, 0])
                ui0 = ca.vertcat(a[i, 0], w[i, 0])
                opti.subject_to(poly_prev == zi0)
                opti.subject_to(input_prev == ui0)

                # RL set constraints
                q, r = divmod(i, N_per_set)
                if r == 0:
                    back = ca.vertcat(x[i, 0], y[i, 0])
                    tA = self.rl_sets["vehicle_0"][q]["back"].A
                    tb = self.rl_sets["vehicle_0"][q]["back"].b
                    opti.subject_to(tA @ back <= tb - shrink_tube)

                    front = ca.vertcat(
                        x[i, 0] + self.vehicle_body.wb * ca.cos(psi[i, 0]),
                        y[i, 0] + self.vehicle_body.wb * ca.sin(psi[i, 0]),
                    )
                    tA = self.rl_sets["vehicle_0"][q]["front"].A
                    tb = self.rl_sets["vehicle_0"][q]["front"].b
                    opti.subject_to(tA @ front <= tb - shrink_tube)

        # Final constraints
        zF = 0
        uF = 0
        for j in range(K + 1):
            zimj = ca.vertcat(
                x[N - 1, j], y[N - 1, j], psi[N - 1, j], v[N - 1, j], delta[N - 1, j]
            )
            uimj = ca.vertcat(a[N - 1, j], w[N - 1, j])
            zF += D[j] * zimj
            uF += D[j] * uimj

        # opti.subject_to(zF[0] == x_interp[-1])
        # opti.subject_to(zF[1] == y_interp[-1])
        # opti.subject_to(zF[2] == psi_interp[-1])

        # RL set constraints
        back = ca.vertcat(zF[0], zF[1])
        tA = self.rl_sets["vehicle_0"][-1]["back"].A
        tb = self.rl_sets["vehicle_0"][-1]["back"].b
        opti.subject_to(tA @ back <= tb - shrink_tube)

        front = ca.vertcat(
            zF[0] + self.vehicle_body.wb * ca.cos(zF[2]),
            zF[1] + self.vehicle_body.wb * ca.sin(zF[2]),
        )
        tA = self.rl_sets["vehicle_0"][-1]["front"].A
        tb = self.rl_sets["vehicle_0"][-1]["front"].b
        opti.subject_to(tA @ front <= tb - shrink_tube)
        # opti.subject_to(zF[3] == 0)
        # opti.subject_to(zF[4] == 0)

        # opti.subject_to(uF[0] == 0)
        # opti.subject_to(uF[1] == 0)

        opti.minimize(J + (N * dt) ** 2)

        # Initial guess
        opti.set_initial(x, np.array(x_interp[:-1]).reshape(N, K + 1))
        opti.set_initial(y, np.array(y_interp[:-1]).reshape(N, K + 1))
        opti.set_initial(psi, np.array(psi_interp[:-1]).reshape(N, K + 1))
        opti.set_initial(v, np.array(v_interp[:-1]).reshape(N, K + 1))
        opti.set_initial(delta, np.array(delta_interp[:-1]).reshape(N, K + 1))

        opti.set_initial(a, np.array(a_interp[:-1]).reshape(N, K + 1))
        opti.set_initial(w, np.array(w_interp[:-1]).reshape(N, K + 1))

        opti.set_initial(dt, dt0)

        p_opts = {"expand": True}
        s_opts = {
            "print_level": 5,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            # "max_iter": 300,
            # "mumps_mem_percent": 64000,
            "linear_solver": "ma97",
        }
        opti.solver("ipopt", p_opts, s_opts)
        sol = opti.solve()

        x_opt = np.array(sol.value(x)).flatten()
        y_opt = np.array(sol.value(y)).flatten()
        psi_opt = np.array(sol.value(psi)).flatten()
        v_opt = np.array(sol.value(v)).flatten()
        delta_opt = np.array(sol.value(delta)).flatten()

        a_opt = np.array(sol.value(a)).flatten()
        w_opt = np.array(sol.value(w)).flatten()

        result = VehiclePrediction()

        result.t = t_interp * N * sol.value(dt)
        result.x = np.append(x_opt, x_opt[-1])
        result.y = np.append(y_opt, y_opt[-1])
        result.psi = np.append(psi_opt, psi_opt[-1])

        result.v = np.append(v_opt, 0)

        result.u_a = np.append(a_opt, 0)
        result.u_steer = np.append(delta_opt, 0)
        result.u_steer_dot = np.append(w_opt, 0)

        self.get_interpolator(K=K, N=N, dt=sol.value(dt), opt=result)

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
        for body_sets in self.rl_sets["vehicle_0"]:
            body_sets["front"].plot(ax, facecolor="g", alpha=0.5)
            body_sets["back"].plot(ax, facecolor="r", alpha=0.5)

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

        plt.figure(figsize=(10, 5))
        for i, body_sets in enumerate(self.rl_sets["vehicle_0"]):
            ax = plt.subplot(2, 4, i + 1)
            body_sets["front"].plot(ax, facecolor="g", alpha=0.5)
            body_sets["back"].plot(ax, facecolor="r", alpha=0.5)

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
    planner = ConflictPlanner()
    zu0 = planner.solve_ws(N=30)
    # planner.plot_result(zu0, key_stride=30)
    result = planner.solve(zu0, K=5, N_per_set=5)
    planner.plot_result(result, key_stride=30)


if __name__ == "__main__":
    main()
