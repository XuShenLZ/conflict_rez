from itertools import product
from typing import Dict, Tuple, Union, List
import numpy as np
import casadi as ca
import dill

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from confrez.control.compute_sets import (
    compute_obstacles,
    compute_parking_lines,
    compute_sets,
    compute_static_vehicles,
)
from confrez.control.dynamic_model import kinematic_bicycle_ct
from confrez.control.utils import plot_car
from confrez.control.vehicle import Vehicle
from confrez.obstacle_types import GeofenceRegion

from confrez.pytypes import VehiclePrediction, VehicleState
from confrez.vehicle_types import VehicleBody, VehicleConfig


class DistributedVehicle(Vehicle):
    """
    Distributed Planner for Multi-vehicle Scenario
    """

    def __init__(
        self,
        rl_file_name: str,
        agent: str,
        color: Dict[str, Tuple[float, float, float]],
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.5,
        dmin: float = 0.05,
        init_offset: VehicleState = VehicleState(),
        vehicle_config: VehicleConfig = VehicleConfig(),
        vehicle_body: VehicleBody = VehicleBody(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        super().__init__(
            rl_file_name, agent, color, vehicle_config, vehicle_body, region
        )

        self.K = K
        self.N_per_set = N_per_set
        self.N = self.N_per_set * (
            self.num_sets - 1
        )  # Number of intervals for collocation

        self.init_offset = init_offset

        self.shrink_tube = shrink_tube
        self.dmin = dmin

        self.others: List[DistributedVehicle] = None

        self.main_opti: ca.Opti = None
        self.dual_opti: Dict[str, ca.Opti] = {}

        self.opt_trajectory: VehiclePrediction = None

        self.other_trajectories: Dict[str, VehiclePrediction] = {}

        self.p_pose_for_main = {}
        self.p_self_pose_for_dual = {}
        self.p_other_pose_for_dual = {}

        self.p_lambda_ij_for_main = {}
        self.p_lambda_ji_for_main = {}

        self.p_s_for_main = {}

        self.opt_lambda_ij: Dict[str, np.ndarray] = {}
        self.opt_lambda_ji: Dict[str, np.ndarray] = {}
        self.opt_s: Dict[str, np.ndarray] = {}

    def solve_single_problem(
        self,
        N_ws: int = 30,
        dt: float = 0.1,
        ws_config: bool = True,
    ):
        """
        solve single vehicle control problem
        """
        zu0 = self.state_ws(
            N=N_ws,
            dt=dt,
            init_offset=self.init_offset,
            shrink_tube=self.shrink_tube,
            spline_ws=ws_config,
        )

        zu0 = self.dual_ws(zu0=zu0)
        zu0 = self.interp_ws_for_collocation(
            zu0=zu0, K=self.K, N_per_set=self.N_per_set
        )

        self.setup_single_final_problem(
            zu0=zu0,
            init_offset=self.init_offset,
            K=self.K,
            N_per_set=self.N_per_set,
            shrink_tube=self.shrink_tube,
            dmin=self.dmin,
        )

        sol = self.solve_single_final_problem()
        self.opt_trajectory = self.get_solution(sol=sol)

    def get_others(self, others: List[Vehicle]) -> None:
        """
        Get the name and horizon lengths of other vehicles
        """
        self.others = [v for v in others if v.agent != self.agent]

    def _setup_main(self):
        print("Setting up main trajectory planning problem")

        N = self.N
        K = self.K

        opti = ca.Opti()
        self.main_opti = opti

        self.dt = opti.variable()

        n_obs = len(self.obstacles)

        n_hps = []
        for obs in self.obstacles:
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

        J = 0

        A, B, D = self.collocation_coefficients(K=K)

        f_ct = kinematic_bicycle_ct(vehicle_body=self.vehicle_body)

        opti.subject_to(self.x[0, 0] == self.init_state.x.x + self.init_offset.x.x)
        opti.subject_to(self.y[0, 0] == self.init_state.x.y + self.init_offset.x.y)
        opti.subject_to(
            self.psi[0, 0] == self.init_state.e.psi + self.init_offset.e.psi
        )

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

                opti.subject_to(self.m[i][k] >= 0)

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
                J += B[k] * error * self.dt

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

                    opti.subject_to(
                        ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj)
                        >= self.dmin
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
                q, r = divmod(i, self.N_per_set)
                if r == 0 and i > 0:
                    back = ca.vertcat(self.x[i, 0], self.y[i, 0])
                    tA = self.rl_tube[q]["back"].A
                    tb = self.rl_tube[q]["back"].b
                    opti.subject_to(tA @ back <= tb - self.shrink_tube)

                    front = ca.vertcat(
                        self.x[i, 0] + self.vehicle_body.wb * ca.cos(self.psi[i, 0]),
                        self.y[i, 0] + self.vehicle_body.wb * ca.sin(self.psi[i, 0]),
                    )
                    tA = self.rl_tube[q]["front"].A
                    tb = self.rl_tube[q]["front"].b
                    opti.subject_to(tA @ front <= tb - self.shrink_tube)

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
        tA = self.rl_tube[-1]["back"].A
        tb = self.rl_tube[-1]["back"].b
        opti.subject_to(tA @ back <= tb - self.shrink_tube)

        front = ca.vertcat(
            self.zF[0] + self.vehicle_body.wb * ca.cos(self.zF[2]),
            self.zF[1] + self.vehicle_body.wb * ca.sin(self.zF[2]),
        )
        tA = self.rl_tube[-1]["front"].A
        tb = self.rl_tube[-1]["front"].b
        opti.subject_to(tA @ front <= tb - self.shrink_tube)

        opti.subject_to(self.zF[3] == 0)
        opti.subject_to(self.zF[4] == 0)

        opti.subject_to(self.uF[0] == 0)
        opti.subject_to(self.uF[1] == 0)

        # Obstacles avoidance with other vehicles
        for other_v in self.others:
            other = other_v.agent
            other_N = other_v.N

            self.p_pose_for_main[other] = {
                "x": opti.parameter(other_N, K + 1),
                "y": opti.parameter(other_N, K + 1),
                "psi": opti.parameter(other_N, K + 1),
            }

            N_min = min(N, other_N)

            self.p_lambda_ij_for_main[other] = [
                [opti.parameter(4) for _ in range(K + 1)] for _ in range(N_min)
            ]

            self.p_lambda_ji_for_main[other] = [
                [opti.parameter(4) for _ in range(K + 1)] for _ in range(N_min)
            ]

            self.p_s_for_main[other] = [
                [opti.parameter(2) for _ in range(K + 1)] for _ in range(N_min)
            ]

            for i, k in product(range(N_min), range(K + 1)):
                lik = self.p_lambda_ij_for_main[other][i][k]
                mik = self.p_lambda_ji_for_main[other][i][k]
                sik = self.p_s_for_main[other][i][k]

                this_t = ca.vertcat(self.x[i, k], self.y[i, k])
                this_R = ca.vertcat(
                    ca.horzcat(ca.cos(-self.psi[i, k]), -ca.sin(-self.psi[i, k])),
                    ca.horzcat(ca.sin(-self.psi[i, k]), ca.cos(-self.psi[i, k])),
                )
                this_A = veh_G @ this_R
                this_b = veh_G @ this_R @ this_t + veh_g

                other_x = self.p_pose_for_main[other]["x"]
                other_y = self.p_pose_for_main[other]["y"]
                other_psi = self.p_pose_for_main[other]["psi"]

                other_t = ca.vertcat(other_x[i, k], other_y[i, k])
                other_R = ca.vertcat(
                    ca.horzcat(ca.cos(-other_psi[i, k]), -ca.sin(-other_psi[i, k])),
                    ca.horzcat(ca.sin(-other_psi[i, k]), ca.cos(-other_psi[i, k])),
                )
                other_A = veh_G @ other_R
                other_b = veh_G @ other_R @ other_t + veh_g

                # See Convex Optimization Book Section 8.2 for this formulation
                opti.subject_to(
                    -ca.dot(this_b, lik) - ca.dot(other_b, mik) >= self.dmin
                )
                opti.subject_to(this_A.T @ lik + sik == np.zeros(2))

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

    def _setup_dual(self):
        """
        setup the dual optimization problem
        """
        print("Setting up dual problem")

        N = self.N
        K = self.K

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        self.lambda_ij = {}
        self.lambda_ji = {}
        self.s_ij = {}

        for other_v in self.others:
            other = other_v.agent
            other_N = other_v.N

            opti = ca.Opti("conic")
            # opti = ca.Opti()
            self.dual_opti[other] = opti

            N_min = min(N, other_N)

            self.p_self_pose_for_dual[other] = {
                "x": opti.parameter(N, K + 1),
                "y": opti.parameter(N, K + 1),
                "psi": opti.parameter(N, K + 1),
            }

            self.p_other_pose_for_dual[other] = {
                "x": opti.parameter(other_N, K + 1),
                "y": opti.parameter(other_N, K + 1),
                "psi": opti.parameter(other_N, K + 1),
            }

            self.lambda_ij[other] = [
                [opti.variable(4) for _ in range(K + 1)] for _ in range(N_min)
            ]

            self.lambda_ji[other] = [
                [opti.variable(4) for _ in range(K + 1)] for _ in range(N_min)
            ]

            self.s_ij[other] = [
                [opti.variable(2) for _ in range(K + 1)] for _ in range(N_min)
            ]

            self.opt_lambda_ij[other] = np.zeros((N_min, K + 1, 4))
            self.opt_lambda_ji[other] = np.zeros((N_min, K + 1, 4))
            self.opt_s[other] = np.zeros((N_min, K + 1, 2))

            J = 0

            for i, k in product(range(N_min), range(K + 1)):
                lik = self.lambda_ij[other][i][k]
                mik = self.lambda_ji[other][i][k]
                sik = self.s_ij[other][i][k]

                opti.subject_to(lik >= 0)
                opti.subject_to(mik >= 0)

                this_x = self.p_self_pose_for_dual[other]["x"]
                this_y = self.p_self_pose_for_dual[other]["y"]
                this_psi = self.p_self_pose_for_dual[other]["psi"]

                this_t = ca.vertcat(
                    this_x[i, k],
                    this_y[i, k],
                )
                this_R = ca.vertcat(
                    ca.horzcat(ca.cos(-this_psi[i, k]), -ca.sin(-this_psi[i, k])),
                    ca.horzcat(ca.sin(-this_psi[i, k]), ca.cos(-this_psi[i, k])),
                )

                this_A = veh_G @ this_R
                this_b = veh_G @ this_R @ this_t + veh_g

                other_x = self.p_other_pose_for_dual[other]["x"]
                other_y = self.p_other_pose_for_dual[other]["y"]
                other_psi = self.p_other_pose_for_dual[other]["psi"]

                other_t = ca.vertcat(other_x[i, k], other_y[i, k])
                other_R = ca.vertcat(
                    ca.horzcat(ca.cos(-other_psi[i, k]), -ca.sin(-other_psi[i, k])),
                    ca.horzcat(ca.sin(-other_psi[i, k]), ca.cos(-other_psi[i, k])),
                )

                other_A = veh_G @ other_R
                other_b = veh_G @ other_R @ other_t + veh_g

                # opti.subject_to(
                #     -ca.dot(this_b, lik) - ca.dot(other_b, mik) >= self.dmin
                # )
                opti.subject_to(this_A.T @ lik + sik == np.zeros(2))
                opti.subject_to(other_A.T @ mik - sik == np.zeros(2))
                opti.subject_to(ca.dot(sik, sik) <= 1)

                J += (
                    -ca.dot(this_b, lik)
                    - ca.dot(other_b, mik)
                    - 0.001 * ca.dot(lik, lik)
                    - 0.001 * ca.dot(mik, mik)
                )

            opti.minimize(-J)

            p_opts = {"expand": True}
            # s_opts = {"print_level": verbose}
            opti.solver("gurobi", p_opts)
            # s_opts = {
            #     "print_level": 0,
            #     "tol": 1e-2,
            #     "constr_viol_tol": 1e-2,
            #     # "max_iter": 300,
            #     # "mumps_mem_percent": 64000,
            #     "linear_solver": "ma97",
            # }
            # opti.solver("ipopt", p_opts, s_opts)

    def solve_main(self):
        """
        Solve the main optimization problem for trajectory
        """
        zu0 = self.opt_trajectory

        # all_dts = [traj.dt for traj in self.other_trajectories.values()]
        # all_dts.append(zu0.dt)
        # self.main_opti.set_value(self.dt, max(all_dts))
        self.main_opti.set_initial(self.dt, zu0.dt)

        N = self.N
        K = self.K

        self.main_opti.set_initial(self.x, zu0.x[:-1].reshape(N, K + 1))
        self.main_opti.set_initial(self.y, zu0.y[:-1].reshape(N, K + 1))
        self.main_opti.set_initial(self.psi, zu0.psi[:-1].reshape(N, K + 1))
        self.main_opti.set_initial(self.v, zu0.v[:-1].reshape(N, K + 1))
        self.main_opti.set_initial(self.delta, zu0.u_steer[:-1].reshape(N, K + 1))

        self.main_opti.set_initial(self.a, zu0.u_a[:-1].reshape(N, K + 1))
        self.main_opti.set_initial(self.w, zu0.u_steer_dot[:-1].reshape(N, K + 1))

        for i, k in product(range(N), range(K + 1)):
            self.main_opti.set_initial(self.l[i][k], zu0.l[i][k])
            self.main_opti.set_initial(self.m[i][k], zu0.m[i][k])

        for other_v in self.others:
            other = other_v.agent
            other_N = other_v.N
            N_min = min(N, other_N)

            self.main_opti.set_value(
                self.p_pose_for_main[other]["x"],
                self.other_trajectories[other].x[:-1].reshape((other_N, K + 1)),
            )
            self.main_opti.set_value(
                self.p_pose_for_main[other]["y"],
                self.other_trajectories[other].y[:-1].reshape((other_N, K + 1)),
            )
            self.main_opti.set_value(
                self.p_pose_for_main[other]["psi"],
                self.other_trajectories[other].psi[:-1].reshape((other_N, K + 1)),
            )

            for i, k in product(range(N_min), range(K + 1)):
                lik = self.p_lambda_ij_for_main[other][i][k]
                mik = self.p_lambda_ji_for_main[other][i][k]
                sik = self.p_s_for_main[other][i][k]

                self.main_opti.set_value(lik, self.opt_lambda_ij[other][i, k, :])
                self.main_opti.set_value(mik, self.opt_lambda_ji[other][i, k, :])
                self.main_opti.set_value(sik, self.opt_s[other][i, k, :])

        sol = self.main_opti.solve()
        print(sol.stats()["return_status"])

        self.opt_trajectory = self.get_solution(sol=sol)

    def solve_dual(self):
        """
        Solve the dual problem
        """
        N = self.N
        K = self.K

        for other_v in self.others:
            other = other_v.agent
            other_N = other_v.N

            print(f"======= pair {self.agent} <-> {other} =====")

            opti = self.dual_opti[other]

            N_min = min(N, other_N)

            opti.set_value(
                self.p_self_pose_for_dual[other]["x"],
                self.opt_trajectory.x[:-1].reshape((N, K + 1)),
            )
            opti.set_value(
                self.p_self_pose_for_dual[other]["y"],
                self.opt_trajectory.y[:-1].reshape((N, K + 1)),
            )
            opti.set_value(
                self.p_self_pose_for_dual[other]["psi"],
                self.opt_trajectory.psi[:-1].reshape((N, K + 1)),
            )

            opti.set_value(
                self.p_other_pose_for_dual[other]["x"],
                self.other_trajectories[other].x[:-1].reshape((other_N, K + 1)),
            )
            opti.set_value(
                self.p_other_pose_for_dual[other]["y"],
                self.other_trajectories[other].y[:-1].reshape((other_N, K + 1)),
            )
            opti.set_value(
                self.p_other_pose_for_dual[other]["psi"],
                self.other_trajectories[other].psi[:-1].reshape((other_N, K + 1)),
            )

            for i, k in product(range(N_min), range(K + 1)):
                lik = self.lambda_ij[other][i][k]
                mik = self.lambda_ji[other][i][k]
                sik = self.s_ij[other][i][k]

                opti.set_initial(lik, self.opt_lambda_ij[other][i, k, :])
                opti.set_initial(mik, self.opt_lambda_ji[other][i, k, :])
                opti.set_initial(sik, self.opt_s[other][i, k, :])

            sol = opti.solve()
            print(sol.stats()["return_status"])

            for i, k in product(range(N_min), range(K + 1)):
                lik = self.lambda_ij[other][i][k]
                mik = self.lambda_ji[other][i][k]
                sik = self.s_ij[other][i][k]

                self.opt_lambda_ij[other][i, k, :] = sol.value(lik)
                self.opt_lambda_ji[other][i, k, :] = sol.value(mik)
                self.opt_s[other][i, k, :] = sol.value(sik)

    def get_others_traj(self):
        """
        get others trajectory
        """
        for v in self.others:
            self.other_trajectories[v.agent] = v.opt_trajectory


def main():
    """
    main function
    """
    rl_file_name = "4v_rl_traj"
    ws_config = {
        "vehicle_0": False,
        "vehicle_1": True,
        "vehicle_2": True,
        "vehicle_3": True,
    }
    colors = {
        "vehicle_0": {
            "front": (255 / 255, 119 / 255, 0),
            "back": (128 / 255, 60 / 255, 0),
        },
        "vehicle_1": {
            "front": (0, 255 / 255, 212 / 255),
            "back": (0, 140 / 255, 117 / 255),
        },
        "vehicle_2": {
            "front": (164 / 255, 164 / 255, 164 / 255),
            "back": (64 / 255, 64 / 255, 64 / 255),
        },
        "vehicle_3": {
            "front": (255 / 255, 0, 149 / 255),
            "back": (128 / 255, 0, 74 / 255),
        },
    }

    init_offsets = {
        "vehicle_0": VehicleState(),
        "vehicle_1": VehicleState(),
        "vehicle_2": VehicleState(),
        "vehicle_3": VehicleState(),
    }

    vehicles: List[DistributedVehicle] = []

    for agent in ws_config.keys():
        print(f"===== Solving single agent problem for {agent} =======")
        vehicle = DistributedVehicle(
            rl_file_name=rl_file_name,
            agent=agent,
            color=colors[agent],
            K=5,
            N_per_set=5,
            init_offset=init_offsets[agent],
        )

        vehicle.solve_single_problem(N_ws=30, dt=0.1, ws_config=ws_config[agent])

        vehicles.append(vehicle)

    for vehicle in vehicles:
        print(f"======= {vehicle.agent} =========")
        vehicle.get_others(vehicles)
        vehicle.get_others_traj()

        vehicle._setup_main()
        vehicle._setup_dual()

        vehicle.solve_dual()

    num_iter = 3
    for i in range(num_iter):
        print(f"========== Iter {i} ===========")
        for vehicle in vehicles:
            vehicle.solve_main()

        for vehicle in vehicles:
            vehicle.get_others_traj()

        for vehicle in vehicles:
            vehicle.solve_dual()

    for vehicle in vehicles:
        vehicle.plot_result(result=vehicle.opt_trajectory)


if __name__ == "__main__":
    main()
