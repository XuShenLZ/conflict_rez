from typing import Dict, Tuple, List
import numpy as np
from itertools import product
from scipy.spatial import distance
import matplotlib.pyplot as plt

from tqdm import tqdm

import casadi as ca

from confrez.control.dynamic_model import kinematic_bicycle_rk
from confrez.obstacle_types import GeofenceRegion
from confrez.vehicle_types import VehicleBody, VehicleConfig
from confrez.control.vehicle import Vehicle

from confrez.pytypes import VehiclePrediction, VehicleState


class VehicleFollower(Vehicle):
    """
    Vehicle that avoid obstacles while following the pre-planned path
    """

    def __init__(
        self,
        rl_file_name: str,
        agent: str,
        color: Dict[str, Tuple[float, float, float]],
        init_offset: VehicleState,
        vehicle_config: VehicleConfig = VehicleConfig(),
        vehicle_body: VehicleBody = VehicleBody(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        super().__init__(
            rl_file_name, agent, color, vehicle_config, vehicle_body, region
        )
        self.init_offset = init_offset

        self.state: VehicleState = self.init_state
        self.state.t = 0
        self.state.x.x += self.init_offset.x.x
        self.state.x.y += self.init_offset.x.y
        self.state.e.psi += self.init_offset.e.psi

        self.pred: VehiclePrediction = None

        self.others: Dict[VehicleFollower] = {}
        self.others_pred: Dict[str, VehiclePrediction] = {}

        self.reference_traj: VehiclePrediction = None
        self.ref_idx_lb: int = 0
        self.ref_idx_ub: int = -1
        self.ref_pair: List[np.ndarray] = []

        self.final_traj: VehiclePrediction = VehiclePrediction()
        self.final_traj.t = [self.state.t]
        self.final_traj.x = [self.state.x.x]
        self.final_traj.y = [self.state.x.y]
        self.final_traj.psi = [self.state.e.psi]
        self.final_traj.v = [self.state.v.v]
        self.final_traj.u_steer = [self.state.u.u_steer]
        self.final_traj.u_a = [self.state.u.u_a]
        self.final_traj.u_steer_dot = [self.state.u.u_steer_dot]

    def plan_single_path(
        self,
        N_ws: int = 30,
        dt_ws: float = 0.1,
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.5,
        dmin: float = 0.05,
        spline_ws: bool = True,
        interp_dt: float = 0.01,
    ):
        """
        plan a single vehicle reference path
        """
        zu0 = self.state_ws(
            N=N_ws,
            dt=dt_ws,
            init_offset=self.init_offset,
            shrink_tube=shrink_tube,
            spline_ws=spline_ws,
        )

        zu0 = self.dual_ws(zu0=zu0)
        zu0 = self.interp_ws_for_collocation(zu0=zu0, K=K, N_per_set=N_per_set)

        self.setup_single_final_problem(
            zu0=zu0,
            init_offset=self.init_offset,
            K=K,
            N_per_set=N_per_set,
            shrink_tube=shrink_tube,
            dmin=dmin,
        )

        sol = self.solve_single_final_problem()
        result = self.get_solution(sol=sol)

        interp_time = np.linspace(
            start=result.t[0],
            stop=result.t[-1],
            num=int((result.t[-1] - result.t[0]) / interp_dt),
            endpoint=True,
        )

        self.reference_traj = self.interpolate_states(interp_time)
        self.reference_xy = np.vstack([self.reference_traj.x, self.reference_traj.y]).T

    def get_others(self, others: List[Vehicle]):
        """
        get other vehicles
        """
        self.others = [v for v in others if v.agent != self.agent]

    def setup_controller(self, dt: float = 0.05, N: int = 30, dmin=0.05):
        """
        setup the predictive controller
        """
        print(f"setting up controller for {self.agent}...")

        n_obs = len(self.obstacles)

        n_hps = []
        for obs in self.obstacles:
            n_hps.append(len(obs.b))

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        self.N = N
        self.dt = dt
        self.horizon_interp_ahead = np.linspace(0, N * dt, N, endpoint=False)

        self.opti = ca.Opti()

        self.x = self.opti.variable(N)
        self.y = self.opti.variable(N)
        self.psi = self.opti.variable(N)
        self.v = self.opti.variable(N)
        self.delta = self.opti.variable(N)

        self.a = self.opti.variable(N)
        self.w = self.opti.variable(N)

        self.l = self.opti.variable(N, sum(n_hps))
        self.m = self.opti.variable(N, 4 * n_obs)

        self.current_x = self.opti.parameter()
        self.current_y = self.opti.parameter()
        self.current_psi = self.opti.parameter()
        self.current_v = self.opti.parameter()
        self.current_delta = self.opti.parameter()

        self.current_ref_x = self.opti.parameter(N)
        self.current_ref_y = self.opti.parameter(N)
        self.current_ref_psi = self.opti.parameter(N)

        J = 0

        f_dt = kinematic_bicycle_rk(dt=dt, vehicle_body=self.vehicle_body)

        self.opti.subject_to(self.x[0] == self.current_x)
        self.opti.subject_to(self.y[0] == self.current_y)
        self.opti.subject_to(self.psi[0] == self.current_psi)

        self.opti.subject_to(self.v[0] == self.current_v)
        self.opti.subject_to(self.delta[0] == self.current_delta)

        self.opti.subject_to(self.l[:] >= 0)
        self.opti.subject_to(self.m[:] >= 0)

        for i in range(N):
            self.opti.subject_to(
                self.opti.bounded(self.region.x_min, self.x[i], self.region.x_max)
            )
            self.opti.subject_to(
                self.opti.bounded(self.region.y_min, self.y[i], self.region.y_max)
            )

            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.v_min,
                    self.v[i],
                    self.vehicle_config.v_max,
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.delta_min,
                    self.delta[i],
                    self.vehicle_config.delta_max,
                )
            )

            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.a_min,
                    self.a[i],
                    self.vehicle_config.a_max,
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    self.vehicle_config.w_delta_min,
                    self.w[i],
                    self.vehicle_config.w_delta_max,
                )
            )

            # Dynamic constraints
            if i < N - 1:
                state = ca.vertcat(
                    self.x[i],
                    self.y[i],
                    self.psi[i],
                    self.v[i],
                    self.delta[i],
                )
                input = ca.vertcat(self.a[i], self.w[i])
                state_p = ca.vertcat(
                    self.x[i + 1],
                    self.y[i + 1],
                    self.psi[i + 1],
                    self.v[i + 1],
                    self.delta[i + 1],
                )

                self.opti.subject_to(state_p == f_dt(state, input))

            # Cost
            error = (
                0
                + 100 * (self.x[i] - self.current_ref_x[i]) ** 2
                + 100 * (self.y[i] - self.current_ref_y[i]) ** 2
                + 100 * (self.psi[i] - self.current_ref_psi[i]) ** 2
                + self.a[i] ** 2
                + (self.v[i] ** 2) * (self.w[i] ** 2)
                + self.delta[i] ** 2
            )
            J += error

            # OBCA constraints
            t = ca.vertcat(self.x[i], self.y[i])
            R = ca.vertcat(
                ca.horzcat(ca.cos(self.psi[i]), -ca.sin(self.psi[i])),
                ca.horzcat(ca.sin(self.psi[i]), ca.cos(self.psi[i])),
            )
            for j, obs in enumerate(self.obstacles):
                idx0 = sum(n_hps[:j])
                idx1 = sum(n_hps[: j + 1])
                lj = self.l[i, idx0:idx1].T
                mj = self.m[i, 4 * j : 4 * (j + 1)].T

                self.opti.subject_to(
                    ca.dot(-veh_g, mj) + ca.dot((obs.A @ t - obs.b), lj) >= dmin
                )
                self.opti.subject_to(veh_G.T @ mj + R.T @ obs.A.T @ lj == np.zeros(2))
                self.opti.subject_to(ca.dot(obs.A.T @ lj, obs.A.T @ lj) == 1)

        # Collision avoidance with other vehicles
        self.p_other_pred = {}

        self.lambda_ij = {}
        self.lambda_ji = {}
        self.s = {}

        self.opt_lambda_ij: Dict[str, np.ndarray] = {}
        self.opt_lambda_ji: Dict[str, np.ndarray] = {}
        self.opt_s: Dict[str, np.ndarray] = {}

        for v in self.others:
            other = v.agent

            self.p_other_pred[other] = {
                "x": self.opti.parameter(N),
                "y": self.opti.parameter(N),
                "psi": self.opti.parameter(N),
            }

            self.lambda_ij[other] = self.opti.variable(N, 4)
            self.lambda_ji[other] = self.opti.variable(N, 4)
            self.s[other] = self.opti.variable(N, 2)

            self.opti.subject_to(self.lambda_ij[other][:] >= 0)
            self.opti.subject_to(self.lambda_ji[other][:] >= 0)

            self.opt_lambda_ij[other] = np.zeros((N, 4))
            self.opt_lambda_ji[other] = np.zeros((N, 4))
            self.opt_s[other] = np.zeros((N, 2))

            for i in range(N):
                lik = self.lambda_ij[other][i, :].T
                mik = self.lambda_ji[other][i, :].T
                sik = self.s[other][i, :].T

                this_t = ca.vertcat(self.x[i], self.y[i])
                this_R = ca.vertcat(
                    ca.horzcat(ca.cos(-self.psi[i]), -ca.sin(-self.psi[i])),
                    ca.horzcat(ca.sin(-self.psi[i]), ca.cos(-self.psi[i])),
                )
                this_A = veh_G @ this_R
                this_b = veh_G @ this_R @ this_t + veh_g

                other_x = self.p_other_pred[other]["x"]
                other_y = self.p_other_pred[other]["y"]
                other_psi = self.p_other_pred[other]["psi"]

                other_t = ca.vertcat(other_x[i], other_y[i])
                other_R = ca.vertcat(
                    ca.horzcat(ca.cos(-other_psi[i]), -ca.sin(-other_psi[i])),
                    ca.horzcat(ca.sin(-other_psi[i]), ca.cos(-other_psi[i])),
                )
                other_A = veh_G @ other_R
                other_b = veh_G @ other_R @ other_t + veh_g

                self.opti.subject_to(
                    -ca.dot(this_b, lik) - ca.dot(other_b, mik) >= dmin
                )
                self.opti.subject_to(this_A.T @ lik + sik == np.zeros(2))
                self.opti.subject_to(other_A.T @ mik - sik == np.zeros(2))
                self.opti.subject_to(ca.dot(sik, sik) <= 1)

        self.opti.minimize(J)

        p_opts = {
            "expand": True,
            "print_time": False,
        }
        s_opts = {
            "print_level": 0,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            # "max_iter": 300,
            # "mumps_mem_percent": 64000,
            "linear_solver": "ma97",
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    def get_current_ref(self):
        """
        look up the closest waypoint and interpolate ahead for reference
        """
        # current_xy = np.array([[self.state.x.x, self.state.x.y]])
        # min_idx = (
        #     distance.cdist(
        #         current_xy,
        #         self.reference_xy[self.ref_idx_lb : self.ref_idx_ub, :],
        #         metric="euclidean",
        #     ).argmin()
        #     + self.ref_idx_lb
        # )
        # self.ref_idx_lb = min_idx
        min_idx = np.abs(self.reference_traj.t - self.state.t).argmin()
        interp_t_span = self.reference_traj.t[min_idx] + self.horizon_interp_ahead
        self.ref_idx_ub = np.abs(self.reference_traj.t - interp_t_span[-1]).argmin()

        self.ref_pair.append(
            np.array(
                [
                    [self.reference_traj.x[min_idx], self.reference_traj.y[min_idx]],
                    [self.state.x.x, self.state.x.y],
                ]
            )
        )
        # if abs(self.state.v.v) < 1e-2:
        #     print("speed is almost zero")

        result = self.interpolate_states(time=interp_t_span)

        if self.pred is None:
            self.pred = result.copy()
            self.pred.l = 0.1*np.random.rand(*self.l.shape)
            self.pred.m = 0.1*np.random.rand(*self.m.shape)

        return result

    def get_others_pred(self):
        """
        get others open-loop prediction
        """
        for v in self.others:
            self.others_pred[v.agent] = v.pred

    def _adv_onestep(self, array: np.ndarray):
        """
        advance the array one step forward
        """
        if len(array.shape) == 1:
            result = np.append(array[1:], array[-1])
        elif len(array.shape) == 2:
            result = np.vstack([array[1:, :], array[-1, :]])
        else:
            raise ValueError(
                "unexpected shape when advancing the array to one step ahead."
            )

        return result

    def step(self):
        """
        step the controller
        """
        self.opti.set_value(self.current_x, self.state.x.x)
        self.opti.set_value(self.current_y, self.state.x.y)
        self.opti.set_value(self.current_psi, self.state.e.psi)
        self.opti.set_value(self.current_v, self.state.v.v)
        self.opti.set_value(self.current_delta, self.state.u.u_steer)

        current_ref = self.get_current_ref()

        self.opti.set_value(self.current_ref_x, current_ref.x)
        self.opti.set_value(self.current_ref_y, current_ref.y)
        self.opti.set_value(self.current_ref_psi, current_ref.psi)

        for v in self.others:
            self.opti.set_value(
                self.p_other_pred[v.agent]["x"],
                self._adv_onestep(self.others_pred[v.agent].x),
            )
            self.opti.set_value(
                self.p_other_pred[v.agent]["y"],
                self._adv_onestep(self.others_pred[v.agent].y),
            )
            self.opti.set_value(
                self.p_other_pred[v.agent]["psi"],
                self._adv_onestep(self.others_pred[v.agent].psi),
            )

            self.opti.set_initial(
                self.lambda_ij[v.agent], self._adv_onestep(self.opt_lambda_ij[v.agent])
            )
            self.opti.set_initial(
                self.lambda_ji[v.agent], self._adv_onestep(self.opt_lambda_ji[v.agent])
            )
            self.opti.set_initial(
                self.s[v.agent], self._adv_onestep(self.opt_s[v.agent])
            )

        self.opti.set_initial(self.x, self._adv_onestep(self.pred.x))
        self.opti.set_initial(self.y, self._adv_onestep(self.pred.y))
        self.opti.set_initial(self.psi, self._adv_onestep(self.pred.psi))
        self.opti.set_initial(self.v, self._adv_onestep(self.pred.v))
        self.opti.set_initial(self.delta, self._adv_onestep(self.pred.u_steer))

        self.opti.set_initial(self.a, self._adv_onestep(self.pred.u_a))
        self.opti.set_initial(self.w, self._adv_onestep(self.pred.u_steer_dot))

        self.opti.set_initial(self.l, self._adv_onestep(self.pred.l))
        self.opti.set_initial(self.m, self._adv_onestep(self.pred.m))

        sol = self.opti.solve()
        # print(sol.stats()["return_status"])

        self.pred.x = sol.value(self.x)
        self.pred.y = sol.value(self.y)
        self.pred.psi = sol.value(self.psi)

        self.pred.v = sol.value(self.v)
        self.pred.u_steer = sol.value(self.delta)

        self.pred.u_a = sol.value(self.a)
        self.pred.u_steer_dot = sol.value(self.w)

        self.pred.l = sol.value(self.l)
        self.pred.m = sol.value(self.m)

        for v in self.others:
            self.opt_lambda_ij[v.agent] = sol.value(self.lambda_ij[v.agent])
            self.opt_lambda_ji[v.agent] = sol.value(self.lambda_ji[v.agent])
            self.opt_s[v.agent] = sol.value(self.s[v.agent])

        self.state.t += self.dt
        self.state.x.x = self.pred.x[1]
        self.state.x.y = self.pred.y[1]
        self.state.e.psi = self.pred.psi[1]

        self.state.v.v = self.pred.v[1]
        self.state.u.u_steer = self.pred.u_steer[1]

        self.state.u.u_a = self.pred.u_a[0]
        self.state.u.u_steer_dot = self.pred.u_steer_dot[0]

        self.final_traj.t.append(self.state.t)
        self.final_traj.x.append(self.state.x.x)
        self.final_traj.y.append(self.state.x.y)
        self.final_traj.psi.append(self.state.e.psi)
        self.final_traj.v.append(self.state.v.v)
        self.final_traj.u_steer.append(self.state.u.u_steer)
        self.final_traj.u_a.append(self.state.u.u_a)
        self.final_traj.u_steer_dot.append(self.state.u.u_steer_dot)


def main():
    """
    main
    """
    rl_file_name = "4v_rl_traj"
    agent = "vehicle_0"

    spline_ws_config = {
        "vehicle_0": False,
        "vehicle_1": True,
        "vehicle_2": True,
        "vehicle_3": True,
    }

    init_offsets = {
        "vehicle_0": VehicleState(),
        "vehicle_1": VehicleState(),
        "vehicle_2": VehicleState(),
        "vehicle_3": VehicleState(),
    }

    vehicles: List[VehicleFollower] = []

    for agent in spline_ws_config.keys():

        vehicle = VehicleFollower(
            rl_file_name=rl_file_name,
            agent=agent,
            color={"front": (1, 0, 0), "back": (0, 1, 0)},
            init_offset=init_offsets[agent],
        )

        vehicles.append(vehicle)

    for v in vehicles:
        v.plan_single_path(spline_ws=spline_ws_config[v.agent])
        v.get_others(vehicles)
        v.setup_controller()
        v.get_current_ref()

    for _ in tqdm(range(500)):
        for v in vehicles:
            v.get_others_pred()

        for v in vehicles:
            v.step()

    for v in vehicles:
        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(v.reference_traj.x, v.reference_traj.y, color="b")
        ax1.plot(v.final_traj.x, v.final_traj.y, "--r")
        for pair in v.ref_pair:
            ax1.plot(pair[:, 0], pair[:, 1], color="g", linewidth=0.25)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(v.final_traj.t, v.final_traj.v)

    plt.show()


if __name__ == "__main__":
    main()
