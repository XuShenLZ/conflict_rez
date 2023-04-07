from itertools import product, combinations
from typing import Dict, Tuple
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
from confrez.control.rect2circles import v2c_ca
from confrez.control.utils import plot_car
from confrez.control.vehicle import Vehicle
from confrez.obstacle_types import GeofenceRegion

from confrez.pytypes import VehiclePrediction, VehicleState
from confrez.vehicle_types import VehicleBody, VehicleConfig


class MultiVehiclePlanner(object):
    """
    Conflict resolution of multiple vehicles
    """

    def __init__(
        self,
        rl_file_name: str,
        ws_config: Dict[str, bool],
        colors: Dict[str, Tuple[float, float, float]],
        init_offsets: Dict[str, VehicleState],
        final_headings: Dict[str, float],
        vehicle_body: VehicleBody = VehicleBody(),
        vehicle_config: VehicleConfig = VehicleConfig(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        self.rl_file_name = rl_file_name
        self.ws_config = ws_config
        self.colors = colors
        self.init_offsets = init_offsets
        self.final_headings = final_headings
        self.vehicle_body = vehicle_body
        self.vehicle_config = vehicle_config
        self.region = region

        self.agents = sorted(self.ws_config.keys())
        self.agent_pairs = list(combinations(self.agents, 2))

        self.rl_tubes = compute_sets(self.rl_file_name)
        self.obstacles = compute_obstacles()

        self.vehicles = {
            agent: Vehicle(
                rl_file_name=self.rl_file_name,
                agent=agent,
                color=self.colors[agent],
                vehicle_config=self.vehicle_config,
                vehicle_body=self.vehicle_body,
                region=self.region,
            )
            for agent in self.agents
        }

    def solve_single_problems(
        self,
        N: int = 30,
        K: int = 5,
        N_per_set: int = 5,
        dt: float = 0.1,
        shrink_tube: float = 0.5,
        dmin: float = 0.05,
    ):
        """
        solve single vehicle control problems
        """
        self.single_results = {agent: VehiclePrediction() for agent in self.agents}

        for agent in self.agents:
            print(f"==== Solving single vehicle problem for {agent} ====")
            vehicle = self.vehicles[agent]

            zu0 = vehicle.state_ws(
                N=N,
                dt=dt,
                init_offset=self.init_offsets[agent],
                final_heading=self.final_headings[agent],
                shrink_tube=shrink_tube,
                spline_ws=self.ws_config[agent],
            )

            zu0 = vehicle.dual_ws(zu0=zu0)
            zu0 = vehicle.interp_ws_for_collocation(zu0=zu0, K=K, N_per_set=N_per_set)

            vehicle.setup_single_final_problem(
                zu0=zu0,
                init_offset=self.init_offsets[agent],
                final_heading=self.final_headings[agent],
                K=K,
                N_per_set=N_per_set,
                shrink_tube=shrink_tube,
                dmin=dmin,
            )

            sol = vehicle.solve_single_final_problem()
            self.single_results[agent] = vehicle.get_solution(sol=sol)

    def solve_final_problem_circles(
        self,
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.5,
        dmin: float = 0.05,
        d_buffer: float = 0.2,
    ):
        """
        solve joint collision avoidance problem. Using circles to approximate the rectangular vehicle body
        """
        print("Solving joint final problem...")
        # Initial guess for time step
        dt0 = np.mean([self.single_results[agent].dt for agent in self.agents])

        opti = ca.Opti()
        dt = opti.variable()
        opti.set_initial(dt, dt0)

        J = 0

        for agent in self.agents:
            vehicle = self.vehicles[agent]

            vehicle.setup_single_final_problem(
                zu0=self.single_results[agent],
                init_offset=self.init_offsets[agent],
                opti=opti,
                dt=dt,
                K=K,
                N_per_set=N_per_set,
                dmin=dmin,
                shrink_tube=shrink_tube,
            )

            J += vehicle.J

        for agent in self.agents:
            this_vehicle = self.vehicles[agent]
            others = self.agents - {agent}
            for other in others:
                other_vehicle = self.vehicles[other]

                N_min = min(this_vehicle.N, other_vehicle.N)

                for i, k in product(range(N_min), range(K)):
                    self_xcs, self_ycs = v2c_ca(
                        this_vehicle.x[i, k],
                        this_vehicle.y[i, k],
                        this_vehicle.psi[i, k],
                        self.vehicle_body,
                    )
                    other_xcs, other_ycs = v2c_ca(
                        other_vehicle.x[i, k],
                        other_vehicle.y[i, k],
                        other_vehicle.psi[i, k],
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
                            >= (self.vehicle_body.w + d_buffer) ** 2
                        )

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

        N_max = np.max([self.vehicles[agent].N for agent in self.agents])

        final_t = np.linspace(
            0, N_max * sol.value(dt), N_max * (K + 1) + 1, endpoint=True
        )

        self.final_results = {agent: VehiclePrediction() for agent in self.agents}
        for agent in self.agents:
            self.vehicles[agent].get_solution(sol=sol)
            self.final_results[agent] = self.vehicles[agent].interpolate_states(final_t)

    def joint_dual_ws(self, K=5, verbose=0):
        """
        warm starting the dual multipliers for joint collision avoidance
        """
        print("warm starting dual variables for joint OBCA...")
        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        opti = ca.Opti()

        obj = 0

        l_joint = {}
        s_joint = {}
        d_joint = {}
        for pair in self.agent_pairs:
            agent, other = pair

            # lambda[agent][other] = mu[other][agent]. So we no longer use notation $\mu$ anymore
            if agent not in l_joint:
                l_joint[agent] = {}
            if other not in l_joint:
                l_joint[other] = {}

            this_vehicle = self.vehicles[agent]
            this_x = self.single_results[agent].x.reshape((this_vehicle.N, K + 1))
            this_y = self.single_results[agent].y.reshape((this_vehicle.N, K + 1))
            this_psi = self.single_results[agent].psi.reshape((this_vehicle.N, K + 1))

            other_vehicle = self.vehicles[other]
            other_x = self.single_results[other].x.reshape((other_vehicle.N, K + 1))
            other_y = self.single_results[other].y.reshape((other_vehicle.N, K + 1))
            other_psi = self.single_results[other].psi.reshape((other_vehicle.N, K + 1))

            N_min = min(this_vehicle.N, other_vehicle.N)

            l_joint[agent][other] = [
                [opti.variable(4) for _ in range(K + 1)] for _ in range(N_min)
            ]
            l_joint[other][agent] = [
                [opti.variable(4) for _ in range(K + 1)] for _ in range(N_min)
            ]
            s_joint[pair] = [
                [opti.variable(2) for _ in range(K + 1)] for _ in range(N_min)
            ]
            d_joint[pair] = [
                [opti.variable() for _ in range(K + 1)] for _ in range(N_min)
            ]

            for i, k in product(range(N_min), range(K + 1)):
                lik = l_joint[agent][other][i][k]
                mik = l_joint[other][agent][i][k]
                sik = s_joint[pair][i][k]
                dik = d_joint[pair][i][k]
                opti.subject_to(lik >= 0)
                opti.subject_to(mik >= 0)

                this_t = ca.vertcat(this_x[i, k], this_y[i, k])
                this_R = np.array(
                    [
                        [np.cos(-this_psi[i, k]), -np.sin(-this_psi[i, k])],
                        [np.sin(-this_psi[i, k]), np.cos(-this_psi[i, k])],
                    ]
                )
                this_A = veh_G @ this_R
                this_b = veh_G @ this_R @ this_t + veh_g

                other_t = ca.vertcat(other_x[i, k], other_y[i, k])
                other_R = np.array(
                    [
                        [np.cos(-other_psi[i, k]), -np.sin(-other_psi[i, k])],
                        [np.sin(-other_psi[i, k]), np.cos(-other_psi[i, k])],
                    ]
                )
                other_A = veh_G @ other_R
                other_b = veh_G @ other_R @ other_t + veh_g

                opti.subject_to(-ca.dot(this_b, lik) - ca.dot(other_b, mik) == dik)
                opti.subject_to(this_A.T @ lik + sik == np.zeros(2))
                opti.subject_to(other_A.T @ mik - sik == np.zeros(2))
                opti.subject_to(ca.dot(sik, sik) <= 1)

                # ================== Old formulation ==================
                # opti.subject_to(
                #     (
                #         ca.dot(-veh_g, mik) + ca.dot((other_A @ this_t - other_b), lik)
                #         == dik
                #     )
                # )
                # opti.subject_to(
                #     veh_G.T @ mik + this_R.T @ other_A.T @ lik == np.zeros(2)
                # )
                # opti.subject_to(ca.dot(other_A.T @ lik, other_A.T @ lik) <= 1)

                obj -= dik

        opti.minimize(obj)
        p_opts = {"expand": True}
        s_opts = {"print_level": verbose}
        opti.solver("ipopt", p_opts, s_opts)

        sol = opti.solve()
        print(sol.stats()["return_status"])

        self.joint_l0 = {}
        self.joint_s0 = {}
        for pair in self.agent_pairs:
            agent, other = pair
            if agent not in self.joint_l0:
                self.joint_l0[agent] = {}
            if other not in self.joint_l0:
                self.joint_l0[other] = {}

            this_vehicle = self.vehicles[agent]

            other_vehicle = self.vehicles[other]

            N_min = min(this_vehicle.N, other_vehicle.N)

            self.joint_l0[agent][other] = [
                [None for _ in range(K + 1)] for _ in range(N_min)
            ]
            self.joint_l0[other][agent] = [
                [None for _ in range(K + 1)] for _ in range(N_min)
            ]
            self.joint_s0[pair] = [[None for _ in range(K + 1)] for _ in range(N_min)]
            for i, k in product(range(N_min), range(K + 1)):
                lik = l_joint[agent][other][i][k]
                mik = l_joint[other][agent][i][k]
                sik = s_joint[pair][i][k]

                self.joint_l0[agent][other][i][k] = sol.value(lik)
                self.joint_l0[other][agent][i][k] = sol.value(mik)
                self.joint_s0[pair][i][k] = sol.value(sik)

    def solve_final_problem_obca(
        self,
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.5,
        dmin: float = 0.05,
        interp_dt: float = None,
    ):
        """
        solve joint collision avoidance problem with OBCA
        NOTE: This provides the exact solution, may be slower than circle approximation
        """

        self.joint_dual_ws(K=K)

        print("Solving joint final problem with obca...")
        # Initial guess for time step
        dt0 = np.mean([self.single_results[agent].dt for agent in self.agents])

        veh_G = self.vehicle_body.A
        veh_g = self.vehicle_body.b

        opti = ca.Opti()
        dt = opti.variable()
        opti.set_initial(dt, dt0)

        J = 0

        for agent in self.agents:
            vehicle = self.vehicles[agent]

            vehicle.setup_single_final_problem(
                zu0=self.single_results[agent],
                init_offset=self.init_offsets[agent],
                final_heading=self.final_headings[agent],
                opti=opti,
                dt=dt,
                K=K,
                N_per_set=N_per_set,
                dmin=dmin,
                shrink_tube=shrink_tube,
            )

            J += vehicle.J

        l_joint = {}
        s_joint = {}
        for pair in self.agent_pairs:
            agent, other = pair
            if agent not in l_joint:
                l_joint[agent] = {}
            if other not in l_joint:
                l_joint[other] = {}

            this_vehicle = self.vehicles[agent]
            this_x = this_vehicle.x
            this_y = this_vehicle.y
            this_psi = this_vehicle.psi

            other_vehicle = self.vehicles[other]
            other_x = other_vehicle.x
            other_y = other_vehicle.y
            other_psi = other_vehicle.psi

            N_min = min(this_vehicle.N, other_vehicle.N)

            l_joint[agent][other] = [
                [opti.variable(4) for _ in range(K + 1)] for _ in range(N_min)
            ]
            l_joint[other][agent] = [
                [opti.variable(4) for _ in range(K + 1)] for _ in range(N_min)
            ]
            s_joint[pair] = [
                [opti.variable(2) for _ in range(K + 1)] for _ in range(N_min)
            ]

            for i, k in product(range(N_min), range(K + 1)):
                lik = l_joint[agent][other][i][k]
                mik = l_joint[other][agent][i][k]
                sik = s_joint[pair][i][k]
                opti.subject_to(lik >= 0)
                opti.subject_to(mik >= 0)

                opti.set_initial(lik, self.joint_l0[agent][other][i][k])
                opti.set_initial(mik, self.joint_l0[other][agent][i][k])
                opti.set_initial(sik, self.joint_s0[pair][i][k])

                this_t = ca.vertcat(this_x[i, k], this_y[i, k])
                this_R = ca.vertcat(
                    ca.horzcat(ca.cos(-this_psi[i, k]), -ca.sin(-this_psi[i, k])),
                    ca.horzcat(ca.sin(-this_psi[i, k]), ca.cos(-this_psi[i, k])),
                )
                this_A = veh_G @ this_R
                this_b = veh_G @ this_R @ this_t + veh_g

                other_t = ca.vertcat(other_x[i, k], other_y[i, k])
                other_R = ca.vertcat(
                    ca.horzcat(ca.cos(-other_psi[i, k]), -ca.sin(-other_psi[i, k])),
                    ca.horzcat(ca.sin(-other_psi[i, k]), ca.cos(-other_psi[i, k])),
                )
                other_A = veh_G @ other_R
                other_b = veh_G @ other_R @ other_t + veh_g

                # See Convex Optimization Book Section 8.2 for this formulation
                opti.subject_to(-ca.dot(this_b, lik) - ca.dot(other_b, mik) >= dmin)
                opti.subject_to(this_A.T @ lik + sik == np.zeros(2))
                opti.subject_to(other_A.T @ mik - sik == np.zeros(2))

                opti.subject_to(ca.dot(sik, sik) <= 1)

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

        N_max = np.max([self.vehicles[agent].N for agent in self.agents])

        if interp_dt is None:
            final_t = np.linspace(
                0, N_max * sol.value(dt), N_max * (K + 1) + 1, endpoint=True
            )
        else:
            final_t = np.arange(0, N_max * sol.value(dt), interp_dt)

        self.final_results = {agent: VehiclePrediction() for agent in self.agents}
        for agent in self.agents:
            self.vehicles[agent].get_solution(sol=sol)
            self.final_results[agent] = self.vehicles[agent].interpolate_states(final_t)

    def plot_results(self, interval: int = None):
        """
        Plot multi vehicle results
        """
        print("Plotting...")
        if interval is None:
            interval = int(
                (
                    self.final_results["vehicle_0"].t[1]
                    - self.final_results["vehicle_0"].t[0]
                )
                * 1000
            )

        plt.figure()
        static_vehicles = compute_static_vehicles()
        parking_lines = compute_parking_lines()

        for agent in sorted(self.agents):
            ax = plt.subplot(2, 1, 1)
            plt.plot(
                self.final_results[agent].t,
                self.final_results[agent].x,
                label=agent + "_final",
            )
            plt.plot(
                self.single_results[agent].t,
                self.single_results[agent].x,
                "--",
                label=agent + "_single",
            )
            ax.set_ylabel("X (m)")
            ax.legend()

            ax = plt.subplot(2, 1, 2)
            plt.plot(
                self.final_results[agent].t,
                self.final_results[agent].y,
                label=agent + "_final",
            )
            plt.plot(
                self.single_results[agent].t,
                self.single_results[agent].y,
                "--",
                label=agent + "_single",
            )
            ax.set_ylabel("Y (m)")
            ax.set_xlabel("Time (s)")
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.rl_file_name + "_XY_trace_single_vs_final.png")

        plt.figure()
        ax = plt.gca()
        for obstalce in self.obstacles:
            obstalce.plot(ax, facecolor="b", alpha=0.5)
        for agent in self.agents:
            plt.plot(
                self.final_results[agent].x,
                self.final_results[agent].y,
                color=self.colors[agent]["front"],
                label=agent,
            )
        plt.axis("equal")
        plt.savefig(self.rl_file_name + "_XY_final_traj.png")

        print("Generating animation...")
        fig = plt.figure()
        ax = plt.gca()

        def plot_frame(i):
            ax.clear()
            for obstacle in self.obstacles:
                obstacle.plot(ax, facecolor=(0 / 255, 128 / 255, 255 / 255))
            for obstacle in static_vehicles:
                obstacle.plot(ax, fill=False, edgecolor="k", hatch="///")
            for line in parking_lines:
                plt.plot(line[:, 0], line[:, 1], "k--", linewidth=1)

            for j, agent in enumerate(sorted(self.agents)):
                ax.plot(
                    self.final_results[agent].x[: i + 1],
                    self.final_results[agent].y[: i + 1],
                    color=self.colors[agent]["front"],
                    label=agent,
                    zorder=j,
                )
                plot_car(
                    self.final_results[agent].x[i],
                    self.final_results[agent].y[i],
                    self.final_results[agent].psi[i],
                    self.vehicle_body,
                    text=j,
                    zorder=10 + j,
                )
            ax.axis("off")
            ax.set_aspect("equal")
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(0.96, 0.97),
                fontsize="large",
            )
            plt.tight_layout()

        ani = FuncAnimation(
            fig,
            plot_frame,
            frames=len(self.final_results["vehicle_0"].t),
            interval=interval,
            repeat=True,
        )

        fps = int(1000 / interval)
        writer = FFMpegWriter(fps=fps)

        animation_filename = f"{self.rl_file_name}_{fps}fps_animation.mp4"
        ani.save(animation_filename, writer=writer)
        print(f"Animation saves as: {animation_filename}")

        plt.show()


def main():
    """
    main
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
    # init_offsets["vehicle_0"].x.x = 0.1
    # init_offsets["vehicle_0"].x.y = 0.1

    final_headings = {
        "vehicle_0": 0,
        "vehicle_1": 3 * np.pi / 2,
        "vehicle_2": np.pi,
        "vehicle_3": np.pi / 2,
    }

    planner = MultiVehiclePlanner(
        rl_file_name=rl_file_name,
        ws_config=ws_config,
        colors=colors,
        init_offsets=init_offsets,
        final_headings=final_headings,
    )

    planner.solve_single_problems(
        N=30, K=5, N_per_set=5, dt=0.1, shrink_tube=0.5, dmin=0.05
    )
    # planner.solve_final_problem_circles(
    #     K=5, N_per_set=5, shrink_tube=0.5, dmin=0.05, d_buffer=0.3
    # )
    planner.solve_final_problem_obca(
        K=5, N_per_set=5, shrink_tube=0.5, dmin=0.05, interp_dt=0.025
    )
    dill.dump(planner.final_results, open(f"{rl_file_name}_opt.pkl", "wb"))
    planner.plot_results()


if __name__ == "__main__":
    main()
