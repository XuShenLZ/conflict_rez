from itertools import product
from turtle import color
from typing import Dict, Tuple
import numpy as np
import casadi as ca

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from confrez.control.compute_sets import compute_obstacles, compute_sets
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
        vehicle_body: VehicleBody = VehicleBody(),
        vehicle_config: VehicleConfig = VehicleConfig(),
        region: GeofenceRegion = GeofenceRegion(),
    ) -> None:
        self.rl_file_name = rl_file_name
        self.ws_config = ws_config
        self.colors = colors
        self.init_offsets = init_offsets
        self.vehicle_body = vehicle_body
        self.vehicle_config = vehicle_config
        self.region = region

        self.agents = set(self.ws_config.keys())

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
                shrink_tube=shrink_tube,
                spline_ws=self.ws_config[agent],
            )

            zu0 = vehicle.dual_ws(zu0=zu0)
            zu0 = vehicle.interp_ws_for_collocation(zu0=zu0, K=K, N_per_set=N_per_set)

            vehicle.setup_single_final_problem(
                zu0=zu0,
                init_offset=self.init_offsets[agent],
                K=K,
                N_per_set=N_per_set,
                shrink_tube=shrink_tube,
                dmin=dmin,
            )

            sol = vehicle.solve_single_final_problem()
            self.single_results[agent] = vehicle.get_solution(sol=sol)

    def solve_final_problem(
        self,
        K: int = 5,
        N_per_set: int = 5,
        shrink_tube: float = 0.5,
        dmin: float = 0.05,
        d_buffer: float = 0.2,
    ):
        """
        solve joint collision avoidance problem
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

    def plot_results(self, interval: int = 40):
        """
        Plot multi vehicle results
        """
        print("Plotting...")
        plt.figure()
        for agent in self.agents:
            ax = plt.subplot(2, 1, 1)
            plt.plot(
                self.single_results[agent].t,
                self.single_results[agent].x,
                label=agent + "_single",
            )
            plt.plot(
                self.final_results[agent].t,
                self.final_results[agent].x,
                label=agent + "_final",
            )
            ax.set_ylabel("X (m)")
            ax.legend()

            ax = plt.subplot(2, 1, 2)
            plt.plot(
                self.single_results[agent].t,
                self.single_results[agent].y,
                label=agent + "_single",
            )
            plt.plot(
                self.final_results[agent].t,
                self.final_results[agent].y,
                label=agent + "_final",
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
                obstacle.plot(ax, facecolor="b", alpha=0.5)
            for agent in self.agents:
                ax.plot(
                    self.final_results[agent].x,
                    self.final_results[agent].y,
                    color=self.colors[agent]["front"],
                    label=agent,
                )
                plot_car(
                    self.final_results[agent].x[i],
                    self.final_results[agent].y[i],
                    self.final_results[agent].psi[i],
                    self.vehicle_body,
                )
            ax.set_aspect("equal")

        ani = FuncAnimation(
            fig,
            plot_frame,
            frames=len(self.single_results["vehicle_0"].t),
            interval=interval,
            repeat=True,
        )

        writer = FFMpegWriter(fps=int(1000 / interval))
        ani.save(self.rl_file_name + "_animation.mp4", writer=writer)

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
    # init_offsets["vehicle_0"].e.psi = np.pi / 20

    planner = MultiVehiclePlanner(
        rl_file_name=rl_file_name,
        ws_config=ws_config,
        colors=colors,
        init_offsets=init_offsets,
    )

    planner.solve_single_problems(
        N=30, K=5, N_per_set=5, dt=0.1, shrink_tube=0.5, dmin=0.05
    )
    planner.solve_final_problem(
        K=5, N_per_set=5, shrink_tube=0.5, dmin=0.05, d_buffer=0.2
    )
    planner.plot_results(interval=40)


if __name__ == "__main__":
    main()
