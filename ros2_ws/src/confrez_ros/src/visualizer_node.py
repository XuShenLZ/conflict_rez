#!/usr/bin/env python3

from confrez.vehicle_types import VehicleBody
import rclpy

from typing import Dict, Tuple
from pathlib import Path

from confrez.pytypes import VehicleState, VehiclePrediction, NodeParamTemplate
from confrez.base_node import MPClabNode
from confrez.control.realtime_visualizer import RealtimeVisualizer

from confrez_ros.msg import VehiclePredictionMsg
from std_msgs.msg import Bool

colors: Dict[str, Tuple[int, int, int]] = {
    "vehicle_0": (255, 119, 0),
    "vehicle_1": (0, 255, 212),
    "vehicle_2": (164, 164, 164),
    "vehicle_3": (255, 0, 149),
}


class VisualizerNodeParams(NodeParamTemplate):
    """
    template that stores all parameters needed for the node as well as default values
    """

    def __init__(self):
        self.timer_period: float = 0.025

        self.num_vehicles: int = 4


class VisualizerNode(MPClabNode):
    """
    Node for real time visualizer
    """

    timer_period: float
    num_vehicles: int

    def __init__(self):
        """
        init
        """
        super().__init__("visualizer")
        self.get_logger().info("initializing visualizer...")

        namespace = self.get_namespace()

        # ========= Parameters
        param_template = VisualizerNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.colors = colors

        self.vis = RealtimeVisualizer(vehicle_body=VehicleBody())
        self.vis.draw_background()
        self.vis.draw_obstacles()

        self.vehicles = [f"vehicle_{id}" for id in range(self.num_vehicles)]

        self.vehicle_pred_subs = {}
        self.vehicle_preds: Dict[str, VehiclePrediction] = {}
        for agent in self.vehicles:
            self.vehicle_pred_subs[agent] = self.create_subscription(
                VehiclePredictionMsg, f"/{agent}/pred", self.vehicle_pred_cb(agent), 10
            )

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def vehicle_pred_cb(self, agent):
        """
        callback for receiving vehicle prediction messages
        """

        def callback(msg):
            pred = VehiclePrediction()
            self.unpack_msg(msg, pred)
            self.vehicle_preds[agent] = pred

        return callback

    def timer_callback(self):
        """
        timer callback
        """
        self.vis.draw_background()
        self.vis.draw_obstacles()

        for agent in self.vehicle_preds:
            # self.vis.draw_traj(self.vehicle_preds[agent], self.colors[agent])

            state = VehicleState()
            state.x.x = self.vehicle_preds[agent].x[0]
            state.x.y = self.vehicle_preds[agent].y[0]
            state.e.psi = self.vehicle_preds[agent].psi[0]

            self.vis.draw_car(state, self.colors[agent])

        self.vis.render()


def main(args=None):
    """
    main
    """
    rclpy.init(args=args)

    node = VisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Visualizer node is terminated")
    finally:
        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
