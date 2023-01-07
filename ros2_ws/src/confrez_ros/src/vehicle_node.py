#!/usr/bin/env python3

from array import array
import rclpy

from typing import Dict, Tuple
from pathlib import Path
import numpy as np

from confrez.pytypes import VehicleState, VehiclePrediction, NodeParamTemplate
from confrez.base_node import MPClabNode
from confrez.control.vehicle_follower import VehicleFollower

from confrez_ros.msg import VehiclePredictionMsg
from std_msgs.msg import Bool

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

final_headings = {
    "vehicle_0": 0,
    "vehicle_1": 3 * np.pi / 2,
    "vehicle_2": np.pi,
    "vehicle_3": np.pi / 2,
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


class VehicleNodeParams(NodeParamTemplate):
    """
    template that stores all parameters needed for the node as well as default values
    """

    def __init__(self):
        self.timer_period: float = 0.05

        self.num_vehicles: int = 4

        self.rl_file_name: str = str(Path.home()) + "/xu_ws/conflict_rez/4v_rl_traj"


class VehicleNode(MPClabNode):
    """
    Node for path following vehicle
    """

    timer_period: float
    num_vehicles: int
    rl_file_name: str

    def __init__(self):
        """
        init
        """
        super().__init__("vehicle")
        self.get_logger().info("initializing Vehicle...")

        namespace = self.get_namespace()

        # ========= Parameters
        param_template = VehicleNodeParams()
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)

        self.agent = namespace[1:]

        self.spline_ws = spline_ws_config[self.agent]
        self.init_offset = init_offsets[self.agent]
        self.final_heading = final_headings[self.agent]
        self.color = colors[self.agent]

        self.vehicle = VehicleFollower(
            self.rl_file_name,
            agent=self.agent,
            color=self.color,
            init_offset=VehicleState(),
            final_heading=self.final_heading,
        )

        # ======== Publishers, Subscribers, Services
        self.pred_pub = self.create_publisher(VehiclePredictionMsg, "pred", 10)

        self.info_pub = self.create_publisher(Bool, "info", 10)

        self.others = [
            f"vehicle_{id}"
            for id in range(self.num_vehicles)
            if f"vehicle_{id}" != self.agent
        ]
        print(f"current vehicle: {self.agent}. Others: {self.others}")

        self.other_pred_subs = {}
        self.other_info_subs = {}
        self.others_info = {}
        for other in self.others:
            self.other_pred_subs[other] = self.create_subscription(
                VehiclePredictionMsg,
                f"/{other}/pred",
                self.vehicle_pred_cb(other),
                10,
            )

            self.other_info_subs[other] = self.create_subscription(
                Bool, f"/{other}/info", self.vehicle_info_cb(other), 10
            )

            self.others_info[other] = False

        self.get_logger().info(f"Setting up {self.agent}...")
        self.vehicle.others = self.others
        self.vehicle.plan_single_path(spline_ws=self.spline_ws)
        self.vehicle.setup_controller()
        self.vehicle.get_current_ref()

        pred = VehiclePrediction()
        pred.x = array("d", self.vehicle.pred.x)
        pred.y = array("d", self.vehicle.pred.y)
        pred.psi = array("d", self.vehicle.pred.psi)
        pred_msg = VehiclePredictionMsg()
        self.populate_msg(pred_msg, pred)
        self.pred_pub.publish(pred_msg)

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def vehicle_pred_cb(self, other):
        def callback(msg):
            pred = VehiclePrediction()
            self.unpack_msg(msg, pred)
            pred.x = np.array(pred.x)
            pred.y = np.array(pred.y)
            pred.psi = np.array(pred.psi)
            self.vehicle.others_pred[other] = pred

        return callback

    def vehicle_info_cb(self, other):
        def callback(msg):
            self.others_info[other] = msg.data

        return callback

    def timer_callback(self):
        """
        timer loop
        """
        info_msg = Bool()
        info_msg.data = True
        self.info_pub.publish(info_msg)

        if all(list(self.others_info.values())):
            self.vehicle.step()

            pred = VehiclePrediction()
            pred.x = array("d", self.vehicle.pred.x)
            pred.y = array("d", self.vehicle.pred.y)
            pred.psi = array("d", self.vehicle.pred.psi)

            pred_msg = VehiclePredictionMsg()
            self.populate_msg(pred_msg, pred)
            self.pred_pub.publish(pred_msg)


def main(args=None):
    """
    main
    """
    rclpy.init(args=args)

    node = VehicleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Vehicle node is terminated")
    finally:
        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
