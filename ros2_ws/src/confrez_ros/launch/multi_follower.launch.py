from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os

import yaml

confrez_dir = get_package_share_directory("confrez_ros")
config_dir = os.path.join(confrez_dir, "config")

global_params_file = os.path.join(config_dir, "global_params.yaml")
vehicle_params_file = os.path.join(config_dir, "vehicle_params.yaml")
visualizer_params_file = os.path.join(config_dir, "visualizer_params.yaml")

with open(global_params_file, "r") as f:
    parsed_global_params = yaml.safe_load(f)

print(parsed_global_params["/**"]["ros__parameters"]["num_vehicles"])


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="confrez_ros",
                executable="visualizer_node.py",
                name="visualizer",
                parameters=[global_params_file, visualizer_params_file],
                output="screen",
            ),
            # Delay simulator so that the visualization is ready
            TimerAction(
                period=0.5,
                actions=[
                    Node(
                        package="confrez_ros",
                        namespace=f"vehicle_{vehicle_id}",
                        executable="vehicle_node.py",
                        name=f"vehicle_{vehicle_id}",
                        parameters=[
                            global_params_file,
                            vehicle_params_file,
                        ],
                        output="screen",
                        emulate_tty=True,
                    )
                    for vehicle_id in range(
                        parsed_global_params["/**"]["ros__parameters"]["num_vehicles"]
                    )
                ],
            ),
        ]
    )
