#!/usr/bin/env python3
"""Launch stretch driver + state publishers without ros2 launch CLI."""
# just like python3 this file
# python3 run_stretch_driver_alternate.py

import argparse
import os
import sys

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, LogInfo
from launch.substitutions import Command
from launch_ros.actions import Node
import launch_ros
import stretch_body.robot_params as params
import importlib.resources


# this fixes rviz launch issue
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so'


def generate_launch_description(args):
    # Check is robot
    if 'HELLO_FLEET_ID' not in os.environ:
        print("\nERROR: Must be run on a robot.")
        sys.exit(1)

    stretch_core_path = get_package_share_path('stretch_core')
    ld = LaunchDescription()

    calibrated_backlash = stretch_core_path / 'config' / 'controller_calibration_head.yaml'
    uncalibrated_backlash = stretch_core_path / 'config' / 'controller_calibration_head_factory_default.yaml'
    if calibrated_backlash.is_file():
        backlash_fpath = calibrated_backlash
    else:
        ld.add_action(
            LogInfo(
                msg='\n\nWARNING: Calibrated backlash params not available. Using uncalibrated params.\n'
            )
        )
        backlash_fpath = uncalibrated_backlash

    controller_calibration_file = (
        args.controller_calibration_file
        if args.controller_calibration_file
        else str(backlash_fpath)
    )

    _, r = params.RobotParams.get_params()
    model_name = r['robot']['model_name']
    tool_name = r['robot']['tool']
    uncalibrated_urdf = (
        importlib.resources.files("stretch_urdf")
        / model_name
        / f"stretch_description_{model_name}_{tool_name}.urdf"
    )
    calibrated_urdf = get_package_share_path('stretch_description') / 'urdf' / 'stretch.urdf'
    if calibrated_urdf.is_file():
        robot_description_content = launch_ros.parameter_descriptions.ParameterValue(
            Command(['xacro ', str(calibrated_urdf)]), value_type=str
        )
    else:
        ld.add_action(
            LogInfo(
                msg='\n\nWARNING: Calibrated URDF not available. Using uncalibrated URDF.\n'
            )
        )
        robot_description_content = launch_ros.parameter_descriptions.ParameterValue(
            Command(['xacro ', str(uncalibrated_urdf)]), value_type=str
        )

    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='log',
        parameters=[{'source_list': ['/stretch/joint_states']}, {'rate': 30.0}],
        arguments=['--ros-args', '--log-level', 'error'],
    )
    ld.add_action(joint_state_publisher)

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': robot_description_content}, {'publish_frequency': 30.0}],
        arguments=['--ros-args', '--log-level', 'error'],
    )
    ld.add_action(robot_state_publisher)

    alternate_driver_path = os.path.join(
        os.path.dirname(__file__),
        'stretch_driver_alternate.py',
    )
    stretch_driver = ExecuteProcess(
        cmd=[
            'python3',
            alternate_driver_path,
            '--ros-args',
            '-r',
            'joint_states:=/stretch/joint_states',
            '-r',
            'cmd_vel:=/stretch/cmd_vel',
            '-p',
            f'rate:={args.rate}',
            '-p',
            f'timeout:={args.timeout}',
            '-p',
            f'controller_calibration_file:={controller_calibration_file}',
            '-p',
            f'broadcast_odom_tf:={args.broadcast_odom_tf}',
            '-p',
            f'fail_out_of_range_goal:={args.fail_out_of_range_goal}',
            '-p',
            f'mode:={args.mode}',
        ],
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(stretch_driver)

    return ld


def main():
    parser = argparse.ArgumentParser(
        description='Run Stretch driver alternate with state publishers.'
    )
    parser.add_argument('--rate', type=float, default=30.0)
    parser.add_argument('--timeout', type=float, default=0.5)
    parser.add_argument('--controller-calibration-file', default='')
    parser.add_argument('--broadcast-odom-tf', choices=['True', 'False'], default='False')
    parser.add_argument('--fail-out-of-range-goal', choices=['True', 'False'], default='False')
    parser.add_argument(
        '--mode',
        choices=['position', 'navigation', 'trajectory', 'gamepad'],
        default='position',
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    ld = generate_launch_description(args)
    ls = LaunchService(argv=[])
    ls.include_launch_description(ld)
    return ls.run()


if __name__ == '__main__':
    sys.exit(main())
