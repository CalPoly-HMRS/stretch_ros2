"""
Setup notes (commands run on the robot):
- Install IK dependencies:
    `pip install --user ikpy urdfpy`
- urdfpy pins an old networkx, which breaks on Python 3.10. Override with:
    `pip install --user "networkx>=2.8,<3"`
- urdfpy 0.0.22 still uses `np.float` (removed in NumPy >= 1.24), so we
    define it manually below to keep URDF parsing working.
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path

import ikpy.chain
import ikpy.utils.geometry
import numpy as np
import stretch_body.hello_utils as hu

# Compatibility shim for urdfpy on NumPy >= 1.24.
if not hasattr(np, "float"):
    np.float = float

# dont ask...

import urdfpy


LINKS_TO_REMOVE = [
    "link_right_wheel",
    "link_left_wheel",
    "caster_link",
    "link_gripper_finger_left",
    "link_gripper_fingertip_left",
    "link_gripper_finger_right",
    "link_gripper_fingertip_right",
    "link_head",
    "link_head_pan",
    "link_head_tilt",
    "link_aruco_right_base",
    "link_aruco_left_base",
    "link_aruco_shoulder",
    "link_aruco_top_wrist",
    "link_aruco_inner_wrist",
    "camera_bottom_screw_frame",
    "camera_link",
    "camera_depth_frame",
    "camera_depth_optical_frame",
    "camera_infra1_frame",
    "camera_infra1_optical_frame",
    "camera_infra2_frame",
    "camera_infra2_optical_frame",
    "camera_color_frame",
    "camera_color_optical_frame",
    "camera_accel_frame",
    "camera_accel_optical_frame",
    "camera_gyro_frame",
    "camera_gyro_optical_frame",
    "laser",
    "base_imu",
    "respeaker_base",
]

JOINTS_TO_REMOVE = [
    "joint_right_wheel",
    "joint_left_wheel",
    "caster_joint",
    "joint_gripper_finger_left",
    "joint_gripper_fingertip_left",
    "joint_gripper_finger_right",
    "joint_gripper_fingertip_right",
    "joint_head",
    "joint_head_pan",
    "joint_head_tilt",
    "joint_aruco_right_base",
    "joint_aruco_left_base",
    "joint_aruco_shoulder",
    "joint_aruco_top_wrist",
    "joint_aruco_inner_wrist",
    "camera_joint",
    "camera_link_joint",
    "camera_depth_joint",
    "camera_depth_optical_joint",
    "camera_infra1_joint",
    "camera_infra1_optical_joint",
    "camera_infra2_joint",
    "camera_infra2_optical_joint",
    "camera_color_joint",
    "camera_color_optical_joint",
    "camera_accel_joint",
    "camera_accel_optical_joint",
    "camera_gyro_joint",
    "camera_gyro_optical_joint",
    "joint_laser",
    "joint_base_imu",
    "joint_respeaker",
]


class StretchIkRos:
    def __init__(
        self,
        hello_node,
        cartesian_urdf_path=None,
        urdf_path=None,
        tool_name=None,
    ):
        """Initialize IK utilities backed by a HelloNode-like ROS2 node.

        Args:
            hello_node: A node instance that exposes `joint_state`,
                `set_joint_poses()`, and `set_joint_velocities()`.
            cartesian_urdf_path: Output path for the rewritten cartesian URDF.
                Defaults to /tmp/iktutorial/stretch.urdf.
            urdf_path: Input Stretch URDF path. Defaults to stretch_user export.
            tool_name: Optional fixed tool name (e.g., tool_stretch_gripper).
        """
        # Expect a HelloNode (or subclass) that provides joint_state and set_joint_poses().
        self.node = hello_node
        self.tool_name = tool_name
        self.cartesian_urdf_path = cartesian_urdf_path or "/tmp/iktutorial/stretch.urdf"
        self.urdf_path = urdf_path or self._get_urdf_path()
        # Build a cartesian URDF and IKPy chain once at startup.
        self.chain = self._make_chain(self.urdf_path, self.cartesian_urdf_path)
        # Defaults for velocity control; override per call as needed.
        self.default_rate_hz = 30.0
        self.default_max_velocity = 20.0
        self.default_max_accel = 20.0

    @staticmethod
    def _get_urdf_path():
        """Return the default Stretch URDF path from stretch_user.

        Returns:
            Absolute path to stretch.urdf in the stretch_user directory.
        """
        fleet_dir = Path(hu.get_fleet_directory())
        return str((fleet_dir / "exported_urdf" / "stretch.urdf").absolute())

    @staticmethod
    def _make_chain(urdf_path, cartesian_urdf_path):
        """Create an IKPy chain after rewriting the URDF for IK.

        Args:
            urdf_path: Original Stretch URDF path.
            cartesian_urdf_path: Output path for the modified URDF.

        Returns:
            An IKPy Chain built from the rewritten URDF.
        """
        # IKPy needs ranged joints, so we rewrite the URDF before building the chain.
        StretchIkRos._build_cartesian_urdf(urdf_path, cartesian_urdf_path)
        return ikpy.chain.Chain.from_urdf_file(cartesian_urdf_path)

    @staticmethod
    def _build_cartesian_urdf(urdf_path, out_path):
        """Rewrite the URDF to a single cartesian chain with ranged joints.

        This removes unrelated branches, adds a virtual base translation joint,
        adds a virtual base rotation joint, and re-parents the mast so IKPy sees
        a linear chain.

        Args:
            urdf_path: Original Stretch URDF path.
            out_path: Destination path for the modified URDF.
        """
        original_urdf = urdfpy.URDF.load(urdf_path)
        modified_urdf = original_urdf.copy()

        links_to_remove = [
            link for link in modified_urdf._links if link.name in LINKS_TO_REMOVE
        ]
        for link in links_to_remove:
            modified_urdf._links.remove(link)

        joints_to_remove = [
            joint for joint in modified_urdf._joints if joint.name in JOINTS_TO_REMOVE
        ]
        for joint in joints_to_remove:
            modified_urdf._joints.remove(joint)

        # Add virtual base yaw and translation joints for IK.
        # IKPy does not support continuous wheel rotation joints.
        joint_base_rotation = urdfpy.Joint(
            name="joint_base_rotation",
            parent="base_link",
            child="link_base_rotation",
            joint_type="revolute",
            axis=np.array([0.0, 0.0, 1.0]),
            origin=np.eye(4, dtype=np.float64),
            limit=urdfpy.JointLimit(
                effort=100.0,
                velocity=1.0,
                lower=-np.pi,
                upper=np.pi,
            ),
        )
        modified_urdf._joints.append(joint_base_rotation)

        link_base_rotation = urdfpy.Link(
            name="link_base_rotation",
            inertial=None,
            visuals=None,
            collisions=None,
        )
        modified_urdf._links.append(link_base_rotation)

        joint_base_translation = urdfpy.Joint(
            name="joint_base_translation",
            parent="link_base_rotation",
            child="link_base_translation",
            joint_type="prismatic",
            axis=np.array([1.0, 0.0, 0.0]),
            origin=np.eye(4, dtype=np.float64),
            limit=urdfpy.JointLimit(
                effort=100.0,
                velocity=1.0,
                lower=-1.0,
                upper=1.0,
            ),
        )
        modified_urdf._joints.append(joint_base_translation)

        link_base_translation = urdfpy.Link(
            name="link_base_translation",
            inertial=None,
            visuals=None,
            collisions=None,
        )
        modified_urdf._links.append(link_base_translation)

        # Re-parent the mast to the new base translation link.
        for joint in modified_urdf._joints:
            if joint.name == "joint_mast":
                joint.parent = "link_base_translation"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        modified_urdf.save(out_path)

    def _bound_range(self, name, value):
        """Clamp a value to an IKPy link's bounds.

        Args:
            name: Link/joint name in the chain.
            value: Candidate value to clamp.

        Returns:
            Clamped value within the link's bounds.
        """
        index = self._get_link_index(name)
        bounds = self.chain.links[index].bounds
        return min(max(value, bounds[0]), bounds[1])

    def _resolve_chain_name(self, name):
        """Resolve a name to the actual IKPy chain link name.

        IKPy stores link names, while many callers use joint names. This
        helper maps between joint_*/link_* variants when needed.

        Args:
            name: Link or joint name to resolve.

        Returns:
            Resolved chain link name.
        """
        names = [link.name for link in self.chain.links]
        if name in names:
            return name
        if name.startswith("joint_"):
            alt = "link_" + name[len("joint_"):]
            if alt in names:
                return alt
        if name.startswith("link_"):
            alt = "joint_" + name[len("link_"):]
            if alt in names:
                return alt
        return name

    def _has_link(self, name):
        """Return True if the chain contains a link/joint name."""
        resolved = self._resolve_chain_name(name)
        return resolved in [link.name for link in self.chain.links]

    def _get_link_index(self, name):
        """Return the index of a link/joint name in the IKPy chain.

        Args:
            name: Link/joint name as defined in the chain.

        Returns:
            Integer index into `self.chain.links`.
        """
        resolved = self._resolve_chain_name(name)
        names = [link.name for link in self.chain.links]
        if resolved not in names:
            raise ValueError(f"Unknown joint/link name: {name}")
        return names.index(resolved)

    def _get_q_value(self, q_vec, name):
        """Return a joint value from an IKPy vector by joint name.

        Args:
            q_vec: IKPy joint vector.
            name: Link/joint name in the chain.

        Returns:
            Joint value as a float.
        """
        return q_vec[self._get_link_index(name)]

    def _set_q_value(self, q_vec, name, value):
        """Set a joint value in an IKPy vector by joint name.

        Args:
            q_vec: IKPy joint vector to mutate.
            name: Link/joint name in the chain.
            value: Value to assign.
        """
        q_vec[self._get_link_index(name)] = value

    @contextmanager
    def _temporary_joint_bounds(self, joint_bounds=None, fixed_joints=None):
        """Temporarily override joint bounds and restore after use.

        Args:
            joint_bounds: Dict of {joint_name: (min, max)} to apply.
            fixed_joints: Dict of {joint_name: value} to lock joints.

        Yields:
            None. Use as a context manager to wrap IK calls.
        """
        # Apply temporary bounds/fixed joints for a single IK solve.
        joint_bounds = joint_bounds or {}
        fixed_joints = fixed_joints or {}
        original_bounds = {}

        try:
            for name, bounds in joint_bounds.items():
                index = self._get_link_index(name)
                original_bounds[name] = self.chain.links[index].bounds
                self.chain.links[index].bounds = bounds

            for name, value in fixed_joints.items():
                index = self._get_link_index(name)
                if name not in original_bounds:
                    original_bounds[name] = self.chain.links[index].bounds
                self.chain.links[index].bounds = (value, value)

            yield
        finally:
            for name, bounds in original_bounds.items():
                index = self._get_link_index(name)
                self.chain.links[index].bounds = bounds

    def _get_tool_name(self, tool_name=None):
        """Resolve the active tool name from args, stored defaults, or node.

        Args:
            tool_name: Optional tool name override.

        Returns:
            Tool name string (e.g., tool_stretch_gripper).
        """
        # Prefer explicit tool_name, otherwise try to read it from the node.
        if tool_name:
            return tool_name
        if self.tool_name:
            return self.tool_name
        if hasattr(self.node, "get_tool"):
            tool = self.node.get_tool()
            if isinstance(tool, str) and tool:
                return tool
        return "tool_stretch_gripper"

    def _joint_pos(self, name):
        """Return a joint position from the latest JointState.

        Args:
            name: Joint name in the JointState message.

        Returns:
            Joint position as a float.
        """
        # Look up a single joint position by name in the latest JointState.
        joint_state = self.node.joint_state
        if name not in joint_state.name:
            raise ValueError(f"Joint state missing {name}")
        return joint_state.position[joint_state.name.index(name)]

    def _joint_pos_optional(self, name, default=0.0):
        """Return a joint position if present, otherwise a default.

        Args:
            name: Joint name in the JointState message.
            default: Value to return if the joint is missing.

        Returns:
            Joint position or default value.
        """
        joint_state = self.node.joint_state
        if name not in joint_state.name:
            return default
        return joint_state.position[joint_state.name.index(name)]

    def _get_arm_extension(self):
        """Return total arm extension from telescoping joints or wrist_extension.

        Returns:
            Arm extension in meters.
        """
        # Stretch arm extension can be represented as 4 telescoping joints.
        joint_state = self.node.joint_state
        arm_joints = [
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
        ]
        if all(name in joint_state.name for name in arm_joints):
            return sum(self._joint_pos(name) for name in arm_joints)
        if "wrist_extension" in joint_state.name:
            return self._joint_pos("wrist_extension")
        raise ValueError("Arm joint states not available")

    def get_current_configuration(self, tool_name=None):
        """Return the IKPy configuration vector for the current robot state.

        Args:
            tool_name: Optional tool name override.

        Returns:
            List of joint values aligned with `self.chain.links`.
        """
        # Build an IKPy configuration vector from the current robot state.
        tool_name = self._get_tool_name(tool_name)
        q_vec = [0.0 for _ in self.chain.links]
        q_base_translation = 0.0
        q_base_rotation = 0.0
        if self._has_link("joint_base_translation"):
            q_base_translation = self._bound_range(
                "joint_base_translation",
                self._joint_pos_optional("joint_base_translate", 0.0),
            )
        if self._has_link("joint_base_rotation"):
            q_base_rotation = self._bound_range(
                "joint_base_rotation",
                self._joint_pos_optional("joint_base_rotate", 0.0),
            )
        q_lift = self._bound_range(
            "joint_lift", self._joint_pos("joint_lift")
        )
        # IKPy models a 4-link telescoping arm; divide total extension equally.
        arm_extension = self._get_arm_extension()
        q_arml = self._bound_range("joint_arm_l0", arm_extension / 4.0)
        q_yaw = self._bound_range(
            "joint_wrist_yaw", self._joint_pos("joint_wrist_yaw")
        )

        if self._has_link("joint_base_translation"):
            self._set_q_value(q_vec, "joint_base_translation", q_base_translation)
        if self._has_link("joint_base_rotation"):
            self._set_q_value(q_vec, "joint_base_rotation", q_base_rotation)
        self._set_q_value(q_vec, "joint_lift", q_lift)
        for arm_joint in [
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
        ]:
            self._set_q_value(q_vec, arm_joint, q_arml)
        self._set_q_value(q_vec, "joint_wrist_yaw", q_yaw)

        if tool_name == "tool_stretch_gripper":
            return q_vec

        if tool_name == "tool_stretch_dex_wrist":
            q_pitch = self._bound_range(
                "joint_wrist_pitch", self._joint_pos("joint_wrist_pitch")
            )
            q_roll = self._bound_range(
                "joint_wrist_roll", self._joint_pos("joint_wrist_roll")
            )
            self._set_q_value(q_vec, "joint_wrist_pitch", q_pitch)
            self._set_q_value(q_vec, "joint_wrist_roll", q_roll)
            return q_vec

        raise ValueError(f"Unsupported tool: {tool_name}")

    def solve_point_ik(
        self,
        target_point,
        q_init=None,
        joint_bounds=None,
        fixed_joints=None,
    ):
        """Solve IK for a target point, optionally with bounds and fixed joints.

        Args:
            target_point: [x, y, z] target in base_link frame.
            q_init: Optional initial configuration. Defaults to current state.
            joint_bounds: Dict of {joint_name: (min, max)} for this solve.
            fixed_joints: Dict of {joint_name: value} to hold fixed.

        Returns:
            IKPy solution vector.
        """
        # Solve IK for a 3D point without orientation constraints.
        q_init = q_init or self.get_current_configuration()
        if fixed_joints:
            q_init = list(q_init)
            for name, value in fixed_joints.items():
                q_init[self._get_link_index(name)] = value

        with self._temporary_joint_bounds(
            joint_bounds=joint_bounds,
            fixed_joints=fixed_joints,
        ):
            return self.chain.inverse_kinematics(
                target_point,
                initial_position=q_init,
            )

    def is_point_reachable(
        self,
        target_point,
        tool_name=None,
        max_error=1e-2,
        allow_base_translation=False,
        allow_base_rotation=False,
        joint_bounds=None,
        fixed_joints=None,
    ):
        """Check whether a target point is reachable under constraints.

        Args:
            target_point: [x, y, z] target in base_link frame.
            tool_name: Optional tool name override.
            max_error: Maximum allowed Cartesian error in meters.
            allow_base_translation: If False, keep base translation fixed.
            allow_base_rotation: If False, keep base rotation fixed.
            joint_bounds: Dict of {joint_name: (min, max)} for this solve.
            fixed_joints: Dict of {joint_name: value} to hold fixed.

        Returns:
            Tuple of (is_reachable, position_error, q_solution).
        """
        tool_name = self._get_tool_name(tool_name)
        q_init = self.get_current_configuration(tool_name)
        fixed_joints = dict(fixed_joints or {})

        if not allow_base_translation and self._has_link("joint_base_translation"):
            fixed_joints.setdefault(
                "joint_base_translation",
                self._get_q_value(q_init, "joint_base_translation"),
            )

        if not allow_base_rotation and self._has_link("joint_base_rotation"):
            fixed_joints.setdefault(
                "joint_base_rotation",
                self._get_q_value(q_init, "joint_base_rotation"),
            )

        q_soln = self.solve_point_ik(
            target_point,
            q_init=q_init,
            joint_bounds=joint_bounds,
            fixed_joints=fixed_joints,
        )
        error = self.compute_position_error(q_soln, target_point)
        return error <= max_error, float(error), q_soln

    def make_target_pose(self, target_point, rpy):
        """Build a 4x4 pose matrix from target point and roll/pitch/yaw.

        Args:
            target_point: [x, y, z] position.
            rpy: [roll, pitch, yaw] in radians.

        Returns:
            4x4 numpy pose matrix.
        """
        # Create a 4x4 pose matrix from target position and roll/pitch/yaw.
        pose = np.eye(4)
        pose[:3, :3] = ikpy.utils.geometry.rpy_matrix(*rpy)
        pose[:3, 3] = np.array(target_point)
        return pose

    def solve_pose_ik(
        self,
        target_pose,
        q_init=None,
        pretarget_pose=None,
        joint_bounds=None,
        fixed_joints=None,
    ):
        """Solve IK for a target pose with optional pretarget and constraints.

        Args:
            target_pose: 4x4 pose matrix in base_link frame.
            q_init: Optional initial configuration. Defaults to current state.
            pretarget_pose: Optional intermediate pose to improve convergence.
            joint_bounds: Dict of {joint_name: (min, max)} for this solve.
            fixed_joints: Dict of {joint_name: value} to hold fixed.

        Returns:
            IKPy solution vector.
        """
        # Solve IK for a full pose; optional pretarget helps convergence.
        q_init = q_init or self.get_current_configuration()
        if fixed_joints:
            q_init = list(q_init)
            for name, value in fixed_joints.items():
                q_init[self._get_link_index(name)] = value

        with self._temporary_joint_bounds(
            joint_bounds=joint_bounds,
            fixed_joints=fixed_joints,
        ):
            if pretarget_pose is not None:
                q_init = self.chain.inverse_kinematics_frame(
                    pretarget_pose,
                    orientation_mode="all",
                    initial_position=q_init,
                )
            return self.chain.inverse_kinematics_frame(
                target_pose,
                orientation_mode="all",
                initial_position=q_init,
            )

    def compute_position_error(self, q_soln, target_point):
        """Return Cartesian position error between FK(q) and the target point.

        Args:
            q_soln: IKPy joint vector.
            target_point: [x, y, z] target position.

        Returns:
            Euclidean position error in meters.
        """
        # Measure Cartesian error between FK(q) and the target point.
        return np.linalg.norm(
            self.chain.forward_kinematics(q_soln)[:3, 3] - target_point
        )

    def clamp_solution(self, q_soln, joint_bounds=None, fixed_joints=None):
        """Clamp a solution against bounds and/or fixed joint values.

        Args:
            q_soln: IKPy joint vector.
            joint_bounds: Dict of {joint_name: (min, max)} to clamp.
            fixed_joints: Dict of {joint_name: value} to enforce.

        Returns:
            New list of joint values after clamping.
        """
        # Clamp IK solution to custom bounds and/or fixed joints before execution.
        joint_bounds = joint_bounds or {}
        fixed_joints = fixed_joints or {}
        clamped = list(q_soln)

        for name, value in fixed_joints.items():
            index = self._get_link_index(name)
            clamped[index] = value

        for name, bounds in joint_bounds.items():
            index = self._get_link_index(name)
            clamped[index] = float(np.clip(clamped[index], bounds[0], bounds[1]))

        return clamped

    def move_to_configuration(self, q_soln, tool_name=None):
        """Send a position command using HelloNode.set_joint_poses().

        Args:
            q_soln: IKPy joint vector.
            tool_name: Optional tool name override.
        """
        # Convert IKPy solution into joint commands via HelloNode.set_joint_poses().
        tool_name = self._get_tool_name(tool_name)
        q_base_translation = (
            self._get_q_value(q_soln, "joint_base_translation")
            if self._has_link("joint_base_translation")
            else 0.0
        )
        q_base_rotation = (
            self._get_q_value(q_soln, "joint_base_rotation")
            if self._has_link("joint_base_rotation")
            else 0.0
        )
        q_lift = self._get_q_value(q_soln, "joint_lift")
        q_arm = (
            self._get_q_value(q_soln, "joint_arm_l0")
            + self._get_q_value(q_soln, "joint_arm_l1")
            + self._get_q_value(q_soln, "joint_arm_l2")
            + self._get_q_value(q_soln, "joint_arm_l3")
        )
        q_yaw = self._get_q_value(q_soln, "joint_wrist_yaw")

        joint_poses = [
            ("lift", q_lift),
            ("arm", q_arm),
            ("wrist_yaw", q_yaw),
        ]

        if self._has_link("joint_base_translation"):
            joint_poses.insert(0, ("base_translate", q_base_translation))
        if self._has_link("joint_base_rotation"):
            joint_poses.insert(0, ("base_rotate", q_base_rotation))

        if tool_name == "tool_stretch_dex_wrist":
            joint_poses.extend(
                [
                    ("wrist_pitch", self._get_q_value(q_soln, "joint_wrist_pitch")),
                    ("wrist_roll", self._get_q_value(q_soln, "joint_wrist_roll")),
                ]
            )

        self.node.set_joint_poses(joint_poses)

    def move_to_configuration_velocity(
        self,
        q_soln,
        tool_name=None,
        rate_hz=None,
        max_velocity=None,
        max_accel=None,
        position_tolerance=1e-2,
        timeout_s=5.0,
        control_base_with_pose=True,
    ):
        """Use a simple velocity servo to reach the target configuration.

        Args:
            q_soln: IKPy joint vector.
            tool_name: Optional tool name override.
            rate_hz: Control loop rate (Hz).
            max_velocity: Per-joint velocity limit.
            max_accel: Per-joint acceleration limit.
            position_tolerance: Stop when max joint error is below this.
            timeout_s: Stop after this many seconds if not converged.
            control_base_with_pose: If True, set base_translate by position.
        """
        # Simple velocity servo to reach the target configuration.
        tool_name = self._get_tool_name(tool_name)
        rate_hz = rate_hz or self.default_rate_hz
        max_velocity = max_velocity or self.default_max_velocity
        max_accel = max_accel or self.default_max_accel
        dt = 1.0 / float(rate_hz)

        q_base_translation = (
            self._get_q_value(q_soln, "joint_base_translation")
            if self._has_link("joint_base_translation")
            else 0.0
        )
        q_base_rotation = (
            self._get_q_value(q_soln, "joint_base_rotation")
            if self._has_link("joint_base_rotation")
            else 0.0
        )
        q_lift = self._get_q_value(q_soln, "joint_lift")
        q_arm = (
            self._get_q_value(q_soln, "joint_arm_l0")
            + self._get_q_value(q_soln, "joint_arm_l1")
            + self._get_q_value(q_soln, "joint_arm_l2")
            + self._get_q_value(q_soln, "joint_arm_l3")
        )
        q_yaw = self._get_q_value(q_soln, "joint_wrist_yaw")
        q_pitch = (
            self._get_q_value(q_soln, "joint_wrist_pitch")
            if tool_name == "tool_stretch_dex_wrist"
            else None
        )
        q_roll = (
            self._get_q_value(q_soln, "joint_wrist_roll")
            if tool_name == "tool_stretch_dex_wrist"
            else None
        )

        if control_base_with_pose:
            base_cmds = []
            if self._has_link("joint_base_translation"):
                base_cmds.append(("base_translate", q_base_translation))
            if self._has_link("joint_base_rotation"):
                base_cmds.append(("base_rotate", q_base_rotation))
            if base_cmds:
                self.node.set_joint_poses(base_cmds)

        start_time = time.time()
        prev_velocities = {}

        while True:
            if (time.time() - start_time) > timeout_s:
                break

            current = {
                "lift": self._joint_pos("joint_lift"),
                "arm": self._get_arm_extension(),
                "wrist_yaw": self._joint_pos("joint_wrist_yaw"),
            }
            if tool_name == "tool_stretch_dex_wrist":
                current["wrist_pitch"] = self._joint_pos("joint_wrist_pitch")
                current["wrist_roll"] = self._joint_pos("joint_wrist_roll")

            targets = {
                "lift": q_lift,
                "arm": q_arm,
                "wrist_yaw": q_yaw,
            }
            if tool_name == "tool_stretch_dex_wrist":
                targets["wrist_pitch"] = q_pitch
                targets["wrist_roll"] = q_roll

            errors = {name: targets[name] - current[name] for name in targets}
            max_err = max(abs(err) for err in errors.values())
            if max_err <= position_tolerance:
                break

            joint_vels = []
            for name, err in errors.items():
                vel_cmd = err / max(dt, 1e-6)
                vel_cmd = float(np.clip(vel_cmd, -max_velocity, max_velocity))

                if name in prev_velocities:
                    accel = (vel_cmd - prev_velocities[name]) / max(dt, 1e-6)
                    accel = float(np.clip(accel, -max_accel, max_accel))
                    vel_cmd = prev_velocities[name] + accel * dt

                prev_velocities[name] = vel_cmd
                joint_vels.append((name, vel_cmd))

            self.node.set_joint_velocities(joint_vels, duration=dt)
            time.sleep(dt)

# Alternatively, a proportional velocity controller that scales with position error
    def move_to_configuration_velocity_p(
        self,
        q_soln,
        tool_name=None,
        rate_hz=None,
        max_velocity=None,
        max_accel=None,
        position_tolerance=1e-2,
        timeout_s=5.0,
        control_base_with_pose=True,
        kp=2.0,
    ):
        """Use a proportional velocity controller to reach the target configuration.

        Args:
            q_soln: IKPy joint vector.
            tool_name: Optional tool name override.
            rate_hz: Control loop rate (Hz).
            max_velocity: Per-joint velocity limit.
            max_accel: Per-joint acceleration limit.
            position_tolerance: Stop when max joint error is below this.
            timeout_s: Stop after this many seconds if not converged.
            control_base_with_pose: If True, set base_translate by position.
            kp: Proportional gain applied to position error.
        """
        # Proportional velocity servo; velocities scale with position error.
        tool_name = self._get_tool_name(tool_name)
        rate_hz = rate_hz or self.default_rate_hz
        max_velocity = max_velocity or self.default_max_velocity
        max_accel = max_accel or self.default_max_accel
        dt = 1.0 / float(rate_hz)

        q_base_translation = (
            self._get_q_value(q_soln, "joint_base_translation")
            if self._has_link("joint_base_translation")
            else 0.0
        )
        q_base_rotation = (
            self._get_q_value(q_soln, "joint_base_rotation")
            if self._has_link("joint_base_rotation")
            else 0.0
        )
        q_lift = self._get_q_value(q_soln, "joint_lift")
        q_arm = (
            self._get_q_value(q_soln, "joint_arm_l0")
            + self._get_q_value(q_soln, "joint_arm_l1")
            + self._get_q_value(q_soln, "joint_arm_l2")
            + self._get_q_value(q_soln, "joint_arm_l3")
        )
        q_yaw = self._get_q_value(q_soln, "joint_wrist_yaw")
        q_pitch = (
            self._get_q_value(q_soln, "joint_wrist_pitch")
            if tool_name == "tool_stretch_dex_wrist"
            else None
        )
        q_roll = (
            self._get_q_value(q_soln, "joint_wrist_roll")
            if tool_name == "tool_stretch_dex_wrist"
            else None
        )

        if control_base_with_pose:
            base_cmds = []
            if self._has_link("joint_base_translation"):
                base_cmds.append(("base_translate", q_base_translation))
            if self._has_link("joint_base_rotation"):
                base_cmds.append(("base_rotate", q_base_rotation))
            if base_cmds:
                self.node.set_joint_poses(base_cmds)

        start_time = time.time()
        prev_velocities = {}

        while True:
            if (time.time() - start_time) > timeout_s:
                break

            current = {
                "lift": self._joint_pos("joint_lift"),
                "arm": self._get_arm_extension(),
                "wrist_yaw": self._joint_pos("joint_wrist_yaw"),
            }
            if tool_name == "tool_stretch_dex_wrist":
                current["wrist_pitch"] = self._joint_pos("joint_wrist_pitch")
                current["wrist_roll"] = self._joint_pos("joint_wrist_roll")

            targets = {
                "lift": q_lift,
                "arm": q_arm,
                "wrist_yaw": q_yaw,
            }
            if tool_name == "tool_stretch_dex_wrist":
                targets["wrist_pitch"] = q_pitch
                targets["wrist_roll"] = q_roll

            errors = {name: targets[name] - current[name] for name in targets}
            max_err = max(abs(err) for err in errors.values())
            if max_err <= position_tolerance:
                break

            joint_vels = []
            for name, err in errors.items():
                vel_cmd = kp * err
                vel_cmd = float(np.clip(vel_cmd, -max_velocity, max_velocity))

                if name in prev_velocities:
                    accel = (vel_cmd - prev_velocities[name]) / max(dt, 1e-6)
                    accel = float(np.clip(accel, -max_accel, max_accel))
                    vel_cmd = prev_velocities[name] + accel * dt

                prev_velocities[name] = vel_cmd
                joint_vels.append((name, vel_cmd))

            self.node.set_joint_velocities(joint_vels, duration=dt)
            time.sleep(dt)
