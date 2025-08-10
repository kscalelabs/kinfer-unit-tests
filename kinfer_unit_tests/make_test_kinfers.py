"""Utility to make kinfer runtimes that test the functionality of the K-Bot."""

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import colorlogging
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

logger = logging.getLogger(__name__)


SIM_DT = 0.02
NUM_COMMANDS = 6  # placeholder for tests

# Joint biases, these are in the order that they appear in the neural network.
JOINT_BIASES: list[tuple[str, float, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0, 1.0),  # 0
    ("dof_right_shoulder_roll_03", math.radians(-10.0), 1.0),  # 1
    ("dof_right_shoulder_yaw_02", 0.0, 1.0),  # 2
    ("dof_right_elbow_02", math.radians(90.0), 1.0),  # 3
    ("dof_right_wrist_00", 0.0, 1.0),  # 4
    ("dof_left_shoulder_pitch_03", 0.0, 1.0),  # 5
    ("dof_left_shoulder_roll_03", math.radians(10.0), 1.0),  # 6
    ("dof_left_shoulder_yaw_02", 0.0, 1.0),  # 7
    ("dof_left_elbow_02", math.radians(-90.0), 1.0),  # 8
    ("dof_left_wrist_00", 0.0, 1.0),  # 9
    ("dof_right_hip_pitch_04", math.radians(-20.0), 0.01),  # 10
    ("dof_right_hip_roll_03", math.radians(-0.0), 2.0),  # 11
    ("dof_right_hip_yaw_03", 0.0, 2.0),  # 12
    ("dof_right_knee_04", math.radians(-50.0), 0.01),  # 13
    ("dof_right_ankle_02", math.radians(30.0), 1.0),  # 14
    ("dof_left_hip_pitch_04", math.radians(20.0), 0.01),  # 15
    ("dof_left_hip_roll_03", math.radians(0.0), 2.0),  # 16
    ("dof_left_hip_yaw_03", 0.0, 2.0),  # 17
    ("dof_left_knee_04", math.radians(50.0), 0.01),  # 18
    ("dof_left_ankle_02", math.radians(-30.0), 1.0),  # 19
]

JOINT_INVERSIONS: list[tuple[str, int]] = [
    ("dof_right_shoulder_pitch_03", 1),  # 0
    ("dof_right_shoulder_roll_03", -1),  # 1
    ("dof_right_shoulder_yaw_02", 1),  # 2
    ("dof_right_elbow_02", -1),  # 3
    ("dof_right_wrist_00", 1),  # 4
    ("dof_left_shoulder_pitch_03", 1),  # 5
    ("dof_left_shoulder_roll_03", 1),  # 6
    ("dof_left_shoulder_yaw_02", 1),  # 7
    ("dof_left_elbow_02", 1),  # 8
    ("dof_left_wrist_00", 1),  # 9
    ("dof_right_hip_pitch_04", -1),  # 10
    ("dof_right_hip_roll_03", 1),  # 11
    ("dof_right_hip_yaw_03", 1),  # 12
    ("dof_right_knee_04", -1),  # 13
    ("dof_right_ankle_02", -1),  # 14
    ("dof_left_hip_pitch_04", 1),  # 15
    ("dof_left_hip_roll_03", 1),  # 16
    ("dof_left_hip_yaw_03", 1),  # 17
    ("dof_left_knee_04", 1),  # 18
    ("dof_left_ankle_02", 1),  # 19
]


InitFn = Callable[[], Array]

StepFn = Callable[
    [Array, Array, Array, Array, Array, Array],  # state inputs
    tuple[Array, Array],  # (targets, carry)
]


@dataclass
class Recipe:
    name: str
    init_fn: InitFn
    step_fn: StepFn
    num_commands: int
    carry_size: tuple[int, ...]


def get_mujoco_model() -> mujoco.MjModel:
    """Get the MuJoCo model for the K-Bot."""
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")


def get_joint_names() -> list[str]:
    """Get the joint names."""
    model = get_mujoco_model()
    return ksim.get_joint_names_in_order(model)[1:]  # drop root joint


def make_zero_recipe(num_joints: int, dt: float) -> Recipe:
    """Sends zeros to all the joints."""
    carry_size = (1,)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        return jnp.zeros(num_joints), jnp.array([t])

    return Recipe("kbot_zero_position", init_fn, step_fn, NUM_COMMANDS, carry_size)


def get_bias_vector(joint_names: list[str]) -> jnp.ndarray:
    """Return an array of neutral/bias angles ordered like `joint_names`."""
    bias_map = {name: bias for name, bias, _ in JOINT_BIASES}
    return jnp.array([bias_map[name] for name in joint_names])


def get_inversion_vector(joint_names: list[str]) -> jnp.ndarray:
    """Return an array of inversion factors (-1 or 1) ordered like `joint_names`."""
    inversion_map = {name: inversion for name, inversion in JOINT_INVERSIONS}
    return jnp.array([inversion_map[name] for name in joint_names])


def make_bias_recipe(joint_names: list[str], dt: float) -> Recipe:
    """Sends the bias values to all the joints."""
    bias_vec = get_bias_vector(joint_names)
    carry_size = (1,)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        return bias_vec, jnp.array([t])

    return Recipe("kbot_bias_position", init_fn, step_fn, NUM_COMMANDS, carry_size)


# (amplitude [rad], frequency [Hz]) for each joint name
JOINT_SINE_PARAMS: dict[str, tuple[float, float]] = {name: (0.15, 0.6) for name, *_ in JOINT_BIASES}


def make_sine_recipe(joint_names: list[str], dt: float) -> Recipe:
    """Bias pose ± small sinusoid on every joint (no mirroring tricks)."""
    bias_vec = get_bias_vector(joint_names)
    amps = jnp.array([JOINT_SINE_PARAMS[n][0] for n in joint_names])
    freqs = jnp.array([JOINT_SINE_PARAMS[n][1] for n in joint_names])
    carry_size = (1,)

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        offsets = amps * jnp.sin(2 * jnp.pi * freqs * t)
        return bias_vec + offsets, jnp.array([t])

    return Recipe("kbot_sine_motion", init_fn, step_fn, NUM_COMMANDS, carry_size)


# Add after JOINT_SINE_PARAMS definition around line 166

# Step parameters for each joint name
JOINT_STEP_PARAMS: dict[str, tuple[float, float, float]] = {
    # (step_size [rad], max_deviation [rad], step_hold_time [s])
    # Arms - smaller, faster steps
    "dof_right_shoulder_pitch_03": (0.05, 0.08, 0.02),
    "dof_right_shoulder_roll_03": (0.08, 0.12, 0.03),
    "dof_right_shoulder_yaw_02": (0.06, 0.10, 0.025),
    "dof_right_elbow_02": (0.10, 0.15, 0.04),
    "dof_right_wrist_00": (0.08, 0.12, 0.02),
    "dof_left_shoulder_pitch_03": (0.05, 0.08, 0.02),
    "dof_left_shoulder_roll_03": (0.08, 0.12, 0.03),
    "dof_left_shoulder_yaw_02": (0.06, 0.10, 0.025),
    "dof_left_elbow_02": (0.10, 0.15, 0.04),
    "dof_left_wrist_00": (0.08, 0.12, 0.02),
    # Legs - larger, slower steps (more conservative)
    "dof_right_hip_pitch_04": (0.03, 0.05, 0.08),
    "dof_right_hip_roll_03": (0.04, 0.06, 0.06),
    "dof_right_hip_yaw_03": (0.05, 0.08, 0.07),
    "dof_right_knee_04": (0.02, 0.04, 0.10),
    "dof_right_ankle_02": (0.06, 0.10, 0.05),
    "dof_left_hip_pitch_04": (0.03, 0.05, 0.08),
    "dof_left_hip_roll_03": (0.04, 0.06, 0.06),
    "dof_left_hip_yaw_03": (0.05, 0.08, 0.07),
    "dof_left_knee_04": (0.02, 0.04, 0.10),
    "dof_left_ankle_02": (0.06, 0.10, 0.05),
}


def get_motion_limits(joint_names: list[str]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return arrays of min and max motion offsets for joints, ordered like `joint_names`."""
    min_offsets = []
    max_offsets = []
    for name in joint_names:
        if name in JOINT_MOTION_LIMITS:
            min_offset, max_offset = JOINT_MOTION_LIMITS[name]
            min_offsets.append(min_offset)
            max_offsets.append(max_offset)
        else:
            # Default conservative limits if not specified
            min_offsets.append(math.radians(-15.0))
            max_offsets.append(math.radians(15.0))
    return jnp.array(min_offsets), jnp.array(max_offsets)


def make_step_recipe(joint_names: list[str], dt: float, seed: int = 42) -> Recipe:
    """Bias pose with random step commands that don't venture too far from bias."""
    num_joints = len(joint_names)
    bias_vec = get_bias_vector(joint_names)
    step_sizes = jnp.array([JOINT_STEP_PARAMS[n][0] for n in joint_names])
    max_deviations = jnp.array([JOINT_STEP_PARAMS[n][1] for n in joint_names])
    step_hold_times = jnp.array([JOINT_STEP_PARAMS[n][2] for n in joint_names])

    # carry: [time, random_state, next_step_times..., current_targets...]
    carry_size = (2 + 2 * num_joints,)

    @jax.jit
    def init_fn() -> Array:
        # Initialize carry: [time=0, random_state=seed, next_step_times..., targets...]
        return jnp.concatenate(
            [
                jnp.array([0.0, float(seed)]),
                step_hold_times,  # initial next step times
                bias_vec,  # initial targets
            ]
        )

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        random_state = carry[1]
        next_step_times = carry[2 : 2 + num_joints]
        current_targets = carry[2 + num_joints :]

        # Check which joints should step
        should_step = t >= next_step_times

        # Simple pseudo-random number generation
        pseudo_random = jnp.sin(random_state + t * 1000.0) * 10000.0
        pseudo_random = pseudo_random - jnp.floor(pseudo_random)

        # Generate step direction for each joint
        random_vals = jnp.array([jnp.sin(pseudo_random * 1000.0 + i * 7.0) for i in range(num_joints)])
        random_directions = jnp.sign(random_vals)

        # Generate step sizes (0.5 to 1.0 of max step size)
        random_scales = 0.5 + 0.5 * jnp.abs(jnp.sin(pseudo_random * 2000.0 + jnp.arange(num_joints) * 13.0))
        step_offsets = random_directions * step_sizes * random_scales

        # Calculate potential new targets
        potential_targets = current_targets + step_offsets

        # Clamp to stay within max_deviation from bias
        min_targets = bias_vec - max_deviations
        max_targets = bias_vec + max_deviations
        clamped_targets = jnp.clip(potential_targets, min_targets, max_targets)

        # Update targets only for joints that should step
        new_targets = jnp.where(should_step, clamped_targets, current_targets)

        # Update next step times only for joints that stepped
        new_next_step_times = jnp.where(should_step, t + step_hold_times, next_step_times)
        new_random_state = random_state + 1.0

        # Build new carry
        new_carry = jnp.concatenate([jnp.array([t, new_random_state]), new_next_step_times, new_targets])

        return new_targets, new_carry

    return Recipe("kbot_step_motion", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_step_echo_recipe(joint_names: list[str], dt: float, seed: int = 42) -> Recipe:
    """Left-side joints get random step commands, right-side joints echo observed left positions."""
    num_joints = len(joint_names)
    bias_vec = get_bias_vector(joint_names)
    inversion_vec = get_inversion_vector(joint_names)
    lr_pairs = get_left_right_pairs(joint_names)

    left_idx, right_idx = zip(*lr_pairs)
    left_idx_arr = jnp.array(left_idx, dtype=jnp.int32)  # (J/2)
    right_idx_arr = jnp.array(right_idx, dtype=jnp.int32)  # (J/2)

    # Get step parameters only for left joints
    left_step_sizes = jnp.array([JOINT_STEP_PARAMS[joint_names[i]][0] for i in left_idx])
    left_max_deviations = jnp.array([JOINT_STEP_PARAMS[joint_names[i]][1] for i in left_idx])
    left_step_hold_times = jnp.array([JOINT_STEP_PARAMS[joint_names[i]][2] for i in left_idx])

    idxs = jnp.arange(num_joints)  # (J,)
    num_left_joints = len(left_idx)

    # carry: [time, random_state, next_step_times_left..., current_left_targets...]
    carry_size = (2 + 2 * num_left_joints,)

    @jax.jit
    def init_fn() -> Array:
        # Initialize carry: [time=0, random_state=seed, next_step_times_left..., left_targets...]
        left_bias = jnp.array([bias_vec[i] for i in left_idx])
        return jnp.concatenate(
            [
                jnp.array([0.0, float(seed)]),
                left_step_hold_times,  # initial next step times for left joints
                left_bias,  # initial targets for left joints
            ]
        )

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        random_state = carry[1]
        next_step_times_left = carry[2 : 2 + num_left_joints]
        current_left_targets = carry[2 + num_left_joints :]

        # === LEFT SIDE STEP LOGIC ===
        # Check which left joints should step
        should_step_left = t >= next_step_times_left

        # Simple pseudo-random number generation
        pseudo_random = jnp.sin(random_state + t * 1000.0) * 10000.0
        pseudo_random = pseudo_random - jnp.floor(pseudo_random)

        # Generate step direction for each left joint
        random_vals = jnp.array([jnp.sin(pseudo_random * 1000.0 + i * 7.0) for i in range(num_left_joints)])
        random_directions = jnp.sign(random_vals)

        # Generate step sizes (0.5 to 1.0 of max step size)
        random_scales = 0.5 + 0.5 * jnp.abs(jnp.sin(pseudo_random * 2000.0 + jnp.arange(num_left_joints) * 13.0))
        step_offsets = random_directions * left_step_sizes * random_scales

        # Calculate potential new targets for left joints
        potential_left_targets = current_left_targets + step_offsets

        # Clamp to stay within max_deviation from bias for left joints
        left_bias = jnp.array([bias_vec[i] for i in left_idx])
        min_left_targets = left_bias - left_max_deviations
        max_left_targets = left_bias + left_max_deviations
        clamped_left_targets = jnp.clip(potential_left_targets, min_left_targets, max_left_targets)

        # Update left targets only for joints that should step
        new_left_targets = jnp.where(should_step_left, clamped_left_targets, current_left_targets)

        # Update next step times only for left joints that stepped
        new_next_step_times_left = jnp.where(should_step_left, t + left_step_hold_times, next_step_times_left)
        new_random_state = random_state + 1.0

        # === COMBINE LEFT COMMANDS WITH RIGHT ECHO ===

        # Create full J-length vector of left targets (0.0 for right joints)
        full_left_targets = jnp.sum(
            jnp.where(
                idxs[:, None] == left_idx_arr,
                new_left_targets[None, :],
                0.0,
            ),
            axis=1,
        )

        # Get the current left angles to echo to right side
        left_angles_now = joint_angles[left_idx_arr]
        right_targets = jnp.sum(
            jnp.where(
                idxs[:, None] == right_idx_arr,
                left_angles_now[None, :],
                0.0,
            ),
            axis=1,
        )

        # Start with left targets, then overwrite right joints with echo targets
        targets = full_left_targets
        right_mask = jnp.sum(idxs[:, None] == right_idx_arr, axis=1).astype(bool)
        targets = jnp.where(right_mask, right_targets, targets)

        # Apply joint inversions (some joints rotate counter-clockwise)
        targets = targets * inversion_vec

        # Build new carry
        new_carry = jnp.concatenate([jnp.array([t, new_random_state]), new_next_step_times_left, new_left_targets])

        return targets, new_carry

    return Recipe("kbot_step_echo_test", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_single_joint_linear_recipe(
    target_joint_name: str, start_pos: float, end_pos: float, num_steps: int, joint_names: list[str], dt: float
) -> Recipe:
    """Moves a single joint linearly from its current position to a target position."""
    bias_vec = get_bias_vector(joint_names)
    target_joint_idx = joint_names.index(target_joint_name)
    carry_size = (1,)  # [time]
    total_duration = num_steps * dt  # Total time for the motion

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate linear interpolation based on elapsed time
        progress = jnp.minimum(t / total_duration, 1.0)
        target_pos = start_pos + (end_pos - start_pos) * progress

        # Set all joints to bias except target joint
        targets = bias_vec.at[target_joint_idx].set(target_pos)

        return targets, jnp.array([t])

    return Recipe(
        f"kbot_linear_{target_joint_name}_{start_pos}_to_{end_pos}", init_fn, step_fn, NUM_COMMANDS, carry_size
    )


def get_left_right_pairs(joint_names: list[str]) -> list[tuple[int, int]]:
    """Return (left_idx, right_idx) pairs for joints that share the same name.

    Ex: ('dof_left_elbow_02', 'dof_right_elbow_02').
    """
    pairs: list[tuple[int, int]] = []
    for i_left, name in enumerate(joint_names):
        if "dof_left_" in name:
            right_name = name.replace("dof_left_", "dof_right_")
            if right_name in joint_names:
                i_right = joint_names.index(right_name)
                pairs.append((i_left, i_right))
    return pairs


def make_echo_recipe(joint_names: list[str], dt: float) -> Recipe:
    """Left-side joints run a small sine wave.

    Right-side joints copy (echo) the observed pose of their left counterparts.
    """
    num_joints = len(joint_names)
    bias_vec = get_bias_vector(joint_names)
    inversion_vec = get_inversion_vector(joint_names)
    lr_pairs = get_left_right_pairs(joint_names)

    left_idx, right_idx = zip(*lr_pairs)
    left_idx_arr = jnp.array(left_idx, dtype=jnp.int32)  # (J/2)
    right_idx_arr = jnp.array(right_idx, dtype=jnp.int32)  # (J/2)

    amps = jnp.array([JOINT_SINE_PARAMS[joint_names[i]][0] for i in left_idx])  # (J/2)
    freqs = jnp.array([JOINT_SINE_PARAMS[joint_names[i]][1] for i in left_idx])  # (J/2)

    idxs = jnp.arange(num_joints)  # (J,)
    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        left_offsets = amps * jnp.sin(2 * jnp.pi * freqs * t)  # (J/2)

        # Make a J-length vector of offsets for the left limbs.
        # i.e. Left offset if it's a left limb, 0.0 otherwise.
        full_left_offsets = jnp.sum(  # (J,)
            jnp.where(
                idxs[:, None] == left_idx_arr,
                left_offsets[None, :],
                0.0,
            ),
            axis=1,
        )

        # Get the current left angles to echo
        left_angles_now = joint_angles[left_idx_arr]
        right_targets = jnp.sum(
            jnp.where(
                idxs[:, None] == right_idx_arr,
                left_angles_now[None, :],
                0.0,
            ),
            axis=1,
        )

        # Sine for left limbs
        targets = bias_vec + full_left_offsets
        # Echo for right limbs
        right_mask = jnp.sum(idxs[:, None] == right_idx_arr, axis=1).astype(bool)
        targets = jnp.where(right_mask, right_targets, targets)

        # Apply joint inversions (some joints rotate counter-clockwise)
        targets = targets * inversion_vec

        return targets, jnp.array([t])

    return Recipe("kbot_echo_test", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_imu_arm_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially moves joints 11, 12, 13 from 0 to π/2 to 0, looping forever."""
    # Target joint names for IDs 11, 12, 13
    target_joint_names = [
        "dof_left_shoulder_pitch_03",  # ID 11
        "dof_left_shoulder_roll_03",  # ID 12 - changed to shoulder roll
        "dof_left_shoulder_yaw_02",  # ID 14
    ]

    # Get indices for these joints - convert to JAX array for JAX-compatible indexing
    target_indices = jnp.array([joint_names.index(name) for name in target_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Motion parameters
    phase_duration = motion_duration  # Use the passed motion_duration for each phase (0->π/2 or π/2->0)
    target_amplitude = math.pi / 2  # π/2 radians

    # Get the shoulder roll bias position (math.radians(10.0))
    shoulder_roll_bias = math.radians(10.0)

    # Carry state: [time, current_joint_idx, phase, phase_start_time]
    # - current_joint_idx: 0, 1, or 2 (which of the 3 joints is moving)
    # - phase: 0 for (0->π/2), 1 for (π/2->0)
    carry_size = (4,)

    @jax.jit
    def init_fn() -> Array:
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        projected_gravity: Array,
        gyroscope: Array,
        accelerometer: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        current_joint_idx = carry[1]
        phase = carry[2]
        phase_start_time = carry[3]

        # Check if current phase is complete
        phase_elapsed = t - phase_start_time
        phase_complete = phase_elapsed >= phase_duration

        # Update phase and joint if needed
        new_joint_idx, new_phase, new_phase_start_time = jax.lax.cond(
            phase_complete,
            lambda: jax.lax.cond(
                phase == 0.0,  # Currently in phase 0 (0->π/2)
                lambda: (current_joint_idx, 1.0, t),  # Move to phase 1 (π/2->0)
                lambda: jax.lax.cond(
                    current_joint_idx < 2.0,  # Not at last joint
                    lambda: (current_joint_idx + 1.0, 0.0, t),  # Next joint, phase 0
                    lambda: (0.0, 0.0, t),  # Loop back to first joint, phase 0
                ),
            ),
            lambda: (current_joint_idx, phase, phase_start_time),  # No change
        )

        # Calculate progress within current phase (0.0 to 1.0)
        current_phase_elapsed = t - new_phase_start_time
        progress = jnp.minimum(current_phase_elapsed / phase_duration, 1.0)

        # Use JAX indexing to get the active joint index
        active_joint_idx = target_indices[jnp.int32(new_joint_idx)]

        # Calculate target position for the active joint
        # Special handling for shoulder roll (index 1) to start from bias position
        base_position = jax.lax.cond(
            new_joint_idx == 1.0,  # If it's the shoulder roll joint
            lambda: shoulder_roll_bias,  # Start from bias position
            lambda: 0.0,  # Other joints start from 0
        )

        active_joint_target = jax.lax.cond(
            new_phase == 0.0,  # Phase 0: base -> base + π/2
            lambda: base_position + progress * target_amplitude,
            lambda: base_position + target_amplitude * (1.0 - progress),  # Phase 1: base + π/2 -> base
        )

        # Set all joints to bias, then update the active joint
        targets = bias_vec
        targets = targets.at[active_joint_idx].set(active_joint_target)

        new_carry = jnp.array([t, new_joint_idx, new_phase, new_phase_start_time])

        return targets, new_carry

    return Recipe("kbot_imu_arm", init_fn, step_fn, NUM_COMMANDS, carry_size)


# Global safety margin for joint limits (0.8 = 80% of hardware limits)
JOINT_SAFETY_MARGIN = 0.8

# Joint motion limits based on actual hardware limits from kbot_metadata.json
# Format: (min_offset_from_bias, max_offset_from_bias) in radians
# Each entry: (hardware_min - bias, hardware_max - bias) * JOINT_SAFETY_MARGIN
JOINT_MOTION_LIMITS: dict[str, tuple[float, float]] = {
    # Right Arm
    "dof_right_shoulder_pitch_03": (math.radians(-180 * JOINT_SAFETY_MARGIN), math.radians(80 * JOINT_SAFETY_MARGIN)),
    "dof_right_shoulder_roll_03": (
        math.radians((-95 - (-10)) * JOINT_SAFETY_MARGIN),
        math.radians((20 - (-10)) * JOINT_SAFETY_MARGIN),
    ),
    "dof_right_shoulder_yaw_02": (math.radians(-95 * JOINT_SAFETY_MARGIN), math.radians(95 * JOINT_SAFETY_MARGIN)),
    "dof_right_elbow_02": (
        math.radians((0 - 90) * JOINT_SAFETY_MARGIN),
        math.radians((142 - 90) * JOINT_SAFETY_MARGIN),
    ),
    "dof_right_wrist_00": (math.radians(-100 * JOINT_SAFETY_MARGIN), math.radians(100 * JOINT_SAFETY_MARGIN)),
    # Left Arm
    "dof_left_shoulder_pitch_03": (math.radians(-80 * JOINT_SAFETY_MARGIN), math.radians(180 * JOINT_SAFETY_MARGIN)),
    "dof_left_shoulder_roll_03": (
        math.radians((-20 - 10) * JOINT_SAFETY_MARGIN),
        math.radians((95 - 10) * JOINT_SAFETY_MARGIN),
    ),
    "dof_left_shoulder_yaw_02": (math.radians(-95 * JOINT_SAFETY_MARGIN), math.radians(95 * JOINT_SAFETY_MARGIN)),
    "dof_left_elbow_02": (
        math.radians((-142 - (-90)) * JOINT_SAFETY_MARGIN),
        math.radians((0 - (-90)) * JOINT_SAFETY_MARGIN),
    ),
    "dof_left_wrist_00": (math.radians(-100 * JOINT_SAFETY_MARGIN), math.radians(100 * JOINT_SAFETY_MARGIN)),
    # Right Leg
    "dof_right_hip_pitch_04": (
        math.radians((-127 - (-20)) * JOINT_SAFETY_MARGIN),
        math.radians((60 - (-20)) * JOINT_SAFETY_MARGIN),
    ),
    "dof_right_hip_roll_03": (
        math.radians((-130 - 0) * JOINT_SAFETY_MARGIN),
        math.radians((12 - 0) * JOINT_SAFETY_MARGIN),
    ),
    "dof_right_hip_yaw_03": (math.radians(-90 * JOINT_SAFETY_MARGIN), math.radians(90 * JOINT_SAFETY_MARGIN)),
    "dof_right_knee_04": (
        math.radians((-155 - (-50)) * JOINT_SAFETY_MARGIN),
        math.radians((0 - (-50)) * JOINT_SAFETY_MARGIN),
    ),
    "dof_right_ankle_02": (
        math.radians((-13 - 30) * JOINT_SAFETY_MARGIN),
        math.radians((72 - 30) * JOINT_SAFETY_MARGIN),
    ),
    # Left Leg
    "dof_left_hip_pitch_04": (
        math.radians((-60 - 20) * JOINT_SAFETY_MARGIN),
        math.radians((127 - 20) * JOINT_SAFETY_MARGIN),
    ),
    "dof_left_hip_roll_03": (
        math.radians((-12 - 0) * JOINT_SAFETY_MARGIN),
        math.radians((130 - 0) * JOINT_SAFETY_MARGIN),
    ),
    "dof_left_hip_yaw_03": (math.radians(-90 * JOINT_SAFETY_MARGIN), math.radians(90 * JOINT_SAFETY_MARGIN)),
    "dof_left_knee_04": (math.radians((0 - 50) * JOINT_SAFETY_MARGIN), math.radians((155 - 50) * JOINT_SAFETY_MARGIN)),
    "dof_left_ankle_02": (
        math.radians((-72 - (-30)) * JOINT_SAFETY_MARGIN),
        math.radians((13 - (-30)) * JOINT_SAFETY_MARGIN),
    ),
}


def _make_arm_sequential_targets(
    joint_names: list[str],
    arm_joint_names: list[str],
    target_indices: jnp.ndarray,
    min_offsets: jnp.ndarray,
    max_offsets: jnp.ndarray,
    bias_vec: jnp.ndarray,
    current_phase: jnp.ndarray,
    progress: jnp.ndarray,
    is_right_arm: bool = False,
) -> jnp.ndarray:
    """Helper function to generate arm sequential targets for left or right arm.

    For is_right_arm:
    - If True, shoulder roll goes to negative limit (right arm away from body)
    - If False, shoulder roll goes to positive limit (left arm away from body)
    """
    targets = bias_vec.copy()

    # Joint indices for easier reference
    shoulder_pitch_idx = target_indices[0]
    shoulder_roll_idx = target_indices[1]
    shoulder_yaw_idx = target_indices[2]
    elbow_idx = target_indices[3]
    wrist_idx = target_indices[4]

    # Get bias positions and limits
    shoulder_pitch_bias = bias_vec[shoulder_pitch_idx]
    shoulder_roll_bias = bias_vec[shoulder_roll_idx]
    shoulder_yaw_bias = bias_vec[shoulder_yaw_idx]
    elbow_bias = bias_vec[elbow_idx]
    wrist_bias = bias_vec[wrist_idx]

    shoulder_pitch_min = shoulder_pitch_bias + min_offsets[0]
    shoulder_pitch_max = shoulder_pitch_bias + max_offsets[0]
    shoulder_roll_min = shoulder_roll_bias + min_offsets[1]
    shoulder_roll_max = shoulder_roll_bias + max_offsets[1]
    shoulder_yaw_min = shoulder_yaw_bias + min_offsets[2]
    shoulder_yaw_max = shoulder_yaw_bias + max_offsets[2]
    elbow_min = elbow_bias + min_offsets[3]
    elbow_max = elbow_bias + max_offsets[3]
    wrist_min = wrist_bias + min_offsets[4]
    wrist_max = wrist_bias + max_offsets[4]

    # Determine shoulder roll safe position based on which arm
    if is_right_arm:
        # Right arm: negative limit moves away from body
        shoulder_roll_safe = shoulder_roll_min
        shoulder_roll_other = shoulder_roll_max
    else:
        # Left arm: positive limit moves away from body
        shoulder_roll_safe = shoulder_roll_max
        shoulder_roll_other = shoulder_roll_min

    # Phase 0: Shoulder roll -> safe limit
    shoulder_roll_target = jnp.where(
        current_phase == 0,
        shoulder_roll_bias + progress * (shoulder_roll_safe - shoulder_roll_bias),
        jnp.where(
            current_phase < 13,  # Phases 1-12: keep at safe limit
            shoulder_roll_safe,
            jnp.where(
                current_phase == 13,  # Phase 13: safe -> other limit
                shoulder_roll_safe + progress * (shoulder_roll_other - shoulder_roll_safe),
                shoulder_roll_other + progress * (shoulder_roll_bias - shoulder_roll_other),  # Phase 14: other -> bias
            ),
        ),
    )

    # Phases 1-3: Shoulder pitch -> min, max, bias
    shoulder_pitch_target = jnp.where(
        current_phase == 1,  # bias -> min
        shoulder_pitch_bias + progress * (shoulder_pitch_min - shoulder_pitch_bias),
        jnp.where(
            current_phase == 2,  # min -> max
            shoulder_pitch_min + progress * (shoulder_pitch_max - shoulder_pitch_min),
            jnp.where(
                current_phase == 3,  # max -> bias
                shoulder_pitch_max + progress * (shoulder_pitch_bias - shoulder_pitch_max),
                shoulder_pitch_bias,  # default: stay at bias
            ),
        ),
    )

    # Phases 4-6: Shoulder yaw -> min, max, bias
    shoulder_yaw_target = jnp.where(
        current_phase == 4,  # bias -> min
        shoulder_yaw_bias + progress * (shoulder_yaw_min - shoulder_yaw_bias),
        jnp.where(
            current_phase == 5,  # min -> max
            shoulder_yaw_min + progress * (shoulder_yaw_max - shoulder_yaw_min),
            jnp.where(
                current_phase == 6,  # max -> bias
                shoulder_yaw_max + progress * (shoulder_yaw_bias - shoulder_yaw_max),
                shoulder_yaw_bias,  # default: stay at bias
            ),
        ),
    )

    # Phases 7-9: Elbow -> min, max, bias
    elbow_target = jnp.where(
        current_phase == 7,  # bias -> min
        elbow_bias + progress * (elbow_min - elbow_bias),
        jnp.where(
            current_phase == 8,  # min -> max
            elbow_min + progress * (elbow_max - elbow_min),
            jnp.where(
                current_phase == 9,  # max -> bias
                elbow_max + progress * (elbow_bias - elbow_max),
                elbow_bias,  # default: stay at bias
            ),
        ),
    )

    # Phases 10-12: Wrist -> min, max, bias
    wrist_target = jnp.where(
        current_phase == 10,  # bias -> min
        wrist_bias + progress * (wrist_min - wrist_bias),
        jnp.where(
            current_phase == 11,  # min -> max
            wrist_min + progress * (wrist_max - wrist_min),
            jnp.where(
                current_phase == 12,  # max -> bias
                wrist_max + progress * (wrist_bias - wrist_max),
                wrist_bias,  # default: stay at bias
            ),
        ),
    )

    # Apply all targets
    targets = targets.at[shoulder_pitch_idx].set(shoulder_pitch_target)
    targets = targets.at[shoulder_roll_idx].set(shoulder_roll_target)
    targets = targets.at[shoulder_yaw_idx].set(shoulder_yaw_target)
    targets = targets.at[elbow_idx].set(elbow_target)
    targets = targets.at[wrist_idx].set(wrist_target)

    return targets


def make_both_arms_simultaneous_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Tests both arms simultaneously: left and right arms move through their sequences at the same time.

    Both arms execute the same 15-phase sequence in parallel, looping forever.
    """
    # Left arm joint names for IDs 11-15
    left_arm_joint_names = [
        "dof_left_shoulder_pitch_03",  # ID 11
        "dof_left_shoulder_roll_03",  # ID 12
        "dof_left_shoulder_yaw_02",  # ID 13
        "dof_left_elbow_02",  # ID 14
        "dof_left_wrist_00",  # ID 15
    ]

    # Right arm joint names for IDs 21-25
    right_arm_joint_names = [
        "dof_right_shoulder_pitch_03",  # ID 21
        "dof_right_shoulder_roll_03",  # ID 22
        "dof_right_shoulder_yaw_02",  # ID 23
        "dof_right_elbow_02",  # ID 24
        "dof_right_wrist_00",  # ID 25
    ]

    # Get indices for both arms
    left_target_indices = jnp.array([joint_names.index(name) for name in left_arm_joint_names])
    right_target_indices = jnp.array([joint_names.index(name) for name in right_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits for both arms
    left_min_offsets, left_max_offsets = get_motion_limits(left_arm_joint_names)
    right_min_offsets, right_max_offsets = get_motion_limits(right_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration
    arm_phases = 15  # 15 phases per arm
    full_cycle_time = arm_phases * phase_duration  # Both arms move together

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which phase both arms are in (they move together)
        cycle_time = t % full_cycle_time
        current_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        current_phase = jnp.clip(current_phase, 0, arm_phases - 1)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Generate targets for both arms simultaneously
        left_targets = _make_arm_sequential_targets(
            joint_names,
            left_arm_joint_names,
            left_target_indices,
            left_min_offsets,
            left_max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_arm=False,
        )

        right_targets = _make_arm_sequential_targets(
            joint_names,
            right_arm_joint_names,
            right_target_indices,
            right_min_offsets,
            right_max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_arm=True,
        )

        # Combine both arm targets (each arm only affects its own joints)
        # Start with left arm targets, then overlay right arm targets
        targets = left_targets
        for i, right_idx in enumerate(right_target_indices):
            targets = targets.at[right_idx].set(right_targets[right_idx])

        return targets, jnp.array([t])

    return Recipe("kbot_both_arms_simultaneous", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_left_arm_sequential_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially moves left arm joints (IDs 11-15) through their range of motion in a collision-avoiding sequence.

    Sequence to avoid self-collision:
    1. Shoulder roll -> positive limit (keeps arm away from body)
    2. Shoulder pitch -> min -> max -> bias
    3. Shoulder yaw -> min -> max -> bias
    4. Elbow -> min -> max -> bias
    5. Wrist -> min -> max -> bias
    6. Shoulder roll -> min -> bias (return to neutral)
    """
    # Left arm joint names for IDs 11-15 (indices 5-9 in JOINT_BIASES)
    left_arm_joint_names = [
        "dof_left_shoulder_pitch_03",  # ID 11 (index 5)
        "dof_left_shoulder_roll_03",  # ID 12 (index 6)
        "dof_left_shoulder_yaw_02",  # ID 13 (index 7)
        "dof_left_elbow_02",  # ID 14 (index 8)
        "dof_left_wrist_00",  # ID 15 (index 9)
    ]

    # Get indices for these joints
    target_indices = jnp.array([joint_names.index(name) for name in left_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits for left arm joints from centralized limits map
    min_offsets, max_offsets = get_motion_limits(left_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration  # Time for each phase
    total_phases = 15
    full_cycle_time = total_phases * phase_duration

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which phase we're in using modulo arithmetic
        cycle_time = t % full_cycle_time
        current_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        current_phase = jnp.clip(current_phase, 0, total_phases - 1)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Use helper function to generate targets
        targets = _make_arm_sequential_targets(
            joint_names,
            left_arm_joint_names,
            target_indices,
            min_offsets,
            max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_arm=False,
        )

        return targets, jnp.array([t])

    return Recipe("kbot_left_arm_sequential", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_right_arm_sequential_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially moves right arm joints (IDs 21-25) through their range of motion in a collision-avoiding sequence.

    Sequence to avoid self-collision:
    1. Shoulder roll -> negative limit (keeps arm away from body for right side)
    2. Shoulder pitch -> min -> max -> bias
    3. Shoulder yaw -> min -> max -> bias
    4. Elbow -> min -> max -> bias
    5. Wrist -> min -> max -> bias
    6. Shoulder roll -> max -> bias (return to neutral)
    """
    # Right arm joint names for IDs 21-25 (indices 0-4 in JOINT_BIASES)
    right_arm_joint_names = [
        "dof_right_shoulder_pitch_03",  # ID 21 (index 0)
        "dof_right_shoulder_roll_03",  # ID 22 (index 1)
        "dof_right_shoulder_yaw_02",  # ID 23 (index 2)
        "dof_right_elbow_02",  # ID 24 (index 3)
        "dof_right_wrist_00",  # ID 25 (index 4)
    ]

    # Get indices for these joints
    target_indices = jnp.array([joint_names.index(name) for name in right_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits for right arm joints from centralized limits map
    min_offsets, max_offsets = get_motion_limits(right_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration  # Time for each phase
    total_phases = 15
    full_cycle_time = total_phases * phase_duration

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which phase we're in using modulo arithmetic
        cycle_time = t % full_cycle_time
        current_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        current_phase = jnp.clip(current_phase, 0, total_phases - 1)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Use helper function to generate targets
        targets = _make_arm_sequential_targets(
            joint_names,
            right_arm_joint_names,
            target_indices,
            min_offsets,
            max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_arm=True,
        )

        return targets, jnp.array([t])

    return Recipe("kbot_right_arm_sequential", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_both_arms_sequential_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially tests both arms: left arm sequence, then right arm sequence, looping forever.

    Sequence:
    1. Left arm full sequence (15 phases)
    2. Right arm full sequence (15 phases)
    3. Loop back to left arm
    """
    # Left arm joint names for IDs 11-15
    left_arm_joint_names = [
        "dof_left_shoulder_pitch_03",  # ID 11
        "dof_left_shoulder_roll_03",  # ID 12
        "dof_left_shoulder_yaw_02",  # ID 13
        "dof_left_elbow_02",  # ID 14
        "dof_left_wrist_00",  # ID 15
    ]

    # Right arm joint names for IDs 21-25
    right_arm_joint_names = [
        "dof_right_shoulder_pitch_03",  # ID 21
        "dof_right_shoulder_roll_03",  # ID 22
        "dof_right_shoulder_yaw_02",  # ID 23
        "dof_right_elbow_02",  # ID 24
        "dof_right_wrist_00",  # ID 25
    ]

    # Get indices for both arms
    left_target_indices = jnp.array([joint_names.index(name) for name in left_arm_joint_names])
    right_target_indices = jnp.array([joint_names.index(name) for name in right_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits for both arms
    left_min_offsets, left_max_offsets = get_motion_limits(left_arm_joint_names)
    right_min_offsets, right_max_offsets = get_motion_limits(right_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration
    arm_phases = 15  # 15 phases per arm
    total_phases = arm_phases * 2  # Left arm + Right arm
    full_cycle_time = total_phases * phase_duration

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which arm and phase we're in
        cycle_time = t % full_cycle_time
        global_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        global_phase = jnp.clip(global_phase, 0, total_phases - 1)

        # Determine if we're testing left arm (phases 0-14) or right arm (phases 15-29)
        is_left_arm_phase = global_phase < arm_phases
        arm_phase = jnp.where(is_left_arm_phase, global_phase, global_phase - arm_phases)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Generate targets for left arm when active
        left_targets = _make_arm_sequential_targets(
            joint_names,
            left_arm_joint_names,
            left_target_indices,
            left_min_offsets,
            left_max_offsets,
            bias_vec,
            arm_phase,
            progress,
            is_right_arm=False,
        )

        # Generate targets for right arm when active
        right_targets = _make_arm_sequential_targets(
            joint_names,
            right_arm_joint_names,
            right_target_indices,
            right_min_offsets,
            right_max_offsets,
            bias_vec,
            arm_phase,
            progress,
            is_right_arm=True,
        )

        # Select which arm's targets to use
        targets = jnp.where(is_left_arm_phase, left_targets, right_targets)

        return targets, jnp.array([t])

    return Recipe("kbot_both_arms_sequential", init_fn, step_fn, NUM_COMMANDS, carry_size)


def _make_leg_sequential_targets(
    joint_names: list[str],
    leg_joint_names: list[str],
    arm_joint_names: list[str],
    leg_target_indices: jnp.ndarray,
    arm_target_indices: jnp.ndarray,
    leg_min_offsets: jnp.ndarray,
    leg_max_offsets: jnp.ndarray,
    arm_min_offsets: jnp.ndarray,
    arm_max_offsets: jnp.ndarray,
    bias_vec: jnp.ndarray,
    current_phase: jnp.ndarray,
    progress: jnp.ndarray,
    is_right_leg: bool = False,
) -> jnp.ndarray:
    """Helper function to generate leg sequential targets for left or right leg.

    17-phase sequence (UPDATED for safety):
    Phase 0: Arm shoulder pitch -> positive (for balance)
    Phases 1-3: Hip pitch -> min, max, bias (clear legs away from body first)
    Phase 4: Hip roll -> safe direction (away from body)
    Phase 5: Hip roll -> opposite direction (toward body)
    Phases 6-8: Hip yaw -> min, max, bias
    Phases 9-11: Knee -> min, max, bias
    Phases 12-14: Ankle -> min, max, bias
    Phase 15: Hip roll -> bias
    Phase 16: Arm shoulder pitch -> bias

    For is_right_leg:
    - If True, right leg moves hip roll positive first (away from body)
    - If False, left leg moves hip roll negative first (away from body)
    """
    targets = bias_vec.copy()

    # Joint indices for leg: [hip_pitch, hip_roll, hip_yaw, knee, ankle]
    hip_pitch_idx = leg_target_indices[0]
    hip_roll_idx = leg_target_indices[1]
    hip_yaw_idx = leg_target_indices[2]
    knee_idx = leg_target_indices[3]
    ankle_idx = leg_target_indices[4]

    # Arm joint: [shoulder_pitch] for stability
    shoulder_pitch_idx = arm_target_indices[0]

    # Get bias positions
    hip_pitch_bias = bias_vec[hip_pitch_idx]
    hip_roll_bias = bias_vec[hip_roll_idx]
    hip_yaw_bias = bias_vec[hip_yaw_idx]
    knee_bias = bias_vec[knee_idx]
    ankle_bias = bias_vec[ankle_idx]
    shoulder_pitch_bias = bias_vec[shoulder_pitch_idx]

    # Calculate limit positions
    hip_pitch_min = hip_pitch_bias + leg_min_offsets[0]
    hip_pitch_max = hip_pitch_bias + leg_max_offsets[0]
    hip_roll_min = hip_roll_bias + leg_min_offsets[1]
    hip_roll_max = hip_roll_bias + leg_max_offsets[1]
    hip_yaw_min = hip_yaw_bias + leg_min_offsets[2]
    hip_yaw_max = hip_yaw_bias + leg_max_offsets[2]
    knee_min = knee_bias + leg_min_offsets[3]
    knee_max = knee_bias + leg_max_offsets[3]
    ankle_min = ankle_bias + leg_min_offsets[4]
    ankle_max = ankle_bias + leg_max_offsets[4]
    # shoulder_pitch_min = shoulder_pitch_bias + arm_min_offsets[0] # Never used
    shoulder_pitch_max = shoulder_pitch_bias + arm_max_offsets[0]

    # Determine hip roll safe directions based on which leg
    if is_right_leg:
        # Right leg: positive moves away from body, negative toward body
        hip_roll_safe = hip_roll_max  # Move away from body first
        hip_roll_opposite = hip_roll_min  # Then test toward body
        hip_roll_hold = hip_roll_min  # Hold toward body during other joint tests
    else:
        # Left leg: negative moves away from body, positive toward body
        hip_roll_safe = hip_roll_min  # Move away from body first
        hip_roll_opposite = hip_roll_max  # Then test toward body
        hip_roll_hold = hip_roll_max  # Hold toward body during other joint tests

    # Phase 0: Arm shoulder pitch -> positive limit (for balance)
    shoulder_pitch_target = jnp.where(
        current_phase == 0,
        shoulder_pitch_bias + progress * (shoulder_pitch_max - shoulder_pitch_bias),
        jnp.where(
            current_phase == 16,  # Phase 16: return to bias
            shoulder_pitch_max + progress * (shoulder_pitch_bias - shoulder_pitch_max),
            jnp.where(
                current_phase < 16,  # Phases 1-15: keep at positive limit
                shoulder_pitch_max,
                shoulder_pitch_bias,  # default: at bias
            ),
        ),
    )

    # Phases 1-3: Hip pitch -> min, max, bias (clear legs away from body FIRST)
    hip_pitch_target = jnp.where(
        current_phase == 1,  # bias -> min
        hip_pitch_bias + progress * (hip_pitch_min - hip_pitch_bias),
        jnp.where(
            current_phase == 2,  # min -> max
            hip_pitch_min + progress * (hip_pitch_max - hip_pitch_min),
            jnp.where(
                current_phase == 3,  # max -> bias
                hip_pitch_max + progress * (hip_pitch_bias - hip_pitch_max),
                hip_pitch_bias,  # default: stay at bias
            ),
        ),
    )

    # Phases 4-5: Hip roll -> safe direction, then opposite direction
    hip_roll_target = jnp.where(
        current_phase == 4,  # bias -> safe direction (away from body)
        hip_roll_bias + progress * (hip_roll_safe - hip_roll_bias),
        jnp.where(
            current_phase == 5,  # safe -> opposite direction (toward body)
            hip_roll_safe + progress * (hip_roll_opposite - hip_roll_safe),
            jnp.where(
                current_phase == 15,  # Phase 15: hold position -> bias
                hip_roll_hold + progress * (hip_roll_bias - hip_roll_hold),
                jnp.where(
                    (current_phase >= 6) & (current_phase <= 14),  # Phases 6-14: keep at hold position
                    hip_roll_hold,
                    hip_roll_bias,  # default: at bias
                ),
            ),
        ),
    )

    # Phases 6-8: Hip yaw -> min, max, bias
    hip_yaw_target = jnp.where(
        current_phase == 6,  # bias -> min
        hip_yaw_bias + progress * (hip_yaw_min - hip_yaw_bias),
        jnp.where(
            current_phase == 7,  # min -> max
            hip_yaw_min + progress * (hip_yaw_max - hip_yaw_min),
            jnp.where(
                current_phase == 8,  # max -> bias
                hip_yaw_max + progress * (hip_yaw_bias - hip_yaw_max),
                hip_yaw_bias,  # default: stay at bias
            ),
        ),
    )

    # Phases 9-11: Knee -> min, max, bias
    knee_target = jnp.where(
        current_phase == 9,  # bias -> min
        knee_bias + progress * (knee_min - knee_bias),
        jnp.where(
            current_phase == 10,  # min -> max
            knee_min + progress * (knee_max - knee_min),
            jnp.where(
                current_phase == 11,  # max -> bias
                knee_max + progress * (knee_bias - knee_max),
                knee_bias,  # default: stay at bias
            ),
        ),
    )

    # Phases 12-14: Ankle -> min, max, bias
    ankle_target = jnp.where(
        current_phase == 12,  # bias -> min
        ankle_bias + progress * (ankle_min - ankle_bias),
        jnp.where(
            current_phase == 13,  # min -> max
            ankle_min + progress * (ankle_max - ankle_min),
            jnp.where(
                current_phase == 14,  # max -> bias
                ankle_max + progress * (ankle_bias - ankle_max),
                ankle_bias,  # default: stay at bias
            ),
        ),
    )

    # Apply all targets
    targets = targets.at[shoulder_pitch_idx].set(shoulder_pitch_target)
    targets = targets.at[hip_pitch_idx].set(hip_pitch_target)
    targets = targets.at[hip_roll_idx].set(hip_roll_target)
    targets = targets.at[hip_yaw_idx].set(hip_yaw_target)
    targets = targets.at[knee_idx].set(knee_target)
    targets = targets.at[ankle_idx].set(ankle_target)

    return targets


def make_left_leg_sequential_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially moves left leg joints through their range of motion in a collision-avoiding sequence.

    Sequence to avoid instability and collisions:
    1. Left arm shoulder pitch -> positive limit (for balance)
    2. Left hip roll -> negative limit, then positive limit (and hold for remainder)
    3. Left hip pitch -> min -> max -> bias
    4. Left hip yaw -> min -> max -> bias
    5. Knee -> min -> max -> bias
    6. Ankle -> min -> max -> bias
    7. Left hip roll -> bias
    8. Left arm shoulder pitch -> bias
    """
    # Left leg joint names
    left_leg_joint_names = [
        "dof_left_hip_pitch_04",  # ID 31
        "dof_left_hip_roll_03",  # ID 32
        "dof_left_hip_yaw_03",  # ID 33
        "dof_left_knee_04",  # ID 34
        "dof_left_ankle_02",  # ID 35
    ]

    # Left arm joint for stability
    left_arm_joint_names = [
        "dof_left_shoulder_pitch_03",  # ID 11
    ]

    # Get indices for these joints
    leg_target_indices = jnp.array([joint_names.index(name) for name in left_leg_joint_names])
    arm_target_indices = jnp.array([joint_names.index(name) for name in left_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits
    leg_min_offsets, leg_max_offsets = get_motion_limits(left_leg_joint_names)
    arm_min_offsets, arm_max_offsets = get_motion_limits(left_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration
    total_phases = 17
    full_cycle_time = total_phases * phase_duration

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which phase we're in using modulo arithmetic
        cycle_time = t % full_cycle_time
        current_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        current_phase = jnp.clip(current_phase, 0, total_phases - 1)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Use helper function to generate targets
        targets = _make_leg_sequential_targets(
            joint_names,
            left_leg_joint_names,
            left_arm_joint_names,
            leg_target_indices,
            arm_target_indices,
            leg_min_offsets,
            leg_max_offsets,
            arm_min_offsets,
            arm_max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_leg=False,
        )

        return targets, jnp.array([t])

    return Recipe("kbot_left_leg_sequential", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_right_leg_sequential_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially moves right leg joints through their range of motion in a collision-avoiding sequence.

    Sequence to avoid instability and collisions:
    1. Right arm shoulder pitch -> positive limit (for balance)
    2. Right hip roll -> positive limit, then negative limit (and hold for remainder)
    3. Right hip pitch -> min -> max -> bias
    4. Right hip yaw -> min -> max -> bias
    5. Knee -> min -> max -> bias
    6. Ankle -> min -> max -> bias
    7. Right hip roll -> bias
    8. Right arm shoulder pitch -> bias
    """
    # Right leg joint names
    right_leg_joint_names = [
        "dof_right_hip_pitch_04",  # ID 41
        "dof_right_hip_roll_03",  # ID 42
        "dof_right_hip_yaw_03",  # ID 43
        "dof_right_knee_04",  # ID 44
        "dof_right_ankle_02",  # ID 45
    ]

    # Right arm joint for stability
    right_arm_joint_names = [
        "dof_right_shoulder_pitch_03",  # ID 21
    ]

    # Get indices for these joints
    leg_target_indices = jnp.array([joint_names.index(name) for name in right_leg_joint_names])
    arm_target_indices = jnp.array([joint_names.index(name) for name in right_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits
    leg_min_offsets, leg_max_offsets = get_motion_limits(right_leg_joint_names)
    arm_min_offsets, arm_max_offsets = get_motion_limits(right_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration
    total_phases = 17
    full_cycle_time = total_phases * phase_duration

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which phase we're in using modulo arithmetic
        cycle_time = t % full_cycle_time
        current_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        current_phase = jnp.clip(current_phase, 0, total_phases - 1)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Use helper function to generate targets
        targets = _make_leg_sequential_targets(
            joint_names,
            right_leg_joint_names,
            right_arm_joint_names,
            leg_target_indices,
            arm_target_indices,
            leg_min_offsets,
            leg_max_offsets,
            arm_min_offsets,
            arm_max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_leg=True,
        )

        return targets, jnp.array([t])

    return Recipe("kbot_right_leg_sequential", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_both_legs_sequential_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Sequentially tests both legs: left leg sequence, then right leg sequence, looping forever.

    Sequence:
    1. Left leg full sequence (17 phases)
    2. Right leg full sequence (17 phases)
    3. Loop back to left leg
    """
    # Left leg joint names
    left_leg_joint_names = [
        "dof_left_hip_pitch_04",  # ID 31
        "dof_left_hip_roll_03",  # ID 32
        "dof_left_hip_yaw_03",  # ID 33
        "dof_left_knee_04",  # ID 34
        "dof_left_ankle_02",  # ID 35
    ]

    # Right leg joint names
    right_leg_joint_names = [
        "dof_right_hip_pitch_04",  # ID 41
        "dof_right_hip_roll_03",  # ID 42
        "dof_right_hip_yaw_03",  # ID 43
        "dof_right_knee_04",  # ID 44
        "dof_right_ankle_02",  # ID 45
    ]

    # Arm joints for stability
    left_arm_joint_names = ["dof_left_shoulder_pitch_03"]  # ID 11
    right_arm_joint_names = ["dof_right_shoulder_pitch_03"]  # ID 21

    # Get indices for both legs
    left_leg_indices = jnp.array([joint_names.index(name) for name in left_leg_joint_names])
    right_leg_indices = jnp.array([joint_names.index(name) for name in right_leg_joint_names])
    left_arm_indices = jnp.array([joint_names.index(name) for name in left_arm_joint_names])
    right_arm_indices = jnp.array([joint_names.index(name) for name in right_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits for both legs
    left_leg_min_offsets, left_leg_max_offsets = get_motion_limits(left_leg_joint_names)
    right_leg_min_offsets, right_leg_max_offsets = get_motion_limits(right_leg_joint_names)
    left_arm_min_offsets, left_arm_max_offsets = get_motion_limits(left_arm_joint_names)
    right_arm_min_offsets, right_arm_max_offsets = get_motion_limits(right_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration
    leg_phases = 17  # 17 phases per leg
    total_phases = leg_phases * 2  # Left leg + Right leg
    full_cycle_time = total_phases * phase_duration

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which leg and phase we're in
        cycle_time = t % full_cycle_time
        global_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        global_phase = jnp.clip(global_phase, 0, total_phases - 1)

        # Determine if we're testing left leg (phases 0-16) or right leg (phases 17-33)
        is_left_leg_phase = global_phase < leg_phases
        leg_phase = jnp.where(is_left_leg_phase, global_phase, global_phase - leg_phases)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Generate targets for left leg when active
        left_targets = _make_leg_sequential_targets(
            joint_names,
            left_leg_joint_names,
            left_arm_joint_names,
            left_leg_indices,
            left_arm_indices,
            left_leg_min_offsets,
            left_leg_max_offsets,
            left_arm_min_offsets,
            left_arm_max_offsets,
            bias_vec,
            leg_phase,
            progress,
            is_right_leg=False,
        )

        # Generate targets for right leg when active
        right_targets = _make_leg_sequential_targets(
            joint_names,
            right_leg_joint_names,
            right_arm_joint_names,
            right_leg_indices,
            right_arm_indices,
            right_leg_min_offsets,
            right_leg_max_offsets,
            right_arm_min_offsets,
            right_arm_max_offsets,
            bias_vec,
            leg_phase,
            progress,
            is_right_leg=True,
        )

        # Select which leg's targets to use
        targets = jnp.where(is_left_leg_phase, left_targets, right_targets)

        return targets, jnp.array([t])

    return Recipe("kbot_both_legs_sequential", init_fn, step_fn, NUM_COMMANDS, carry_size)


def make_both_legs_simultaneous_recipe(joint_names: list[str], dt: float, motion_duration: float = 2.0) -> Recipe:
    """Tests both legs simultaneously: left and right legs move through their sequences at the same time.

    Both legs execute the same 17-phase sequence in parallel, looping forever.
    """
    # Left leg joint names
    left_leg_joint_names = [
        "dof_left_hip_pitch_04",  # ID 31
        "dof_left_hip_roll_03",  # ID 32
        "dof_left_hip_yaw_03",  # ID 33
        "dof_left_knee_04",  # ID 34
        "dof_left_ankle_02",  # ID 35
    ]

    # Right leg joint names
    right_leg_joint_names = [
        "dof_right_hip_pitch_04",  # ID 41
        "dof_right_hip_roll_03",  # ID 42
        "dof_right_hip_yaw_03",  # ID 43
        "dof_right_knee_04",  # ID 44
        "dof_right_ankle_02",  # ID 45
    ]

    # Arm joints for stability
    left_arm_joint_names = ["dof_left_shoulder_pitch_03"]  # ID 11
    right_arm_joint_names = ["dof_right_shoulder_pitch_03"]  # ID 21

    # Get indices for both legs
    left_leg_indices = jnp.array([joint_names.index(name) for name in left_leg_joint_names])
    right_leg_indices = jnp.array([joint_names.index(name) for name in right_leg_joint_names])
    left_arm_indices = jnp.array([joint_names.index(name) for name in left_arm_joint_names])
    right_arm_indices = jnp.array([joint_names.index(name) for name in right_arm_joint_names])
    bias_vec = get_bias_vector(joint_names)

    # Get motion limits for both legs
    left_leg_min_offsets, left_leg_max_offsets = get_motion_limits(left_leg_joint_names)
    right_leg_min_offsets, right_leg_max_offsets = get_motion_limits(right_leg_joint_names)
    left_arm_min_offsets, left_arm_max_offsets = get_motion_limits(left_arm_joint_names)
    right_arm_min_offsets, right_arm_max_offsets = get_motion_limits(right_arm_joint_names)

    # Motion parameters
    phase_duration = motion_duration
    leg_phases = 17  # 17 phases per leg
    full_cycle_time = leg_phases * phase_duration  # Both legs move together

    carry_size = (1,)  # [time]

    @jax.jit
    def init_fn() -> Array:
        return jnp.zeros(carry_size)

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt

        # Calculate which phase both legs are in (they move together)
        cycle_time = t % full_cycle_time
        current_phase = jnp.floor(cycle_time / phase_duration).astype(jnp.int32)
        current_phase = jnp.clip(current_phase, 0, leg_phases - 1)

        # Time progress within current phase (0.0 to 1.0)
        phase_time = cycle_time % phase_duration
        progress = jnp.minimum(phase_time / phase_duration, 1.0)

        # Generate targets for both legs simultaneously
        left_targets = _make_leg_sequential_targets(
            joint_names,
            left_leg_joint_names,
            left_arm_joint_names,
            left_leg_indices,
            left_arm_indices,
            left_leg_min_offsets,
            left_leg_max_offsets,
            left_arm_min_offsets,
            left_arm_max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_leg=False,
        )

        right_targets = _make_leg_sequential_targets(
            joint_names,
            right_leg_joint_names,
            right_arm_joint_names,
            right_leg_indices,
            right_arm_indices,
            right_leg_min_offsets,
            right_leg_max_offsets,
            right_arm_min_offsets,
            right_arm_max_offsets,
            bias_vec,
            current_phase,
            progress,
            is_right_leg=True,
        )

        # Combine both leg targets (each leg only affects its own joints)
        # Start with left leg targets, then overlay right leg targets
        targets = left_targets
        for i, right_idx in enumerate(right_leg_indices):
            targets = targets.at[right_idx].set(right_targets[right_idx])
        # Also apply right arm target
        for i, right_idx in enumerate(right_arm_indices):
            targets = targets.at[right_idx].set(right_targets[right_idx])

        return targets, jnp.array([t])

    return Recipe("kbot_both_legs_simultaneous", init_fn, step_fn, NUM_COMMANDS, carry_size)


def build_kinfer_file(recipe: Recipe, joint_names: list[str], out_dir: Path) -> Path:
    """Build a kinfer file for a given recipe."""
    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=recipe.num_commands,
        carry_size=recipe.carry_size,
    )
    kinfer_blob = pack(
        export_fn(recipe.init_fn, metadata),  # type: ignore[arg-type]
        export_fn(recipe.step_fn, metadata),  # type: ignore[arg-type]
        metadata,
    )
    out_path = out_dir / f"{recipe.name}.kinfer"
    out_path.write_bytes(kinfer_blob)
    return out_path


def main() -> None:
    colorlogging.configure()
    parser = argparse.ArgumentParser()
    default_output = Path(__file__).parent / "assets"
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Output path for the kinfer model (default: %(default)s)",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = get_joint_names()
    num_joints = len(joint_names)
    logger.info("Number of joints: %s", num_joints)
    logger.info("Joint names: %s", joint_names)

    recipes = [
        make_zero_recipe(num_joints=num_joints, dt=SIM_DT),
        make_bias_recipe(joint_names=joint_names, dt=SIM_DT),
        make_sine_recipe(joint_names=joint_names, dt=SIM_DT),
        make_single_joint_linear_recipe(
            target_joint_name="dof_right_knee_04",
            start_pos=0.0,
            end_pos=-13.0,
            num_steps=1000,
            joint_names=joint_names,
            dt=SIM_DT,
        ),
        make_echo_recipe(joint_names=joint_names, dt=SIM_DT),
        make_step_recipe(joint_names=joint_names, dt=SIM_DT, seed=42),
        make_step_echo_recipe(joint_names=joint_names, dt=SIM_DT, seed=42),
        make_imu_arm_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=1.0),
        make_left_arm_sequential_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_right_arm_sequential_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_both_arms_sequential_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_both_arms_simultaneous_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_left_leg_sequential_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_right_leg_sequential_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_both_legs_sequential_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
        make_both_legs_simultaneous_recipe(joint_names=joint_names, dt=SIM_DT, motion_duration=2.0),
    ]
    for recipe in recipes:
        out_path = build_kinfer_file(recipe, joint_names, out_dir)
        logger.info("kinfer model written to %s", out_path)


if __name__ == "__main__":
    main()
