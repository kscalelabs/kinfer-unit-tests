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
        quaternion: Array,
        initial_heading: Array,
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
        quaternion: Array,
        initial_heading: Array,
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
        quaternion: Array,
        initial_heading: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        offsets = amps * jnp.sin(2 * jnp.pi * freqs * t)
        return bias_vec + offsets, jnp.array([t])

    return Recipe("kbot_sine_motion", init_fn, step_fn, NUM_COMMANDS, carry_size)


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
        quaternion: Array,
        initial_heading: Array,
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
        quaternion: Array,
        initial_heading: Array,
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
    ]
    for recipe in recipes:
        out_path = build_kinfer_file(recipe, joint_names, out_dir)
        logger.info("kinfer model written to %s", out_path)


if __name__ == "__main__":
    main()
