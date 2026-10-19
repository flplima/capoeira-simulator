"""The humanoid as an MJX model: explicit self-collision pairs and solver settings."""

from __future__ import annotations

import mujoco
from mujoco import mjx
import numpy as np

from env_consts import STRONG_ARMS

# Self-collision geom pairs kept in MJX (arms vs head/chest, left leg vs right leg).
# All 17 body geoms still collide with the floor plane.
MJX_SELF_PAIRS = [
    ("thigh_l", "thigh_r"),
    ("shin_l", "shin_r"),
    ("thigh_l", "shin_r"),
    ("thigh_r", "shin_l"),
    ("foot_l", "shin_r"),
    ("foot_r", "shin_l"),
    ("foot_l", "foot_r"),
    ("forearm_l", "head"),
    ("forearm_r", "head"),
    ("hand_l", "head"),
    ("hand_r", "head"),
    ("upperarm_l", "head"),
    ("upperarm_r", "head"),
    ("forearm_l", "chest"),
    ("forearm_r", "chest"),
    ("hand_l", "chest"),
    ("hand_r", "chest"),
]


def load_mjx_model(
    xml_path: str = "humanoid.xml", impl: str = "jax", strong_arms: bool = False
) -> tuple[mujoco.MjModel, mjx.Model]:
    """Load humanoid.xml with explicit self-collision pairs and MJX solver settings."""
    with open(xml_path) as f:
        xml = f.read()
    pair_xml = (
        "  <contact>\n"
        + "\n".join(f'    <pair geom1="{a}" geom2="{b}"/>' for a, b in MJX_SELF_PAIRS)
        + "\n  </contact>\n</mujoco>"
    )
    xml_mjx = xml.replace('contype="6" conaffinity="5"', 'contype="2" conaffinity="1"').replace("</mujoco>", pair_xml)
    m = mujoco.MjModel.from_xml_string(xml_mjx)
    if strong_arms:
        for i in range(m.nu):
            name = m.actuator(i).name
            joint = "shoulder" if name.startswith("shoulder") else "elbow" if name.startswith("elbow") else None
            if joint:
                kp, kv = m.actuator_gainprm[i, 0], m.actuator_biasprm[i, 2]
                strong_kp, strong_force = STRONG_ARMS[joint]
                m.actuator_gainprm[i, 0] = strong_kp
                m.actuator_biasprm[i, 1:3] = -strong_kp, kv * np.sqrt(strong_kp / kp)
                m.actuator_forcerange[i] = -strong_force, strong_force
    m.opt.iterations = 4
    m.opt.ls_iterations = 8
    mx = mjx.put_model(m, impl=impl)
    return m, mx
