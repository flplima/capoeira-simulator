"""MuJoCo model of the mannequin, built from the same skeleton as the web rig.

The bone offsets match src/rig/rig.ts and the world is Y-up like three.js, so a
body's orientation here is the bone's orientation there. Hinge joints follow
the web's anatomical channels (src/data/channels.ts) with the same signs:
positive flex is forward/up, positive abd is away from the midline, positive
rot is internal rotation, and the hinge order reproduces the Euler orders in
applyPose (YXZ for the trunk and neck, XZY for shoulders and hips). Segment
masses follow de Leva (1996) for a 70 kg body, as in src/physics/mass.ts.

    uv run python model.py > humanoid.xml   # write the MJCF
"""

from dataclasses import dataclass, field

BODY_MASS = 70.0

# name, parent, offset from parent (x, y, z) — identical to rig.ts.
BONES = [
    ("hips", None, (0, 0.97, 0)),
    ("spine", "hips", (0, 0.10, 0)),
    ("chest", "spine", (0, 0.17, 0)),
    ("neck", "chest", (0, 0.23, 0)),
    ("head", "neck", (0, 0.09, 0)),
    ("upperarm_l", "chest", (0.18, 0.19, 0)),
    ("forearm_l", "upperarm_l", (0, -0.29, 0)),
    ("hand_l", "forearm_l", (0, -0.25, 0)),
    ("upperarm_r", "chest", (-0.18, 0.19, 0)),
    ("forearm_r", "upperarm_r", (0, -0.29, 0)),
    ("hand_r", "forearm_r", (0, -0.25, 0)),
    ("thigh_l", "hips", (0.095, -0.06, 0)),
    ("shin_l", "thigh_l", (0, -0.43, 0)),
    ("foot_l", "shin_l", (0, -0.41, 0)),
    ("thigh_r", "hips", (-0.095, -0.06, 0)),
    ("shin_r", "thigh_r", (0, -0.43, 0)),
    ("foot_r", "shin_r", (0, -0.41, 0)),
]

# de Leva (1996) mass fractions; the trunk thirds go to hips/spine/chest.
MASS = {
    "hips": 0.1117, "spine": 0.1633, "chest": 0.1596, "neck": 0.0, "head": 0.0694,
    "upperarm": 0.0271, "forearm": 0.0162, "hand": 0.0061,
    "thigh": 0.1416, "shin": 0.0433, "foot": 0.0137,
}


@dataclass
class Joint:
    name: str
    axis: tuple[float, float, float]
    range: tuple[float, float]  # degrees
    kp: float
    force: float


@dataclass
class Body:
    name: str
    joints: list[Joint] = field(default_factory=list)


def trunk(prefix: str, share: float, flex: tuple, side: float, twist: float, kp: float, force: float) -> list[Joint]:
    """YXZ order: twist (Y), then flex (X), then side bend (−Z), like applyPose."""
    return [
        Joint(f"{prefix}_twist", (0, 1, 0), (-twist * share, twist * share), kp, force),
        Joint(f"{prefix}_flex", (1, 0, 0), (flex[0] * share, flex[1] * share), kp, force),
        Joint(f"{prefix}_side", (0, 0, -1), (-side * share, side * share), kp, force),
    ]


def limb(prefix: str, left: bool, flex: tuple, abd: tuple, rot: tuple, kp: float, force: float) -> list[Joint]:
    """XZY order: flex (−X), then abduction (±Z), then internal rotation (∓Y)."""
    return [
        Joint(f"{prefix}_flex", (-1, 0, 0), flex, kp, force),
        Joint(f"{prefix}_abd", (0, 0, 1) if left else (0, 0, -1), abd, kp, force),
        Joint(f"{prefix}_rot", (0, -1, 0) if left else (0, 1, 0), rot, kp, force),
    ]


def joints_for(bone: str) -> list[Joint]:
    if bone == "spine":
        return trunk("spine", 0.5, (-60, 110), 60, 80, 600, 300)
    if bone == "chest":
        return trunk("chest", 0.5, (-60, 110), 60, 80, 500, 250)
    if bone == "neck":
        return trunk("neck", 1.0, (-70, 80), 45, 80, 60, 40)
    side = bone[-1]
    left = side == "l"
    # (While falling and getting up the arms brace: env.STRONG_ARMS.)
    if bone.startswith("upperarm"):
        return limb(f"shoulder_{side}", left, (-90, 190), (-45, 190), (-100, 100), 120, 80)
    if bone.startswith("forearm"):
        return [Joint(f"elbow_{side}", (-1, 0, 0), (0, 160), 80, 60)]
    if bone.startswith("thigh"):
        return limb(f"hip_{side}", left, (-60, 140), (-40, 90), (-60, 60), 700, 400)
    if bone.startswith("shin"):
        return [Joint(f"knee_{side}", (1, 0, 0), (0, 160), 700, 400)]
    if bone.startswith("foot"):
        return [
            Joint(f"ankle_{side}", (1, 0, 0), (-45, 60), 400, 200),
            Joint(f"ankle_{side}_inv", (0, 0, 1) if left else (0, 0, -1), (-25, 25), 200, 100),
        ]
    return []


def geoms_for(bone: str) -> list[str]:
    """Collision/mass shapes in the bone's frame (Y-up, limbs hang along −Y)."""
    kind = bone[:-2] if bone[-2:] in ("_l", "_r") else bone
    mass = MASS[kind] * BODY_MASS
    if bone == "hips":
        return [f'<geom name="hips" type="ellipsoid" size="0.16 0.11 0.115" pos="0 -0.01 0" mass="{mass:.3f}"/>']
    if bone == "spine":
        return [f'<geom name="spine" type="capsule" fromto="0 0.02 0 0 0.14 0" size="0.11" mass="{mass:.3f}"/>']
    if bone == "chest":
        return [f'<geom name="chest" type="capsule" fromto="-0.06 0.12 0 0.06 0.12 0" size="0.12" mass="{mass:.3f}"/>']
    if bone == "neck":
        return ['<geom name="neck" type="capsule" fromto="0 -0.02 0 0 0.08 0" size="0.045" mass="0.5"/>']
    if bone == "head":
        return [f'<geom name="head" type="sphere" size="0.105" pos="0 0.1 0" mass="{mass - 0.5:.3f}"/>']
    if kind == "upperarm":
        return [f'<geom name="{bone}" type="capsule" fromto="0 -0.03 0 0 -0.26 0" size="0.046" mass="{mass:.3f}"/>']
    if kind == "forearm":
        return [f'<geom name="{bone}" type="capsule" fromto="0 -0.03 0 0 -0.22 0" size="0.038" mass="{mass:.3f}"/>']
    if kind == "hand":
        return [f'<geom name="{bone}" type="capsule" fromto="0 -0.03 0 0 -0.12 0" size="0.032" mass="{mass:.3f}"/>']
    if kind == "thigh":
        return [f'<geom name="{bone}" type="capsule" fromto="0 -0.05 0 0 -0.38 0" size="0.072" mass="{mass:.3f}"/>']
    if kind == "shin":
        return [f'<geom name="{bone}" type="capsule" fromto="0 -0.03 0 0 -0.36 0" size="0.054" mass="{mass:.3f}"/>']
    if kind == "foot":
        # Flat sole 7.5 cm under the ankle, heel 6 cm behind it, toes 17 cm ahead.
        return [
            f'<geom name="{bone}" type="box" size="0.045 0.03 0.115" pos="0 -0.045 0.055" mass="{mass:.3f}" friction="1.2 0.02 0.002"/>'
        ]
    raise ValueError(bone)


SITES = {
    # Points matched against the video's landmarks when fitting reference poses.
    "hips": [("hip_joint_l", (0.095, -0.06, 0)), ("hip_joint_r", (-0.095, -0.06, 0))],
    "chest": [("shoulder_l", (0.18, 0.19, 0)), ("shoulder_r", (-0.18, 0.19, 0))],
    "head": [("nose", (0, 0.09, 0.1)), ("ear_l", (0.09, 0.1, 0)), ("ear_r", (-0.09, 0.1, 0))],
    "shin_l": [("knee_l", (0, 0, 0))], "shin_r": [("knee_r", (0, 0, 0))],
    "foot_l": [("ankle_l", (0, 0, 0)), ("heel_l", (0, -0.075, -0.06)), ("toe_l", (0, -0.075, 0.17))],
    "foot_r": [("ankle_r", (0, 0, 0)), ("heel_r", (0, -0.075, -0.06)), ("toe_r", (0, -0.075, 0.17))],
    "forearm_l": [("elbow_l", (0, 0, 0))], "forearm_r": [("elbow_r", (0, 0, 0))],
    "hand_l": [("wrist_l", (0, 0, 0)), ("fingers_l", (0, -0.15, 0))],
    "hand_r": [("wrist_r", (0, 0, 0)), ("fingers_r", (0, -0.15, 0))],
}


def joint_list() -> list[Joint]:
    return [j for name, _, _ in BONES for j in joints_for(name)]


def mjcf() -> str:
    children: dict[str | None, list[str]] = {}
    for name, parent, _ in BONES:
        children.setdefault(parent, []).append(name)
    offsets = {name: offset for name, _, offset in BONES}

    def body(name: str, depth: int) -> str:
        pad = "  " * depth
        x, y, z = offsets[name]
        lines = [f'{pad}<body name="{name}" pos="{x} {y} {z}">']
        if name == "hips":
            lines.append(f'{pad}  <freejoint name="root"/>')
        for j in joints_for(name):
            a = " ".join(str(v) for v in j.axis)
            lines.append(f'{pad}  <joint name="{j.name}" axis="{a}" range="{j.range[0]} {j.range[1]}"/>')
        for g in geoms_for(name):
            lines.append(f"{pad}  {g}")
        for site, pos in SITES.get(name, []):
            lines.append(f'{pad}  <site name="{site}" pos="{pos[0]} {pos[1]} {pos[2]}" size="0.01"/>')
        for child in children.get(name, []):
            lines.append(body(child, depth + 1))
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    actuators = "\n".join(
        f'    <position name="{j.name}" joint="{j.name}" kp="{j.kp}" dampratio="1" forcerange="{-j.force} {j.force}" ctrlrange="{j.range[0] * 3.14159 / 180:.4f} {j.range[1] * 3.14159 / 180:.4f}"/>'
        for j in joint_list()
    )
    return f"""<mujoco model="capoeirista">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 -9.81 0" integrator="implicitfast"/>
  <default>
    <joint damping="2" armature="0.02" limited="true"/>
    <geom contype="6" conaffinity="5" condim="3" friction="0.9 0.02 0.002" solref="0.015 1"/>
  </default>
  <worldbody>
    <light pos="0 4 3" dir="0 -1 -0.6"/>
    <camera name="front" pos="0 1.0 4.2" xyaxes="1 0 0 0 1 0" mode="trackcom"/>
    <camera name="side" pos="4.2 1.0 0" xyaxes="0 0 -1 0 1 0" mode="trackcom"/>
    <geom name="floor" type="plane" size="20 20 0.1" quat="0.7071068 -0.7071068 0 0" contype="1" conaffinity="2" friction="1.0 0.02 0.002"/>
{body("hips", 2)}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    print(mjcf())
