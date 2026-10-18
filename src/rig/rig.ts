import {
  Bone,
  BufferAttribute,
  BufferGeometry,
  CapsuleGeometry,
  Color,
  CylinderGeometry,
  Euler,
  Matrix4,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  Skeleton,
  SkinnedMesh,
  SphereGeometry,
  TorusGeometry,
  Vector3,
  type EulerOrder,
} from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { CH, type Pose } from "../data/channels";

// A minimalist mannequin: a real bone hierarchy driving one SkinnedMesh. Each
// body segment is a primitive rigidly weighted to its bone, so the figure
// reads like a wooden artist's model in a white abadá with a corda at the waist.

export type BoneName =
  | "hips" | "spine" | "chest" | "neck" | "head"
  | "upperarm_l" | "forearm_l" | "hand_l" | "upperarm_r" | "forearm_r" | "hand_r"
  | "thigh_l" | "shin_l" | "foot_l" | "thigh_r" | "shin_r" | "foot_r";

/** Rest pose: [parent, offset from parent]. Limbs hang along -Y, feet point +Z, the figure faces +Z. */
const BONES: [BoneName, BoneName | null, [number, number, number]][] = [
  ["hips", null, [0, 0.97, 0]],
  ["spine", "hips", [0, 0.1, 0]],
  ["chest", "spine", [0, 0.17, 0]],
  ["neck", "chest", [0, 0.23, 0]],
  ["head", "neck", [0, 0.09, 0]],
  ["upperarm_l", "chest", [0.18, 0.19, 0]],
  ["forearm_l", "upperarm_l", [0, -0.29, 0]],
  ["hand_l", "forearm_l", [0, -0.25, 0]],
  ["upperarm_r", "chest", [-0.18, 0.19, 0]],
  ["forearm_r", "upperarm_r", [0, -0.29, 0]],
  ["hand_r", "forearm_r", [0, -0.25, 0]],
  ["thigh_l", "hips", [0.095, -0.06, 0]],
  ["shin_l", "thigh_l", [0, -0.43, 0]],
  ["foot_l", "shin_l", [0, -0.41, 0]],
  ["thigh_r", "hips", [-0.095, -0.06, 0]],
  ["shin_r", "thigh_r", [0, -0.43, 0]],
  ["foot_r", "shin_r", [0, -0.41, 0]],
];

export const PALETTE = {
  body: "#a7a29a",
  joint: "#77736d",
  pants: "#f1eee6",
  corda: "#e0a82e",
};
type Paint = keyof typeof PALETTE;

interface Part {
  bone: BoneName;
  paint: Paint;
  geometry: BufferGeometry;
  /** Placement in the bone's local space. */
  at?: [number, number, number];
  rotate?: [number, number, number];
  scale?: [number, number, number];
}

const capsule = (r: number, len: number) => new CapsuleGeometry(r, len, 6, 14);
const sphere = (r: number) => new SphereGeometry(r, 18, 12);

function parts(): Part[] {
  const out: Part[] = [
    { bone: "hips", paint: "pants", geometry: sphere(1), at: [0, -0.01, 0], scale: [0.16, 0.11, 0.115] },
    { bone: "hips", paint: "corda", geometry: new TorusGeometry(1, 0.075, 8, 40), at: [0, 0.055, 0], rotate: [Math.PI / 2, 0, 0], scale: [0.158, 0.118, 0.16] },
    { bone: "spine", paint: "body", geometry: capsule(0.11, 0.06), at: [0, 0.08, 0], scale: [1.05, 1, 0.78] },
    { bone: "chest", paint: "body", geometry: capsule(0.125, 0.1), at: [0, 0.12, 0], scale: [1.18, 1, 0.74] },
    { bone: "neck", paint: "joint", geometry: new CylinderGeometry(0.042, 0.048, 0.12, 12), at: [0, 0.03, 0] },
    { bone: "head", paint: "body", geometry: sphere(0.105), at: [0, 0.1, 0.005], scale: [0.92, 1.1, 1] },
    { bone: "head", paint: "joint", geometry: sphere(0.022), at: [0, 0.085, 0.1] },
  ];
  for (const s of ["l", "r"] as const) {
    out.push(
      { bone: `upperarm_${s}`, paint: "joint", geometry: sphere(0.056) },
      { bone: `upperarm_${s}`, paint: "body", geometry: capsule(0.046, 0.19), at: [0, -0.145, 0] },
      { bone: `forearm_${s}`, paint: "joint", geometry: sphere(0.043) },
      { bone: `forearm_${s}`, paint: "body", geometry: capsule(0.038, 0.17), at: [0, -0.125, 0] },
      { bone: `hand_${s}`, paint: "joint", geometry: capsule(0.032, 0.08), at: [0, -0.07, 0], scale: [1, 1, 0.62] },
      { bone: `thigh_${s}`, paint: "pants", geometry: capsule(0.072, 0.27), at: [0, -0.215, 0] },
      { bone: `shin_${s}`, paint: "pants", geometry: sphere(0.062) },
      { bone: `shin_${s}`, paint: "pants", geometry: capsule(0.054, 0.29), at: [0, -0.2, 0] },
      { bone: `foot_${s}`, paint: "body", geometry: sphere(0.042) },
      { bone: `foot_${s}`, paint: "body", geometry: capsule(0.04, 0.15), at: [0, -0.034, 0.065], rotate: [Math.PI / 2, 0, 0], scale: [1.05, 1, 0.7] },
    );
  }
  return out;
}

/** Points that can touch the floor: [bone, local position, radius]. The lowest one decides the height. */
const CONTACTS: [BoneName, [number, number, number], number][] = [
  ["hips", [0, -0.01, 0], 0.11],
  ["spine", [0, 0.08, 0], 0.11],
  ["chest", [0, 0.12, 0], 0.12],
  ["head", [0, 0.1, 0], 0.11],
  ...(["l", "r"] as const).flatMap((s): [BoneName, [number, number, number], number][] => [
    [`upperarm_${s}`, [0, 0, 0], 0.056],
    [`forearm_${s}`, [0, 0, 0], 0.043],
    [`hand_${s}`, [0, -0.04, 0], 0.032],
    [`hand_${s}`, [0, -0.12, 0], 0.032],
    [`shin_${s}`, [0, 0, 0], 0.062],
    [`foot_${s}`, [0, -0.034, -0.01], 0.042],
    [`foot_${s}`, [0, -0.034, 0.14], 0.04],
  ]),
];

export type EffectorName = "foot_l" | "foot_r" | "hand_l" | "hand_r";
export const EFFECTOR_NAMES: EffectorName[] = ["foot_l", "foot_r", "hand_l", "hand_r"];

/** Limb chains that can be pinned to the floor: [upper, middle, end] bones and the probes that touch it. */
const EFFECTORS: Record<EffectorName, { chain: [BoneName, BoneName, BoneName]; probes: [[number, number, number], number][] }> = {
  foot_l: { chain: ["thigh_l", "shin_l", "foot_l"], probes: [[[0, -0.034, -0.01], 0.042], [[0, -0.034, 0.14], 0.04]] },
  foot_r: { chain: ["thigh_r", "shin_r", "foot_r"], probes: [[[0, -0.034, -0.01], 0.042], [[0, -0.034, 0.14], 0.04]] },
  hand_l: { chain: ["upperarm_l", "forearm_l", "hand_l"], probes: [[[0, -0.12, 0], 0.032]] },
  hand_r: { chain: ["upperarm_r", "forearm_r", "hand_r"], probes: [[[0, -0.12, 0], 0.032]] },
};

export interface Effector {
  chain: [Bone, Bone, Bone];
  probes: { point: Object3D; radius: number }[];
}

export interface Rig {
  object: SkinnedMesh;
  bones: Record<BoneName, Bone>;
  contacts: { point: Object3D; radius: number }[];
  effectors: Record<EffectorName, Effector>;
}

export function buildRig(): Rig {
  const bones = {} as Record<BoneName, Bone>;
  for (const [name, parent, offset] of BONES) {
    const bone = new Bone();
    bone.name = name;
    bone.position.set(...offset);
    bones[name] = bone;
    if (parent) bones[parent].add(bone);
  }
  bones.hips.updateMatrixWorld(true);

  const boneIndex = new Map(BONES.map(([name], i) => [name, i]));
  const color = new Color();
  const local = new Matrix4();
  const geometries = parts().map((part) => {
    const g = part.geometry.toNonIndexed();
    g.deleteAttribute("uv");
    local.compose(
      new Vector3(...(part.at ?? [0, 0, 0])),
      new Quaternion().setFromEuler(new Euler(...(part.rotate ?? [0, 0, 0]))),
      new Vector3(...(part.scale ?? [1, 1, 1])),
    );
    g.applyMatrix4(local.premultiply(bones[part.bone].matrixWorld));
    const count = g.attributes.position.count;
    const colors = new Float32Array(count * 3);
    const skinIndex = new Uint16Array(count * 4);
    const skinWeight = new Float32Array(count * 4);
    color.set(PALETTE[part.paint]);
    const index = boneIndex.get(part.bone)!;
    for (let v = 0; v < count; v++) {
      colors.set([color.r, color.g, color.b], v * 3);
      skinIndex[v * 4] = index;
      skinWeight[v * 4] = 1;
    }
    g.setAttribute("color", new BufferAttribute(colors, 3));
    g.setAttribute("skinIndex", new BufferAttribute(skinIndex, 4));
    g.setAttribute("skinWeight", new BufferAttribute(skinWeight, 4));
    return g;
  });

  const geometry = mergeGeometries(geometries)!;
  const material = new MeshStandardMaterial({ vertexColors: true, roughness: 0.82, metalness: 0 });
  const mesh = new SkinnedMesh(geometry, material);
  mesh.add(bones.hips);
  mesh.bind(new Skeleton(BONES.map(([name]) => bones[name])));
  mesh.castShadow = true;
  mesh.frustumCulled = false;

  const contacts = CONTACTS.map(([bone, at, radius]) => {
    const point = new Object3D();
    point.position.set(...at);
    bones[bone].add(point);
    return { point, radius };
  });
  const effectors = Object.fromEntries(
    EFFECTOR_NAMES.map((name) => {
      const { chain, probes } = EFFECTORS[name];
      return [
        name,
        {
          chain: chain.map((b) => bones[b]) as [Bone, Bone, Bone],
          probes: probes.map(([at, radius]) => {
            const point = new Object3D();
            point.position.set(...at);
            bones[chain[2]].add(point);
            return { point, radius };
          }),
        },
      ];
    }),
  ) as Record<EffectorName, Effector>;
  return { object: mesh, bones, contacts, effectors };
}

// Anatomical angles → bone Euler angles. Signs follow from limbs hanging along
// -Y and the figure facing +Z with its left side toward +X.
const D = Math.PI / 180;
const euler = new Euler();
const tmp = new Vector3();

function set(bone: Bone, x: number, y: number, z: number, order: EulerOrder) {
  euler.set(x * D, y * D, z * D, order);
  bone.quaternion.setFromEuler(euler);
}

/** Abduction is applied first (frontal plane), then flexion swings that plane forward. */
function limb(bone: Bone, pose: Pose, joint: string, left: boolean) {
  const flex = pose[CH[`${joint}.flex`]];
  const abd = pose[CH[`${joint}.abd`]];
  const rot = pose[CH[`${joint}.rot`]];
  set(bone, -flex, left ? -rot : rot, left ? abd : -abd, "XZY");
}

export function applyPose(rig: Rig, pose: Pose) {
  const b = rig.bones;
  const p = (key: string) => pose[CH[key]];

  // Root: heading (yaw) first, then lean (pitch) and tilt (roll) in the body frame.
  b.hips.position.set(p("root.x"), 0, p("root.z"));
  set(b.hips, p("root.pitch"), p("root.yaw"), -p("root.roll"), "YXZ");

  set(b.spine, p("torso.flex") * 0.5, p("torso.twist") * 0.5, -p("torso.side") * 0.5, "YXZ");
  set(b.chest, p("torso.flex") * 0.5, p("torso.twist") * 0.5, -p("torso.side") * 0.5, "YXZ");
  set(b.neck, p("head.flex") * 0.4, p("head.twist") * 0.4, -p("head.side") * 0.4, "YXZ");
  set(b.head, p("head.flex") * 0.6, p("head.twist") * 0.6, -p("head.side") * 0.6, "YXZ");

  for (const s of ["l", "r"] as const) {
    const left = s === "l";
    limb(b[`upperarm_${s}`], pose, `shoulder_${s}`, left);
    set(b[`forearm_${s}`], -p(`elbow_${s}.flex`), 0, 0, "XYZ");
    limb(b[`thigh_${s}`], pose, `hip_${s}`, left);
    set(b[`shin_${s}`], p(`knee_${s}.flex`), 0, 0, "XYZ");
    set(b[`foot_${s}`], p(`ankle_${s}.flex`), 0, 0, "XYZ");
  }

  // Stand on the floor: drop the figure until its lowest contact touches y = 0,
  // then raise it by `lift` for airborne moments.
  rig.object.updateMatrixWorld(true);
  let lowest = Infinity;
  for (const c of rig.contacts) lowest = Math.min(lowest, rig.object.worldToLocal(c.point.getWorldPosition(tmp)).y - c.radius);
  b.hips.position.y = -lowest + p("root.lift");
  rig.object.updateMatrixWorld(true);
}
