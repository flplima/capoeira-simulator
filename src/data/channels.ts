// The animation channels a pose is made of. Every pose — named pose, keyframe,
// sampled frame — is a flat Float64Array indexed by these channels, so
// interpolation, blending and mirroring are plain array maths.
//
// Angles are degrees in anatomical terms, relative to the rest pose (standing
// straight, arms hanging, facing +Z). See README.md "Pose conventions".

export const JOINTS = {
  root: ["x", "z", "lift", "yaw", "pitch", "roll"],
  torso: ["flex", "side", "twist"],
  head: ["flex", "side", "twist"],
  shoulder_l: ["flex", "abd", "rot"],
  elbow_l: ["flex"],
  shoulder_r: ["flex", "abd", "rot"],
  elbow_r: ["flex"],
  hip_l: ["flex", "abd", "rot"],
  knee_l: ["flex"],
  ankle_l: ["flex"],
  hip_r: ["flex", "abd", "rot"],
  knee_r: ["flex"],
  ankle_r: ["flex"],
} as const;

export type JointName = keyof typeof JOINTS;
export const JOINT_NAMES = Object.keys(JOINTS) as JointName[];

export interface Channel {
  joint: JointName;
  field: string;
  key: string;
  /** Linear channels are metres; everything else is an angle in degrees. */
  angular: boolean;
}

export const CHANNELS: Channel[] = JOINT_NAMES.flatMap((joint) =>
  JOINTS[joint].map((field) => ({
    joint,
    field,
    key: `${joint}.${field}`,
    angular: !(joint === "root" && (field === "x" || field === "z" || field === "lift")),
  })),
);

export const CHANNEL_COUNT = CHANNELS.length;
export const CH: Record<string, number> = Object.fromEntries(CHANNELS.map((c, i) => [c.key, i]));

export type Pose = Float64Array;
export const restPose = (): Pose => new Float64Array(CHANNEL_COUNT);

// Mirroring swaps left and right: _l and _r joints trade values (their angles
// are already anatomical, so no sign flip), and everything measured toward the
// character's left — x, yaw, roll, lateral bends and twists — changes sign.
const NEGATED = new Set(["root.x", "root.yaw", "root.roll", "torso.side", "torso.twist", "head.side", "head.twist"]);

const swapSide = (joint: string) =>
  joint.endsWith("_l") ? joint.slice(0, -2) + "_r" : joint.endsWith("_r") ? joint.slice(0, -2) + "_l" : joint;

const MIRROR_SOURCE = CHANNELS.map((c) => CH[`${swapSide(c.joint)}.${c.field}`]);
const MIRROR_SIGN = CHANNELS.map((c) => (NEGATED.has(c.key) ? -1 : 1));

export function mirrorPose(src: Pose, out: Pose = restPose()): Pose {
  const tmp = src === out ? Float64Array.from(src) : src;
  for (let i = 0; i < CHANNEL_COUNT; i++) out[i] = tmp[MIRROR_SOURCE[i]] * MIRROR_SIGN[i];
  return out;
}

/** Shortest signed difference between two angles, in (-180, 180]. */
export function wrapAngle(d: number): number {
  d = ((d + 180) % 360 + 360) % 360 - 180;
  return d === -180 ? 180 : d;
}
