import type { Rig } from "../rig/rig";
import { Quaternion, Vector3 } from "three";
import { blend } from "./choreography";
import type { LiveGinga } from "./simulation";

const axis = new Vector3();
const hinge = new Quaternion();

export function effectiveGingaStyle(g: LiveGinga): "angola" | "padrao" | "regional" {
  if (g.gingaStyle !== "auto") return g.gingaStyle;
  if (g.bpm <= 76) return "angola";
  if (g.bpm >= 104) return "regional";
  return "padrao";
}

export function styleJointOffset(g: LiveGinga, jointName: string, w: number, style: "angola" | "padrao" | "regional"): number {
  if (w <= 0 || style === "padrao") return 0;
  if (style === "angola") {
    switch (jointName) {
      case "spine_flex":
        return 0.18 * w;
      case "chest_flex":
        return 0.14 * w;
      case "spine_side":
        return 0.08 * Math.sin(2 * Math.PI * g.phase) * w;
      case "neck_flex":
        return -0.28 * w;
      case "shoulder_l_abd":
      case "shoulder_r_abd":
        return 0.24 * w;
      case "elbow_l":
      case "elbow_r":
        return -0.18 * w;
      case "hip_l_flex":
      case "hip_r_flex":
        return 0.15 * w;
      case "knee_l":
      case "knee_r":
        return 0.3 * w;
      case "ankle_l":
      case "ankle_r":
        return -0.15 * w;
      default:
        return 0;
    }
  }
  // Regional: upright trunk, compact high guard.
  switch (jointName) {
    case "spine_flex":
      return -0.11 * w;
    case "chest_flex":
      return -0.08 * w;
    case "neck_flex":
      return 0.12 * w;
    case "shoulder_l_flex":
    case "shoulder_r_flex":
      return 0.18 * w;
    case "shoulder_l_abd":
    case "shoulder_r_abd":
      return -0.14 * w;
    case "elbow_l":
    case "elbow_r":
      return 0.26 * w;
    case "hip_l_flex":
    case "hip_r_flex":
      return -0.05 * w;
    case "knee_l":
    case "knee_r":
      return -0.1 * w;
    case "ankle_l":
    case "ankle_r":
      return 0.05 * w;
    default:
      return 0;
  }
}

/** Pose the rig from the simulation state: root from the free joint, each bone the product of its hinges. */
export function poseRig(g: LiveGinga, rig: Rig) {
  const q = g.data.qpos as Float64Array;
  for (const bone of Object.values(rig.bones)) bone.quaternion.identity();
  const style = effectiveGingaStyle(g);
  const gingaWeight = g.recover ? 0 : g.move ? Math.max(0, 1 - blend(g, g.kickT) * 1.6) : 1;
  const hipYOffset = style === "angola" ? -0.042 * gingaWeight : style === "regional" ? 0.012 * gingaWeight : 0;
  const hips = rig.bones.hips;
  hips.position.set(q[0], q[1] + hipYOffset, q[2]);
  hips.quaternion.set(q[4], q[5], q[6], q[3]).normalize();
  for (const { bone, joints } of g.bodies) {
    const quat = rig.bones[bone].quaternion.identity();
    for (const j of joints) {
      const joint = g.joints[j];
      const angle = q[joint.adr] + styleJointOffset(g, joint.name, gingaWeight, style);
      quat.multiply(hinge.setFromAxisAngle(axis.fromArray(joint.axis), angle));
    }
  }
  rig.object.updateMatrixWorld(true);
}
