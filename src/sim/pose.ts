import type { Rig } from "../rig/rig";
import { Quaternion, Vector3 } from "three";
import type { LiveGinga } from "./simulation";

const axis = new Vector3();
const hinge = new Quaternion();

/** Pose the rig from the simulation state: root from the free joint, each bone the product of its hinges. */
export function poseRig(g: LiveGinga, rig: Rig) {
  const q = g.data.qpos as Float64Array;
  for (const bone of Object.values(rig.bones)) bone.quaternion.identity();
  const hips = rig.bones.hips;
  hips.position.set(q[0], q[1], q[2]);
  hips.quaternion.set(q[4], q[5], q[6], q[3]).normalize();
  for (const { bone, joints } of g.bodies) {
    const quat = rig.bones[bone].quaternion.identity();
    for (const j of joints) {
      const joint = g.joints[j];
      quat.multiply(hinge.setFromAxisAngle(axis.fromArray(joint.axis), q[joint.adr]));
    }
  }
  rig.object.updateMatrixWorld(true);
}
