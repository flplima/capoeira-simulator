import { Vector3 } from "three";
import type { BoneName, Rig } from "../rig/rig";

/** Body mass of the mannequin, kg. */
export const BODY_MASS = 70;

/**
 * Segment masses and centres of mass, after de Leva (1996), "Adjustments to
 * Zatsiorsky-Seluyanov's segment inertia parameters" (male values). Each
 * segment is [bone, centre of mass in the bone's local space, fraction of body
 * mass]; the positions are the published proximal-to-distal ratios applied to
 * this rig's segment lengths. Fractions sum to 1.
 */
export const SEGMENTS: [BoneName, [number, number, number], number][] = [
  ["head", [0, 0.08, 0], 0.0694],
  ["chest", [0, 0.13, 0], 0.1596],
  ["spine", [0, 0.08, 0], 0.1633],
  ["hips", [0, 0, 0], 0.1117],
  ...(["l", "r"] as const).flatMap((s): [BoneName, [number, number, number], number][] => [
    [`upperarm_${s}`, [0, -0.29 * 0.577, 0], 0.0271],
    [`forearm_${s}`, [0, -0.25 * 0.457, 0], 0.0162],
    [`hand_${s}`, [0, -0.07, 0], 0.0061],
    [`thigh_${s}`, [0, -0.43 * 0.41, 0], 0.1416],
    [`shin_${s}`, [0, -0.41 * 0.446, 0], 0.0433],
    [`foot_${s}`, [0, -0.04, 0.06], 0.0137],
  ]),
];

const local = new Vector3();

/** World positions of every segment's centre of mass (matrices must be current). */
export function segmentPositions(rig: Rig, out: Vector3[] = SEGMENTS.map(() => new Vector3())): Vector3[] {
  SEGMENTS.forEach(([bone, at], i) => out[i].copy(local.set(...at)).applyMatrix4(rig.bones[bone].matrixWorld));
  return out;
}

/** Whole-body centre of mass in world space. */
export function centerOfMass(rig: Rig, out = new Vector3(), points = segmentPositions(rig)): Vector3 {
  out.set(0, 0, 0);
  SEGMENTS.forEach(([, , fraction], i) => out.addScaledVector(points[i], fraction));
  return out;
}

/**
 * Moment of inertia about the vertical axis through the centre of mass, kg·m².
 * Segments are point masses plus a constant for the trunk's own girth, which is
 * what keeps a tucked spin from becoming infinitely fast.
 */
export function verticalInertia(rig: Rig, points = segmentPositions(rig), com = centerOfMass(rig, new Vector3(), points)): number {
  let inertia = 0.9;
  SEGMENTS.forEach(([, , fraction], i) => {
    const dx = points[i].x - com.x;
    const dz = points[i].z - com.z;
    inertia += BODY_MASS * fraction * (dx * dx + dz * dz);
  });
  return inertia;
}
