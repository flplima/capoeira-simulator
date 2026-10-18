import { CH } from "../data/channels";
import type { Clip } from "../data/compile";
import { sampleClip } from "../anim/interp";
import { applyPose, buildRig, type Rig } from "../rig/rig";
import { verticalInertia } from "./mass";

const LIFT = CH["root.lift"];
const YAW = CH["root.yaw"];

/** Turns smaller than this between two keyframes are left as authored. */
const MIN_SPIN = 45;
const SPIN_SAMPLES = 64;
/**
 * How strongly the spin rate follows 1/inertia. 1 would be free rotation with
 * conserved angular momentum; on the ground the supporting foot and hands
 * exchange torque with the floor, so only part of that effect shows.
 */
const MOMENTUM_EXPONENT = 0.7;

let shared: Rig | null = null;

/**
 * Precomputes the parts of a clip that follow physics rather than the keyframes:
 *
 * - **Flight.** A stretch of keyframes with `root.lift` > 0, bracketed by
 *   keyframes on the floor, is a jump: its height becomes the ballistic arc for
 *   that flight time under gravity, h(t) = g/2 · (t − t₀)(t₁ − t).
 * - **Spins.** Between two keyframes that turn the body more than 45°, the yaw
 *   rate is redistributed by the moment of inertia about the vertical axis —
 *   faster while the limbs are tucked, slower while a leg is out — keeping the
 *   total turn and the keyframe times exactly as authored.
 */
export function bakeDynamics(clip: Clip, rig: Rig = (shared ??= buildRig())): Clip {
  clip.airborne = undefined;
  clip.spins = undefined;

  const n = clip.times.length;
  const airborne: [number, number][] = [];
  for (let k = 1; k < n - 1; k++) {
    if (!(clip.frames[k][LIFT] > 0) || clip.frames[k - 1][LIFT] > 0) continue;
    let end = k;
    while (end < n && clip.frames[end][LIFT] > 0) end++;
    if (end < n) airborne.push([clip.times[k - 1], clip.times[end]]);
  }

  const spins: { t0: number; t1: number; yaw: Float64Array }[] = [];
  const segments = clip.loop ? n : n - 1;
  for (let k = 0; k < segments; k++) {
    const t0 = clip.times[k];
    const t1 = k + 1 < n ? clip.times[k + 1] : clip.duration;
    const y0 = clip.frames[k][YAW];
    const y1 = clip.frames[(k + 1) % n][YAW];
    if (Math.abs(y1 - y0) < MIN_SPIN) continue;

    const authored = new Float64Array(SPIN_SAMPLES + 1);
    const inertia = new Float64Array(SPIN_SAMPLES + 1);
    for (let s = 0; s <= SPIN_SAMPLES; s++) {
      const pose = sampleClip(clip, t0 + ((t1 - t0) * s) / SPIN_SAMPLES);
      authored[s] = pose[YAW];
      pose[YAW] = 0;
      applyPose(rig, pose);
      inertia[s] = verticalInertia(rig);
    }
    // At the end of a loop, sampling wraps to frame 0; pin the last sample to the target turn.
    authored[SPIN_SAMPLES] = y1;
    const mean = inertia.reduce((a, b) => a + b, 0) / inertia.length;
    const rates = new Float64Array(SPIN_SAMPLES);
    let total = 0;
    for (let s = 0; s < SPIN_SAMPLES; s++) {
      const weight = (mean / ((inertia[s] + inertia[s + 1]) / 2)) ** MOMENTUM_EXPONENT;
      rates[s] = (authored[s + 1] - authored[s]) * weight;
      total += rates[s];
    }
    if (Math.abs(total) < 1e-9) continue;
    const yaw = new Float64Array(SPIN_SAMPLES + 1);
    yaw[0] = y0;
    for (let s = 0; s < SPIN_SAMPLES; s++) yaw[s + 1] = yaw[s] + (rates[s] * (y1 - y0)) / total;
    spins.push({ t0, t1, yaw });
  }

  clip.airborne = airborne.length ? airborne : undefined;
  clip.spins = spins.length ? spins : undefined;
  return clip;
}
