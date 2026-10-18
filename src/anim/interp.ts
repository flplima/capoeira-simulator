import { CH, CHANNELS, CHANNEL_COUNT, restPose, type Pose } from "../data/channels";
import type { Clip } from "../data/compile";

// Monotone cubic (PCHIP) interpolation per channel. Unlike Catmull-Rom it never
// overshoots: two equal keyframes give a true hold, and an extreme keyframe —
// the top of a kick — is reached exactly instead of being sailed past.

interface Point {
  t: number;
  v: Pose;
}

function point(clip: Clip, k: number): Point {
  const n = clip.times.length;
  if (!clip.loop) {
    const c = Math.min(Math.max(k, 0), n - 1);
    return { t: clip.times[c], v: clip.frames[c] };
  }
  const wraps = Math.floor(k / n);
  const kk = k - wraps * n;
  return { t: clip.times[kk] + wraps * clip.duration, v: clip.frames[kk] };
}

function tangent(prev: Point, cur: Point, next: Point, c: number): number {
  const h0 = cur.t - prev.t;
  const h1 = next.t - cur.t;
  if (h0 <= 0 || h1 <= 0) return 0;
  const d0 = (cur.v[c] - prev.v[c]) / h0;
  const d1 = (next.v[c] - cur.v[c]) / h1;
  if (d0 * d1 <= 0) return 0;
  const w1 = 2 * h1 + h0;
  const w2 = h1 + 2 * h0;
  return (w1 + w2) / (w1 / d0 + w2 / d1);
}

/** Index of the keyframe at or before `t` (t already wrapped/clamped). */
export function segmentAt(clip: Clip, t: number): number {
  let i = 0;
  while (i + 1 < clip.times.length && clip.times[i + 1] <= t) i++;
  return i;
}

export function sampleClip(clip: Clip, time: number, out: Pose = restPose()): Pose {
  const t = clip.loop
    ? ((time % clip.duration) + clip.duration) % clip.duration
    : Math.min(Math.max(time, 0), clip.duration);
  const i = segmentAt(clip, t);
  const p0 = point(clip, i - 1);
  const p1 = point(clip, i);
  const p2 = point(clip, i + 1);
  const p3 = point(clip, i + 2);
  const h = p2.t - p1.t;
  if (h <= 0) {
    out.set(p1.v);
    return out;
  }
  const s = (t - p1.t) / h;
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;
  for (let c = 0; c < CHANNEL_COUNT; c++) {
    const m1 = tangent(p0, p1, p2, c);
    const m2 = tangent(p1, p2, p3, c);
    out[c] = h00 * p1.v[c] + h10 * h * m1 + h01 * p2.v[c] + h11 * h * m2;
  }
  applyBaked(clip, t, out);
  return out;
}

/** Standard gravity, m/s². */
export const G = 9.81;
const LIFT = CH["root.lift"];
const YAW = CH["root.yaw"];

/** Overrides from bakeDynamics: ballistic height in flight, inertia-timed spins. */
function applyBaked(clip: Clip, t: number, out: Pose) {
  for (const [t0, t1] of clip.airborne ?? []) if (t > t0 && t < t1) out[LIFT] = 0.5 * G * (t - t0) * (t1 - t);
  for (const spin of clip.spins ?? []) {
    if (t < spin.t0 || t > spin.t1) continue;
    const x = ((t - spin.t0) / (spin.t1 - spin.t0)) * (spin.yaw.length - 1);
    const i = Math.min(Math.floor(x), spin.yaw.length - 2);
    out[YAW] = spin.yaw[i] + (spin.yaw[i + 1] - spin.yaw[i]) * (x - i);
  }
}

// Proximal-to-distal sequencing: in a real kick or swing the hips lead and each
// joint further out follows a beat later, which is what makes the foot whip.
// Each channel is sampled LAG seconds behind per level of distance from the
// pelvis. The curves are the same, so every authored pose is still reached.
const LAG = 0.022;
const LEVEL: Record<string, number> = { root: 0, torso: 1, hip: 1, shoulder: 2, head: 2, knee: 2, elbow: 3, ankle: 3 };
const CHANNEL_LEVEL = CHANNELS.map((c) => LEVEL[c.joint.replace(/_[lr]$/, "")]);
const MAX_LEVEL = Math.max(...CHANNEL_LEVEL);
const lagged = Array.from({ length: MAX_LEVEL + 1 }, () => restPose());

export function sampleWithLag(clip: Clip, time: number, out: Pose = restPose()): Pose {
  for (let level = 0; level <= MAX_LEVEL; level++) sampleClip(clip, time - level * LAG, lagged[level]);
  for (let c = 0; c < CHANNEL_COUNT; c++) out[c] = lagged[CHANNEL_LEVEL[c]][c];
  return out;
}
