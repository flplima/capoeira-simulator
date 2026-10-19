import type { Placement } from "./types";

// ---- small math helpers (row-major 3×3, quaternions w x y z) -------------

export function yawOf(w: number, x: number, y: number, z: number) {
  return Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y));
}

export function quatMatrix(w: number, x: number, y: number, z: number, out: number[]) {
  out[0] = 1 - 2 * (y * y + z * z);
  out[1] = 2 * (x * y - w * z);
  out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z);
  out[4] = 1 - 2 * (x * x + z * z);
  out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y);
  out[7] = 2 * (y * z + w * x);
  out[8] = 1 - 2 * (x * x + y * y);
}

export function nlerp(a: ArrayLike<number>, ai: number, b: ArrayLike<number>, bi: number, t: number, out: number[], oi: number) {
  const sign = a[ai] * b[bi] + a[ai + 1] * b[bi + 1] + a[ai + 2] * b[bi + 2] + a[ai + 3] * b[bi + 3] < 0 ? -1 : 1;
  let n = 0;
  for (let k = 0; k < 4; k++) {
    out[oi + k] = a[ai + k] + (sign * b[bi + k] - a[ai + k]) * t;
    n += out[oi + k] ** 2;
  }
  n = Math.sqrt(n);
  for (let k = 0; k < 4; k++) out[oi + k] /= n;
}

export const wrap = (x: number) => ((x % 1) + 1) % 1;

export const smoothstep = (x: number) => {
  const t = Math.min(Math.max(x, 0), 1);
  return t * t * (3 - 2 * t);
};

export function placeXZ(p: Placement, x: number, z: number): [number, number] {
  const c = Math.cos(p[2]);
  const s = Math.sin(p[2]);
  return [c * x + s * z + p[0], -s * x + c * z + p[1]];
}

export function unplaceXZ(p: Placement, x: number, z: number): [number, number] {
  const c = Math.cos(p[2]);
  const s = Math.sin(p[2]);
  const dx = x - p[0];
  const dz = z - p[1];
  return [c * dx - s * dz, s * dx + c * dz];
}

/** A reference pose and its feet points → world, in place (env.place_pose; velocities aren't needed here). */
export function placePose(p: Placement, q: number[], feet: number[]) {
  [q[0], q[2]] = placeXZ(p, q[0], q[2]);
  const hw = Math.cos(p[2] / 2);
  const hy = Math.sin(p[2] / 2);
  const [w, x, y, z] = [q[3], q[4], q[5], q[6]];
  q[3] = hw * w - hy * y;
  q[4] = hw * x + hy * z;
  q[5] = hw * y + hy * w;
  q[6] = hw * z - hy * x;
  for (let f = 0; f < 2; f++) [feet[2 * f], feet[2 * f + 1]] = placeXZ(p, feet[2 * f], feet[2 * f + 1]);
}
