import { blend, clip, target } from "./choreography";
import { contacts, floorClip } from "./floor";
import { quatMatrix, yawOf } from "./math";
import { EXTRA_MOVES } from "./types";
import type { LiveGinga } from "./simulation";

const tempoInput = (bpm: number) => (bpm - 95) / 35;

export function features(g: LiveGinga): number[] {
  const r = g.recover;
  const extraZeros = EXTRA_MOVES.map(() => 0);
  if (r) {
    const total = g.files.reference.floor!.fallTime + floorClip(g, r.move, r.side).duration;
    return [0, 0, 0, 0, 0, 1, r.move === "quatro" ? 1 : 0, r.side === "r" ? 1 : -1, r.t / total, r.move === "joelhos" ? 1 : 0, 0, ...extraZeros];
  }
  if (!g.move) return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...extraZeros];
  const duration = clip(g, g.move, g.side).duration;
  const kick = [g.move === "martelo" ? 1 : 0, g.move === "bencao" ? 1 : 0, g.side === "r" ? 1 : -1, g.kickT / duration, blend(g, g.kickT)];
  return [...kick, 0, 0, 0, 0, 0, g.move === "esquiva" ? 1 : 0, ...EXTRA_MOVES.map((m) => (g.move === m ? 1 : 0))];
}

/** The policy's input, exactly as ml/env.py builds it. */
export function observe(g: LiveGinga): Float64Array {
  const d = g.data;
  const q = d.qpos as Float64Array;
  const v = d.qvel as Float64Array;
  const xpos = d.xpos as Float64Array;
  const sites = d.site_xpos as Float64Array;
  const { nq, nv } = g.model;
  const o = g.obs;
  let n = 0;
  // Heading frame (yaw_matrix): world vectors into the body's heading.
  const yaw = yawOf(q[3], q[4], q[5], q[6]);
  const F = g.frame;
  F.fill(0);
  F[0] = F[8] = Math.cos(yaw);
  F[2] = -Math.sin(yaw);
  F[6] = Math.sin(yaw);
  F[4] = 1;
  const R = g.rot;
  quatMatrix(q[3], q[4], q[5], q[6], R);
  const { q: ref, feet: goal } = target(g, g.controlDt);
  const { feet } = contacts(g);
  const rotate = (x: number, y: number, z: number, scale: number) => {
    o[n++] = (F[0] * x + F[1] * y + F[2] * z) * scale;
    o[n++] = (F[3] * x + F[4] * y + F[5] * z) * scale;
    o[n++] = (F[6] * x + F[7] * y + F[8] * z) * scale;
  };

  o[n++] = Math.sin(2 * Math.PI * g.phase);
  o[n++] = Math.cos(2 * Math.PI * g.phase);
  o[n++] = q[1];
  // (frame @ R)[:, :2], row-major.
  for (let r = 0; r < 3; r++)
    for (let col = 0; col < 2; col++) o[n++] = F[3 * r] * R[col] + F[3 * r + 1] * R[3 + col] + F[3 * r + 2] * R[6 + col];
  for (let k = 7; k < nq; k++) o[n++] = q[k];
  for (let k = 6; k < nv; k++) o[n++] = 0.1 * v[k];
  rotate(v[0], v[1], v[2], 1);
  // Freejoint angular velocity v[3:6] is already in body-local coordinates.
  o[n++] = 0.2 * v[3];
  o[n++] = 0.2 * v[4];
  o[n++] = 0.2 * v[5];
  for (const id of g.effectorIds) rotate(xpos[3 * id] - q[0], xpos[3 * id + 1] - q[1], xpos[3 * id + 2] - q[2], 1);
  o[n++] = feet[0];
  o[n++] = feet[1];
  for (let k = 7; k < nq; k++) o[n++] = ref[k] - q[k];
  rotate(ref[0] - q[0], ref[1] - q[1], ref[2] - q[2], 1);
  g.soles.forEach(([heel, toe], f) => {
    const ox = goal[2 * f] - (sites[3 * heel] + sites[3 * toe]) / 2;
    const oz = goal[2 * f + 1] - (sites[3 * heel + 2] + sites[3 * toe + 2]) / 2;
    o[n++] = F[0] * ox + F[2] * oz;
    o[n++] = F[6] * ox + F[8] * oz;
  });
  o[n++] = tempoInput(g.bpm);
  for (const f of features(g)) if (n < o.length) o[n++] = f;
  if (n !== o.length) throw new Error(`observation has ${n} values, the policy expects ${o.length}`);
  return o;
}
