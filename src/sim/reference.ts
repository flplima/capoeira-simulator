import { nlerp, wrap } from "./math";
import type { LiveGinga } from "./simulation";

export function neighbours(g: LiveGinga, bpm: number): [number, number, number] {
  const t = g.files.reference.tempos;
  const b = Math.min(Math.max(bpm, t[0]), t[t.length - 1]);
  let j = t.findIndex((x) => x >= b);
  if (j < 0) j = t.length - 1;
  const i = Math.max(j - 1, 0);
  return [i, j, i === j ? 0 : (b - t[i]) / (t[j] - t[i])];
}

export function sampleReference(g: LiveGinga, phase: number, bpm: number, out: number[]) {
  const { samples, nq, qpos } = g.files.reference;
  const [i, j, w] = neighbours(g, bpm);
  const x = wrap(phase) * samples;
  const k = Math.floor(x) % samples;
  const k1 = (k + 1) % samples;
  const a = x - Math.floor(x);
  const pair: number[][] = [];
  for (const t of [i, j]) {
    const table = qpos[t];
    const q = new Array(nq);
    for (let c = 0; c < nq; c++) q[c] = table[k * nq + c] + (table[k1 * nq + c] - table[k * nq + c]) * a;
    nlerp(table, k * nq + 3, table, k1 * nq + 3, a, q, 3);
    pair.push(q);
  }
  for (let c = 0; c < nq; c++) out[c] = pair[0][c] + (pair[1][c] - pair[0][c]) * w;
  nlerp(pair[0], 3, pair[1], 3, w, out, 3);
  return out;
}

export function feetAt(g: LiveGinga, phase: number, bpm: number, out: number[]) {
  const { samples, feet } = g.files.reference;
  const [i, j, w] = neighbours(g, bpm);
  const x = wrap(phase) * samples;
  const k = Math.floor(x) % samples;
  const k1 = (k + 1) % samples;
  const a = x - Math.floor(x);
  for (let c = 0; c < 4; c++) {
    const fi = feet[i][k * 4 + c] + (feet[i][k1 * 4 + c] - feet[i][k * 4 + c]) * a;
    const fj = feet[j][k * 4 + c] + (feet[j][k1 * 4 + c] - feet[j][k * 4 + c]) * a;
    out[c] = fi + (fj - fi) * w;
  }
  return out;
}
