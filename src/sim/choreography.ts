import { cyclePeriod } from "../music/toques";
import { floorClip, land, recoverTarget } from "./floor";
import { nlerp, placePose, placeXZ, smoothstep, unplaceXZ, wrap } from "./math";
import { feetAt, sampleReference } from "./reference";
import { MOVE_NETWORK, type Move, type Placement, type Side } from "./types";
import type { LiveGinga } from "./simulation";

/** Furthest (m) a kick follows its support foot away from the corner (env.ANCHOR_LIMIT). */
export const ANCHOR_LIMIT = 0.25;

export function clip(g: LiveGinga, move: Move, side: Side) {
  return g.files.reference.kicks.clips[`${move}_${side}`];
}

export function blend(g: LiveGinga, t: number) {
  const { blendIn, blendOut } = g.files.reference.kicks;
  const scale = g.feintMode && g.move && MOVE_NETWORK[g.move] === "kicks" ? 0.45 : 1;
  return scale * smoothstep(t / blendIn) * smoothstep((clip(g, g.move!, g.side).duration - t) / blendOut);
}

/** Reference pose and feet targets `ahead` seconds from now, in the world (Choreography.target). */
export function target(g: LiveGinga, ahead: number): { q: number[]; feet: number[] } {
  if (g.recover) return recoverTarget(g, ahead);
  const out = ownTarget(g, ahead);
  placePose(g.placement, out.q, out.feet);
  return out;
}

/** The ginga or kick in the triangle's own frame (Choreography._target). */
export function ownTarget(g: LiveGinga, ahead: number): { q: number[]; feet: number[] } {
  if (!g.move) {
    const phase = g.phase + ahead / g.period;
    return { q: sampleReference(g, phase, g.bpm, g.refNext), feet: feetAt(g, phase, g.bpm, g.refFeet) };
  }
  const base = sampleReference(g, g.phase, g.bpm, g.ginga);
  const gf = feetAt(g, g.phase, g.bpm, g.refFeet);
  const kick = clip(g, g.move, g.side);
  const fps = kick.fps ?? g.files.reference.kicks.fps!;
  const t = g.kickT + ahead;
  const nq = g.model.nq;
  const frames = kick.qpos.length / nq;
  const x = Math.min(Math.max(t, 0), kick.duration) * fps;
  const k = Math.min(Math.floor(x), frames - 2);
  const a = x - k;
  const kq = g.kickPose;
  for (let c = 0; c < nq; c++) kq[c] = kick.qpos[k * nq + c] + (kick.qpos[(k + 1) * nq + c] - kick.qpos[k * nq + c]) * a;
  nlerp(kick.qpos, k * nq + 3, kick.qpos, (k + 1) * nq + 3, a, kq, 3);
  kq[0] += g.anchor[0];
  kq[2] += g.anchor[1];
  for (let c = 0; c < 4; c++) g.kickFeet[c] = kick.feet[k * 4 + c] + (kick.feet[(k + 1) * 4 + c] - kick.feet[k * 4 + c]) * a + g.anchor[c % 2];
  const b = blend(g, t);
  const q = g.refNext;
  for (let c = 0; c < nq; c++) q[c] = base[c] + (kq[c] - base[c]) * b;
  nlerp(base, 3, kq, 3, b, q, 3);
  for (let c = 0; c < 4; c++) gf[c] += (g.kickFeet[c] - gf[c]) * b;
  return { q, feet: gf };
}

/** World xz of the support foot (0 = left foot while right leg is back in 0..0.5, 1 = right foot in 0.5..1). */
export function supportFootXZ(g: LiveGinga, supportIndex?: 0 | 1): [number, number] {
  const idx = supportIndex ?? (wrap(g.phase) < 0.5 ? 0 : 1);
  const [heel, toe] = g.soles[idx];
  const sites = g.data.site_xpos as Float64Array;
  return [(sites[3 * heel] + sites[3 * toe]) / 2, (sites[3 * heel + 2] + sites[3 * toe + 2]) / 2];
}

/** Rotate the humanoid and its reference placement around a floor pivot (px, pz) by dYaw (radians). */
export function rotateAroundPivot(g: LiveGinga, dYaw: number, px: number, pz: number) {
  if (Math.abs(dYaw) < 1e-7) return;
  const q = g.data.qpos as Float64Array;
  const v = g.data.qvel as Float64Array;
  const c = Math.cos(dYaw);
  const s = Math.sin(dYaw);
  const dx = q[0] - px;
  const dz = q[2] - pz;
  q[0] = px + c * dx + s * dz;
  q[2] = pz - s * dx + c * dz;
  const hw = Math.cos(dYaw / 2);
  const hy = Math.sin(dYaw / 2);
  const [w, x, y, z] = [q[3], q[4], q[5], q[6]];
  q[3] = hw * w - hy * y;
  q[4] = hw * x + hy * z;
  q[5] = hw * y + hy * w;
  q[6] = hw * z - hy * x;
  const vx = v[0];
  const vz = v[2];
  v[0] = c * vx + s * vz;
  v[2] = -s * vx + c * vz;
  const pdx = g.placement[0] - px;
  const pdz = g.placement[1] - pz;
  g.placement = [px + c * pdx + s * pdz, pz - s * pdx + c * pdz, g.placement[2] + dYaw];
  g.mujoco.mj_forward(g.model, g.data);
}

/** Apply continuous navigation (walk/strafe/turn/orbit) while gingando. */
export function stepNavigation(g: LiveGinga, dt: number) {
  let fwd = g.nav.forward || (g.nudgeInput.timer > 0 ? g.nudgeInput.forward : 0);
  let side = g.nav.side || (g.nudgeInput.timer > 0 ? g.nudgeInput.side : 0);
  let turn = g.nav.turn || (g.nudgeInput.timer > 0 ? g.nudgeInput.turn : 0);
  if (g.nudgeInput.timer > 0) g.nudgeInput.timer = Math.max(0, g.nudgeInput.timer - dt);
  if (g.nav.orbit) {
    const q = g.data.qpos as Float64Array;
    const dist = Math.hypot(q[0], q[2]);
    const desiredYaw = dist > 0.25 ? Math.atan2(-q[0], -q[2]) : g.placement[2] + 0.65 * dt;
    const yawErr = Math.atan2(Math.sin(desiredYaw - g.placement[2]), Math.cos(desiredYaw - g.placement[2]));
    if (!side) side = 0.75;
    if (!fwd) fwd = Math.max(-0.5, Math.min(0.5, (dist - 1.0) * 0.8));
    if (!turn) turn = Math.max(-1, Math.min(1, yawErr * 1.8));
  }
  if (fwd === 0 && side === 0 && turn === 0) return;
  if (turn !== 0) {
    const swing = Math.sin(2 * Math.PI * g.phase) ** 2;
    const [px, pz] = supportFootXZ(g);
    rotateAroundPivot(g, turn * 1.15 * swing * dt, px, pz);
  }
  if (fwd !== 0 || side !== 0) {
    const base = reanchor(g);
    const yaw = base[2];
    const leadFwd = 0.055 * fwd;
    const leadSide = 0.05 * side;
    const wx = Math.sin(yaw) * leadFwd + Math.cos(yaw) * leadSide;
    const wz = Math.cos(yaw) * leadFwd - Math.sin(yaw) * leadSide;
    g.placement = [base[0] + wx, base[1] + wz, yaw];
  }
}

/** Move the choreography on by dt (Choreography.advance). */
export function advanceChoreography(g: LiveGinga, dt: number) {
  if (g.recover) {
    const r = g.recover;
    const { fallTime } = g.files.reference.floor!;
    const falling = r.t < fallTime;
    r.t += dt;
    if (falling && r.t >= fallTime) land(g);
    if (r.t >= fallTime + floorClip(g, r.move, r.side).duration) {
      // Up in the base: the ginga carries on from here, its triangle where the body stands.
      g.placement = [...r.place];
      g.phase = floorClip(g, r.move, r.side).phase;
      g.recover = null;
      g.notify();
    }
    return;
  }
  if (g.move) {
    g.kickT += dt;
    if (g.maneuver?.type === "role" && g.kickT >= 0.25 && g.kickT <= 1.55) {
      const w = 1 - Math.cos((2 * Math.PI * (g.kickT - 0.25)) / 1.3);
      const dir = g.side === "r" ? 1 : -1;
      const [px, pz] = supportFootXZ(g, g.side === "r" ? 0 : 1);
      rotateAroundPivot(g, dir * ((1.5 * Math.PI) / 1.3) * w * dt, px, pz);
      const yaw = g.placement[2];
      g.placement = [
        g.placement[0] + Math.cos(yaw) * dir * 0.22 * w * dt,
        g.placement[1] - Math.sin(yaw) * dir * 0.22 * w * dt,
        yaw,
      ];
    } else if (g.maneuver?.type === "passagem-de-negativa" && g.kickT >= 0.3 && g.kickT <= 1.35) {
      const w = 1 - Math.cos((2 * Math.PI * (g.kickT - 0.3)) / 1.05);
      const dir = g.side === "r" ? -1 : 1;
      const [px, pz] = supportFootXZ(g, g.side === "r" ? 0 : 1);
      rotateAroundPivot(g, dir * ((0.5 * Math.PI) / 1.05) * w * dt, px, pz);
      const yaw = g.placement[2];
      g.placement = [
        g.placement[0] + Math.sin(yaw) * 0.14 * w * dt,
        g.placement[1] + Math.cos(yaw) * 0.14 * w * dt,
        yaw,
      ];
    }
    if (g.kickT >= clip(g, g.move, g.side).duration) {
      if (g.maneuver?.type === "passagem-de-negativa") g.phase = wrap(g.phase + 0.5);
      g.placement = reanchor(g);
      g.move = null;
      g.maneuver = null;
      g.notify();
    }
    return;
  }
  if (g.maneuver && !g.pending) {
    const m = g.maneuver;
    m.t += dt;
    if (m.type === "troca-de-base") {
      const w = 0.5 * (1 - Math.cos((2 * Math.PI * m.t) / m.duration));
      const effPeriod = cyclePeriod(Math.min(130, g.bpm + 38 * w));
      const next = g.phase + dt / effPeriod;
      if (next >= 1) g.cycles++;
      g.phase = wrap(next);
      const base = reanchor(g);
      const yaw = base[2];
      g.placement = [base[0] + Math.sin(yaw) * 0.035 * w, base[1] + Math.cos(yaw) * 0.035 * w, yaw];
    } else if (m.type === "giro-de-base") {
      const w = 1 - Math.cos((2 * Math.PI * m.t) / m.duration);
      const dir = m.side === "r" ? 1 : -1;
      const [px, pz] = supportFootXZ(g);
      rotateAroundPivot(g, dir * ((2 * Math.PI) / m.duration) * w * dt, px, pz);
      const next = g.phase + dt / g.period;
      if (next >= 1) g.cycles++;
      g.phase = wrap(next);
      const base = reanchor(g);
      g.placement = [base[0], base[1], g.placement[2]];
    }
    if (m.t >= m.duration) {
      g.placement = reanchor(g);
      g.maneuver = null;
      g.notify();
    }
    return;
  }
  const step = dt / g.period;
  if (g.pending && wrap(g.pending.entry - g.phase) <= step) {
    g.phase = g.pending.entry;
    g.move = g.pending.move;
    g.side = g.pending.side;
    g.kickT = 0;
    g.pending = null;
    g.anchor = supportAnchor(g);
    g.notify();
    return;
  }
  if (g.pendingFeintGinga && Math.abs((g.phase % 0.5) - 0.25) <= Math.max(step * 2, 0.025)) {
    g.phase = wrap(1 - g.phase);
    g.pendingFeintGinga = false;
    g.notify();
    return;
  }
  const next = g.phase + step;
  if (next >= 1) g.cycles++;
  g.phase = wrap(next);
  stepNavigation(g, dt);
}

/** Where the ginga's triangle goes when a kick or an esquiva ends: under the feet as they stand, without turning
 * (Choreography.reanchor). */
export function reanchor(g: LiveGinga): Placement {
  const local = feetAt(g, g.phase, g.bpm, [0, 0, 0, 0]);
  const sites = g.data.site_xpos as Float64Array;
  const [l0x, l0z] = placeXZ(g.placement, local[0], local[1]);
  const [l1x, l1z] = placeXZ(g.placement, local[2], local[3]);
  let fx = 0;
  let fz = 0;
  for (const [heel, toe] of g.soles) {
    fx += (sites[3 * heel] + sites[3 * toe]) / 4;
    fz += (sites[3 * heel + 2] + sites[3 * toe + 2]) / 4;
  }
  return [g.placement[0] + fx - (l0x + l1x) / 2, g.placement[1] + fz - (l0z + l1z) / 2, g.placement[2]];
}

/** How far the support foot stands from where the kick plans it, capped at ANCHOR_LIMIT. */
export function supportAnchor(g: LiveGinga): [number, number] {
  const support = g.side === "r" ? 0 : 1;
  const planned = clip(g, g.move!, g.side).feet;
  const [heel, toe] = g.soles[support];
  const sites = g.data.site_xpos as Float64Array;
  const [fx, fz] = unplaceXZ(g.placement, (sites[3 * heel] + sites[3 * toe]) / 2, (sites[3 * heel + 2] + sites[3 * toe + 2]) / 2);
  const dx = fx - planned[2 * support];
  const dz = fz - planned[2 * support + 1];
  const scale = Math.min(1, ANCHOR_LIMIT / (Math.hypot(dx, dz) || Infinity));
  return [dx * scale, dz * scale];
}
