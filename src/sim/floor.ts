import { nlerp, placePose, placeXZ, yawOf } from "./math";
import type { Landing, Placement, Side } from "./types";
import type { LiveGinga } from "./simulation";

const GRAVITY = 9.81; // m/s² (env.GRAVITY)

export function floorClip(g: LiveGinga, move: Landing, side: Side) {
  return g.files.reference.floor!.clips[`${move}_${side}`];
}

/** The way up from a landing at time t, in its own frame (env.Floor.sample). */
export function sampleFloor(g: LiveGinga, move: Landing, side: Side, t: number) {
  const { fps } = g.files.reference.floor!;
  const clip = floorClip(g, move, side);
  const nq = g.model.nq;
  const frames = clip.qpos.length / nq;
  const x = Math.min(Math.max(t, 0), clip.duration) * fps;
  const k = Math.min(Math.floor(x), frames - 2);
  const a = x - k;
  const q = g.floorPose;
  for (let c = 0; c < nq; c++) q[c] = clip.qpos[k * nq + c] + (clip.qpos[(k + 1) * nq + c] - clip.qpos[k * nq + c]) * a;
  nlerp(clip.qpos, k * nq + 3, clip.qpos, (k + 1) * nq + 3, a, q, 3);
  for (let c = 0; c < 4; c++) g.floorFeet[c] = clip.feet[k * 4 + c] + (clip.feet[(k + 1) * 4 + c] - clip.feet[k * 4 + c]) * a;
  return { q, feet: g.floorFeet };
}

/** Going down from the pose balance was lost in to the landing, then the way up (Choreography._recover_target). */
export function recoverTarget(g: LiveGinga, ahead: number): { q: number[]; feet: number[] } {
  const r = g.recover!;
  const { fallTime } = g.files.reference.floor!;
  const t = r.t + ahead;
  const out = sampleFloor(g, r.move, r.side, Math.max(t - fallTime, 0));
  placePose(r.place, out.q, out.feet);
  if (t >= fallTime) return out;
  // Fast at first, settling into the landing, but never ahead of gravity: the hips drop no further than
  // they would falling freely from their speed at the start. All joints fold into the landing pose at once.
  const x = Math.min(t / fallTime, 1);
  const reach = 1 - (1 - x) ** 2;
  const q = out.q;
  const drop = r.onset[1] - q[1];
  const fallen = Math.max(-r.sink * t + 0.5 * GRAVITY * t * t, 0);
  const b = drop > 0.01 ? Math.min(reach, fallen / drop) : reach;
  const landing = q.slice(3, 7);
  for (let c = 0; c < q.length; c++) q[c] = r.onset[c] + (q[c] - r.onset[c]) * (g.armQpos.has(c) ? reach : b);
  nlerp(r.onset, 3, landing, 0, b, q, 3);
  // Blended joint by joint, the legs can swing through the floor on the way: the pose rises out of it.
  q[1] += sinking(g, q);
  return out;
}

/** How far (m) a pose's feet, knees or hands are below the floor (GingaEnv._sinking). */
export function sinking(g: LiveGinga, q: number[]): number {
  (g.groundData.qpos as Float64Array).set(q);
  g.mujoco.mj_kinematics(g.model, g.groundData);
  const sites = g.groundData.site_xpos as Float64Array;
  const { clear } = g.files.reference.floor!.ground;
  let low = 0;
  g.groundSites.forEach((id, i) => (low = Math.min(low, sites[3 * id + 1] - clear[i])));
  return -low;
}

/** Losing balance: land on hands and knees (going forwards, or bent far forward), in the queda de quatro
 * (backwards) or the negativa (sideways), as the body faces (Choreography.begin_recover, GingaEnv.lean). */
export function beginRecover(g: LiveGinga) {
  g.stuckFor = 0;
  const q = g.data.qpos as Float64Array;
  const v = g.data.qvel as Float64Array;
  const com = g.data.subtree_com as Float64Array;
  const sites = g.data.site_xpos as Float64Array;
  // Where the centre of mass is past the middle of the feet, plus the hips' drift.
  const lean = [3, 5].map((axis, i) => {
    const feet = g.soles.reduce((sum, [heel, toe]) => sum + (sites[3 * heel + axis - 3] + sites[3 * toe + axis - 3]) / 4, 0);
    return com[axis] - feet + 0.3 * v[2 * i];
  });
  const yaw = yawOf(q[3], q[4], q[5], q[6]);
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const sideWay = c * lean[0] - s * lean[1];
  const forward = s * lean[0] + c * lean[1];
  // How far the trunk is bent forward (GingaEnv.bent): the hips-to-head line from the vertical.
  const xpos = g.data.xpos as Float64Array;
  const up = [0, 1, 2].map((i) => xpos[3 * g.headId + i] - q[i]);
  const bent = Math.atan2(s * up[0] + c * up[2], up[1]);
  let move: Landing;
  let side: Side;
  if (bent > g.files.reference.floor!.bent) {
    move = "joelhos";
    side = g.phase < 0.25 || g.phase >= 0.75 ? "r" : "l";
  } else if (Math.abs(forward) >= Math.abs(sideWay)) {
    move = forward > 0 ? "joelhos" : "quatro";
    side = g.phase < 0.25 || g.phase >= 0.75 ? "r" : "l";
  } else {
    // Fold the leg that carries the weight: the foot nearest where the body is going over (negativa_r folds
    // the left leg).
    const feet = g.soles.map(([heel, toe]) => [0, 2].map((a) => (sites[3 * heel + a] + sites[3 * toe + a]) / 2));
    const mid = [0, 1].map((a) => (feet[0][a] + feet[1][a]) / 2);
    const [dl, dr] = feet.map((f) => Math.hypot(f[0] - mid[0] - lean[0], f[1] - mid[1] - lean[1]));
    move = "negativa";
    side = dl <= dr ? "r" : "l";
  }
  const landing = sampleFloor(g, move, side, 0).q;
  const norm = Math.hypot(lean[0], lean[1]);
  const tx = q[0] + (norm > 1e-6 ? (0.25 * lean[0]) / norm : 0);
  const tz = q[2] + (norm > 1e-6 ? (0.25 * lean[1]) / norm : 0);
  const place: Placement = [0, 0, yaw];
  const [lx, lz] = placeXZ(place, landing[0], landing[2]);
  place[0] = tx - lx;
  place[1] = tz - lz;
  g.move = null;
  g.pending = null;
  g.maneuver = null;
  g.pendingFeintGinga = false;
  g.recover = { move, side, t: 0, onset: Array.from(q), place, sink: v[1] };
  g.notify();
}

/** Arms braced (env.STRONG_ARMS) or relaxed (GingaEnv.brace): stiffness, critical damping and torque limit. */
export function brace(g: LiveGinga, on: boolean) {
  const strong = g.files.reference.floor?.strongArms;
  if (on === g.braced || !strong) return;
  g.braced = on;
  const m = g.model;
  const gain = m.actuator_gainprm as Float64Array;
  const bias = m.actuator_biasprm as Float64Array;
  const force = m.actuator_forcerange as Float64Array;
  const G = gain.length / m.nu;
  const B = bias.length / m.nu;
  if (!g.arms) {
    g.arms = [];
    for (let i = 0; i < m.nu; i++) {
      const name = m.actuator(i).name;
      const joint = name.startsWith("shoulder") ? "shoulder" : name.startsWith("elbow") ? "elbow" : null;
      if (!joint) continue;
      const [kp, kv, f] = [gain[G * i], bias[B * i + 2], force[2 * i + 1]];
      const [strongKp, strongF] = strong[joint];
      g.arms.push({ id: i, relaxed: [kp, kv, f], braced: [strongKp, kv * Math.sqrt(strongKp / kp), strongF] });
    }
  }
  for (const arm of g.arms) {
    const [kp, kv, f] = on ? arm.braced : arm.relaxed;
    gain[G * arm.id] = kp;
    bias[B * arm.id + 1] = -kp;
    bias[B * arm.id + 2] = kv;
    force[2 * arm.id] = -f;
    force[2 * arm.id + 1] = f;
  }
}

/** The way up starts where the body actually landed (Choreography.land). */
export function land(g: LiveGinga) {
  const r = g.recover!;
  const q = g.data.qpos as Float64Array;
  const landing = sampleFloor(g, r.move, r.side, 0).q;
  const [lx, lz] = placeXZ([0, 0, r.place[2]], landing[0], landing[2]);
  r.place[0] = q[0] - lx;
  r.place[1] = q[2] - lz;
}

export function contacts(g: LiveGinga): { feet: [number, number]; bad: boolean } {
  const feet: [number, number] = [0, 0];
  let bad = false;
  const list = g.data.contact;
  const n = g.data.ncon;
  for (let i = 0; i < n; i++) {
    const c = list.get(i)!;
    const other = c.geom1 === g.floor ? c.geom2 : c.geom2 === g.floor ? c.geom1 : -1;
    c.delete();
    if (other < 0) continue;
    const f = g.feetGeoms.indexOf(other);
    if (f >= 0) feet[f] = 1;
    else if (g.never.has(other) || (!g.recover && !g.allowed.has(other))) bad = true;
  }
  list.delete();
  return { feet, bad };
}
