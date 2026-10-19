import { target } from "./choreography";
import { beginRecover, contacts } from "./floor";
import { quatMatrix, wrap } from "./math";
import { MOVE_NETWORK } from "./types";
import type { LiveGinga } from "./simulation";

/** How long the body may lie on the floor before it is stood back up. */
const GET_UP_AFTER = 1.2;
/** How strongly the ginga's phase is pulled toward the music's bar each control step. */
const SYNC_GAIN = 0.08;
/** The pull never runs the cycle more than this much faster or slower: a jump in the reference topples the body. */
const SYNC_LIMIT = 0.1;

/** How fast the reference moves on: slower on the way up while the body lags behind it (GingaEnv.pace). */
export function pace(g: LiveGinga) {
  const floor = g.files.reference.floor;
  if (!g.recover || !floor || g.recover.t < floor.fallTime) return 1;
  const q = g.data.qpos as Float64Array;
  const ref = target(g, 0).q;
  const lag = Math.hypot(q[0] - ref[0], q[1] - ref[1], q[2] - ref[2]);
  return Math.min(Math.max(1 - (lag - floor.waitFrom) / floor.waitSpan, floor.minPace), 1);
}

/** Balance is lost (GingaEnv.trouble): a bad contact, the head low, or (outside an active move) the hips tipped, sunk or away. */
export function trouble(g: LiveGinga) {
  const q = g.data.qpos as Float64Array;
  const ref = target(g, 0).q;
  const { troubleTilt, troubleDrop } = g.files.reference.floor!;
  // The angle between the hips' up axis and the reference's (heading aside).
  const R = g.rot;
  quatMatrix(q[3], q[4], q[5], q[6], R);
  const up = [R[1], R[4], R[7]];
  quatMatrix(ref[3], ref[4], ref[5], ref[6], R);
  const dot = up[0] * R[1] + up[1] * R[4] + up[2] * R[7];
  const inMove = g.move !== null || g.maneuver !== null || g.sinceKick < 0.5;
  return (
    contacts(g).bad ||
    low(g, ref) ||
    (!inMove &&
      (Math.acos(Math.min(1, Math.max(-1, dot))) > troubleTilt ||
        q[1] < ref[1] - troubleDrop ||
        Math.hypot(q[0] - ref[0], q[2] - ref[2]) > 0.5))
  );
}

/** Hips or head too low for anything but a fall (GingaEnv.low), against the reference pose `ref`. */
export function low(g: LiveGinga, ref: ArrayLike<number>) {
  const { hips, head, below } = g.files.policy.low ?? { hips: 0.4, head: 0.8, below: 0.3 };
  const headBelow = g.move && MOVE_NETWORK[g.move] === "esquiva" ? below + 0.08 : below;
  (g.groundData.qpos as Float64Array).set(ref);
  g.mujoco.mj_kinematics(g.model, g.groundData);
  const refHead = (g.groundData.xpos as Float64Array)[3 * g.headId + 1];
  const q = g.data.qpos as Float64Array;
  const ownHead = (g.data.xpos as Float64Array)[3 * g.headId + 1];
  return q[1] < Math.min(hips, ref[1] - below) || ownHead < Math.min(head, refHead - headBelow);
}

/**
 * After a step: notice a loss of balance and hand over to the recovery network (or, without one, notice
 * the fall and stand the body up again after a moment), and follow the music's bar.
 */
export function settle(g: LiveGinga, barPhase: number | null) {
  const q = g.data.qpos as Float64Array;
  // A fall, once noticed, holds until the body is stood up: checking again with the ginga's rules while the
  // recovery is still under way cleared it every other step, and the stand-up timer kept starting over.
  if (g.fallen) {
    // Down until GET_UP_AFTER.
  } else if (g.canRecover) {
    if (g.recover) {
      // Down and up again: a failure only if the head hits the floor, or, once landed, the body is lost or stuck.
      const ref = target(g, 0).q;
      const { fallTime, lost, minPace, stuck } = g.files.reference.floor!;
      const landed = g.recover.t >= fallTime;
      const away = landed && Math.hypot(q[0] - ref[0], q[2] - ref[2]) > lost;
      g.stuckFor = landed && pace(g) <= minPace ? g.stuckFor + g.controlDt : 0;
      g.fallen = contacts(g).bad || away || q[1] < 0.05 || g.stuckFor >= stuck;
    } else if (!g.wasRecovering && trouble(g)) {
      beginRecover(g);
    }
  } else {
    g.fallen = contacts(g).bad || low(g, target(g, 0).q);
  }
  g.fallenFor = g.fallen ? g.fallenFor + g.controlDt : 0;
  // Stood up on the beat, if the music plays.
  if (g.fallenFor > GET_UP_AFTER) g.reset(barPhase ?? g.phase);
  if (barPhase !== null && !g.busy) {
    // Nudge the cycle toward the bar, the shortest way round (not while a kick is coming or under way).
    const error = wrap(barPhase - g.phase + 0.5) - 0.5;
    const limit = (SYNC_LIMIT * g.controlDt) / g.period;
    g.phase = wrap(g.phase + Math.min(Math.max(SYNC_GAIN * error, -limit), limit));
  }
}
