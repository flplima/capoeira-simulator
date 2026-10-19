import { clip } from "./choreography";
import { wrap } from "./math";
import { MOVE_NETWORK, type Maneuver, type Move, type Side } from "./types";
import type { LiveGinga } from "./simulation";

// Where each leg can start a kick, as in ml/env.py: the base with that leg back.
const ENTRY_CENTRE: Record<Side, number> = { r: 0, l: 0.5 };

/** Whether a move can be asked for: the kicks always (when present in the reference), the dodges only with
 * their own network and reference. */
export function available(g: LiveGinga, move: Move) {
  return (MOVE_NETWORK[move] === "kicks" || !!g.files.policy.networks.esquiva) && !!clip(g, move, "r");
}

/** Whether a standing or floor displacement maneuver can be asked for. */
export function availableManeuver(g: LiveGinga, maneuver: Maneuver) {
  if (maneuver === "role") return g.available("cocorinha");
  if (maneuver === "passagem-de-negativa") return g.available("esquiva-baixa");
  return true;
}

/** Request a standing or floor displacement maneuver (troca-de-base, giro-de-base, role, passagem-de-negativa). */
export function requestManeuver(g: LiveGinga, type: Maneuver) {
  if (g.busy || !g.availableManeuver(type)) return null;
  const side: Side = wrap(g.phase) < 0.5 ? "r" : "l";
  if (type === "troca-de-base") {
    g.maneuver = { type, t: 0, duration: 1.35, side, startYaw: g.placement[2] };
    g.notify();
    return g.maneuver;
  }
  if (type === "giro-de-base") {
    g.maneuver = { type, t: 0, duration: 1.6, side, startYaw: g.placement[2] };
    g.notify();
    return g.maneuver;
  }
  const baseMove: Move = type === "role" ? "cocorinha" : "esquiva-baixa";
  const pending = g.request(baseMove);
  if (!pending) return null;
  g.maneuver = {
    type,
    t: 0,
    duration: clip(g, baseMove, pending.side).duration,
    side: pending.side,
    startYaw: g.placement[2],
  };
  g.notify();
  return g.maneuver;
}

/**
 * Ask for a kick. The leg and the moment come from ml/choose.py, which tried
 * every leg and entry in simulation for this tempo and point of the cycle;
 * without that table, the leg whose base comes first is used. Ignored while
 * another move is under way.
 */
export function request(g: LiveGinga, move: Move) {
  if (g.busy || !g.available(move)) return null;
  const choices = g.files.reference.choices;
  const table = choices?.moves[move];
  let choice: { side: Side; entry: number };
  if (choices && table) {
    const t = choices.tempos.reduce((best, x, i) => (Math.abs(x - g.bpm) < Math.abs(choices.tempos[best] - g.bpm) ? i : best), 0);
    choice = table[t][Math.floor(g.phase * choices.bins) % choices.bins];
  } else {
    const side = (["r", "l"] as Side[]).reduce((a, b) => (wrap(ENTRY_CENTRE[a] - g.phase) <= wrap(ENTRY_CENTRE[b] - g.phase) ? a : b));
    choice = { side, entry: ENTRY_CENTRE[side] };
  }
  g.pending = { move, side: choice.side, entry: wrap(choice.entry) };
  g.notify();
  return g.pending;
}
