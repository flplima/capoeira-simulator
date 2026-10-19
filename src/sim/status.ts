import { FROM_LANDING, LANDING_NAMES, MANEUVER_NAMES, MOVE_NAMES, MOVE_NETWORK, type Move, type Side } from "./types";
import type { LiveGinga } from "./simulation";

/** Which way a move goes, for the status line: a kick names the kicking leg (the one back in the base); the
 * esquiva from the base with the right leg back drops over the front (left) leg, to the left. */
export const which = (move: Move, side: Side) =>
  move === "esquiva" ? `para a ${side === "r" ? "esquerda" : "direita"}` : `perna ${side === "r" ? "direita" : "esquerda"}`;

export function statusOf(g: LiveGinga) {
  const moveLabel = g.maneuver
    ? MANEUVER_NAMES[g.maneuver.type]
    : g.move
      ? `${g.feintMode && MOVE_NETWORK[g.move] === "kicks" ? "Finta de " : ""}${MOVE_NAMES[g.move]} · ${which(g.move, g.side)}`
      : null;
  const pendingLabel = g.pending
    ? g.maneuver
      ? `${MANEUVER_NAMES[g.maneuver.type]} · esperando o momento`
      : `${MOVE_NAMES[g.pending.move]} · ${which(g.pending.move, g.pending.side)} · esperando o momento`
    : null;
  return {
    time: g.phase * g.period,
    duration: g.period,
    keyframe: Math.floor(g.phase * 4),
    label: g.fallen
      ? "física em tempo real · caiu — recomeçando"
      : g.recover
        ? g.recover.t < g.files.reference.floor!.fallTime
          ? `caindo · ${LANDING_NAMES[g.recover.move]}`
          : `levantando · ${FROM_LANDING[g.recover.move]}`
        : pendingLabel
          ? pendingLabel
          : moveLabel
            ? moveLabel
            : `física em tempo real · ciclo ${g.cycles + 1} · ${Math.round(g.bpm)} bpm`,
    mirrored: false,
    resting: false,
  };
}
