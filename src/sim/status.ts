import { dict, t } from "../i18n/lang";
import { MANEUVER_NAMES, MOVE_NAMES, MOVE_NETWORK, type Move, type Side } from "./types";
import type { LiveGinga } from "./simulation";

/** Which way a move goes, for the status line: a kick names the kicking leg (the one back in the base); the
 * esquiva from the base with the right leg back drops over the front (left) leg, to the left. */
export const which = (move: Move, side: Side) =>
  move === "esquiva" ? t(side === "r" ? "toLeft" : "toRight") : t(side === "r" ? "legRight" : "legLeft");

export function statusOf(g: LiveGinga) {
  const moveLabel = g.maneuver
    ? MANEUVER_NAMES[g.maneuver.type]
    : g.move
      ? `${g.feintMode && MOVE_NETWORK[g.move] === "kicks" ? t("feintOf", { name: MOVE_NAMES[g.move] }) : MOVE_NAMES[g.move]} · ${which(g.move, g.side)}`
      : null;
  const pendingLabel = g.pending
    ? g.maneuver
      ? t("statusWaiting", { name: MANEUVER_NAMES[g.maneuver.type] })
      : t("statusWaiting", { name: `${MOVE_NAMES[g.pending.move]} · ${which(g.pending.move, g.pending.side)}` })
    : null;
  return {
    time: g.phase * g.period,
    duration: g.period,
    keyframe: Math.floor(g.phase * 4),
    label: g.fallen
      ? t("statusFell")
      : g.recover
        ? g.recover.t < g.files.reference.floor!.fallTime
          ? t("statusFalling", { landing: dict().landings[g.recover.move] })
          : t("statusGettingUp", { from: dict().fromLandings[g.recover.move] })
        : pendingLabel
          ? pendingLabel
          : moveLabel
            ? moveLabel
            : t("statusCycle", { n: g.cycles + 1, bpm: Math.round(g.bpm) }),
    mirrored: false,
    resting: false,
  };
}
