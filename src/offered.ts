import type { LiveGinga } from "./sim/simulation";
import type { Maneuver, Move } from "./sim/types";

/**
 * What the app offers a visitor, in the order of the buttons and of the number keys (1 to n). The bar: at
 * least nine tries in ten end standing back in the ginga over the three toques together, none under 85 %,
 * and the body reads as the move. `scripts/measure-moves.ts` counts that for every move and maneuver the
 * networks know, so the next export shows which one earns its way back here.
 */
export const OFFERED = [
  { kind: "move", id: "meia-lua-de-frente", topic: "meia-lua-de-frente" },
  { kind: "move", id: "joelhada", topic: "joelhada" },
  { kind: "move", id: "esquiva", topic: "esquiva-lateral" },
  { kind: "move", id: "cocorinha", topic: "cocorinha" },
  { kind: "maneuver", id: "giro-de-base", topic: "giro-de-base" },
] as const satisfies readonly ({ kind: "move"; id: Move; topic: string } | { kind: "maneuver"; id: Maneuver; topic: string })[];

export type Offered = (typeof OFFERED)[number];

/** What a number key asks for: 1 is the first button, in the order of the buttons. */
export const offeredForKey = (key: string): Offered | undefined => (/^[1-9]$/.test(key) ? OFFERED[Number(key) - 1] : undefined);

/** Ask the body for it; null when it is busy. */
export const ask = (sim: LiveGinga, offered: Offered) =>
  offered.kind === "move" ? sim.request(offered.id) : sim.requestManeuver(offered.id);
