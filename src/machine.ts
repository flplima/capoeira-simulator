import { assign, emit, fromPromise, setup } from "xstate";
import { fetchText } from "./data/load";
import { loadTexts, type MovementText } from "./data/text";
import { OFFERED, type Offered } from "./offered";
import { DEFAULT_TOQUE, MAX_BPM, MIN_BPM, TOQUES } from "./music/toques";

export const SPEEDS = [0.25, 0.5, 0.75, 1, 1.25] as const;

/** The info tabs: the ginga and what the app offers (src/offered.ts); each names public/data/movements/<id>.yaml. */
export type Topic = "ginga" | Offered["topic"];
export const TOPICS: Topic[] = ["ginga", ...OFFERED.map((o) => o.topic)];

interface Context {
  /** How to do each trained movement, for the info panel. */
  texts: Partial<Record<Topic, MovementText>>;
  /** The movement the info panel shows: the ginga, or the move last asked for. */
  topic: Topic;
  speed: number;
  /** Berimbau toque and its tempo; the simulated ginga follows the tempo. */
  toque: string;
  bpm: number;
  /** Whether the berimbau is sounding. */
  music: boolean;
}

type Event =
  | { type: "TOGGLE_PLAY" }
  | { type: "SET_SPEED"; speed: number }
  | { type: "FASTER" }
  | { type: "SLOWER" }
  | { type: "STEP" }
  | { type: "SHOW"; topic: Topic }
  | { type: "SET_TOQUE"; id: string }
  | { type: "SET_BPM"; bpm: number }
  | { type: "TOGGLE_MUSIC" };

/** Commands for the animation loop, which owns the simulation. */
export type Emitted = { type: "step" };

const shiftSpeed = (speed: number, by: 1 | -1) => {
  const i = SPEEDS.findIndex((s) => s >= speed);
  return SPEEDS[Math.min(Math.max((i < 0 ? SPEEDS.length - 1 : i) + by, 0), SPEEDS.length - 1)];
};

export const capoeiraMachine = setup({
  types: {
    context: {} as Context,
    events: {} as Event,
    emitted: {} as Emitted,
  },
  actors: {
    // The texts are a nicety: the simulation runs without them.
    loadTexts: fromPromise(() => loadTexts(fetchText, TOPICS).catch(() => ({}))),
  },
}).createMachine({
  id: "capoeira",
  context: {
    texts: {},
    topic: "ginga",
    speed: 1,
    toque: DEFAULT_TOQUE.id,
    bpm: DEFAULT_TOQUE.bpm,
    music: false,
  },
  invoke: {
    src: "loadTexts",
    onDone: { actions: assign({ texts: ({ event }) => event.output }) },
  },
  initial: "playing",
  on: {
    SET_SPEED: { actions: assign({ speed: ({ event }) => event.speed }) },
    FASTER: { actions: assign({ speed: ({ context }) => shiftSpeed(context.speed, 1) }) },
    SLOWER: { actions: assign({ speed: ({ context }) => shiftSpeed(context.speed, -1) }) },
    SHOW: { actions: assign({ topic: ({ event }) => event.topic }) },
    SET_TOQUE: {
      actions: assign(({ event, context }) => {
        const toque = TOQUES.find((t) => t.id === event.id);
        return toque ? { toque: toque.id, bpm: toque.bpm } : { toque: context.toque };
      }),
    },
    SET_BPM: { actions: assign({ bpm: ({ event }) => Math.round(Math.min(Math.max(event.bpm, MIN_BPM), MAX_BPM)) }) },
    TOGGLE_MUSIC: { actions: assign({ music: ({ context }) => !context.music }) },
    // Stepping is for studying a single moment, so it pauses.
    STEP: { target: ".paused", actions: emit({ type: "step" } as const) },
  },
  states: {
    playing: { on: { TOGGLE_PLAY: "paused" } },
    paused: { on: { TOGGLE_PLAY: "playing" } },
  },
});
