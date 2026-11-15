// How often the body the app ships stays up: the same LiveGinga the browser runs (MuJoCo WASM, the
// exported networks), without the rendering. Every move and maneuver the networks know (not only the ones
// the app offers, see src/offered.ts), the walking, the orbit and the change of toque is asked many times
// at each toque, from the ginga, at a moment picked at random; a try stands when the body neither falls
// nor loses its balance from the request until SETTLE seconds of ginga after it.
//
//   npx --yes tsx@4.19.2 scripts/measure-moves.ts                         # 30 tries per action and toque
//   npx --yes tsx@4.19.2 scripts/measure-moves.ts --tries 60 --bpm 72,90,120 --only martelo,joelhada
//   npx --yes tsx@4.19.2 scripts/measure-moves.ts --only ginga --bpm 60,66,72  # the ginga alone, per tempo
//
// The table goes to stderr and the counts, as JSON, to stdout.
import { readFile } from "node:fs/promises";
import load from "@mujoco/mujoco";
import { LiveGinga } from "../src/sim/simulation";
import { MANEUVERS, MOVES, type LiveFiles } from "../src/sim/types";
import { TOQUES } from "../src/music/toques";

const option = (name: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i < 0 ? undefined : process.argv[i + 1];
};
const TRIES = Number(option("tries")) || 30;
const BPMS = option("bpm")?.split(",").map(Number) ?? TOQUES.map((t) => t.bpm);
const ONLY = option("only")?.split(",");
const wanted = (name: string) => !ONLY || ONLY.includes(name);
/** Seconds of ginga after an action before it counts as finished standing. */
const SETTLE = 3;
/** Seconds of ginga alone at each toque. */
const GINGA_SECONDS = 300;
const NUDGES: Record<string, [number, number, number]> = {
  "andar: frente": [1, 0, 0],
  "andar: trás": [-1, 0, 0],
  "andar: esquerda": [0, -1, 0],
  "andar: direita": [0, 1, 0],
  "girar: esquerda": [0, 0, 1],
  "girar: direita": [0, 0, -1],
};

const dir = new URL("../public/data/sim/ginga/", import.meta.url);
const read = (name: string) => readFile(new URL(name, dir));
const files: LiveFiles = {
  xml: (await read("humanoid.xml")).toString(),
  policy: JSON.parse((await read("policy.json")).toString()),
  weights: new Float32Array(new Uint8Array(await read("policy.bin")).buffer),
  reference: JSON.parse((await read("reference.json")).toString()),
};
const mujoco = await load();

/** A small seeded generator (mulberry32), so two runs ask at the same moments. */
function seeded(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Run {
  readonly g = new LiveGinga(mujoco, files);
  constructor(public bpm: number) {}

  get down() {
    return this.g.fallen || this.g.recover !== null;
  }

  /** Advance `seconds`; true when the body fell or lost its balance on the way. */
  advance(seconds: number) {
    let down = false;
    for (let t = 0; t < seconds; t += this.g.controlDt) {
      this.g.update(this.g.controlDt, this.bpm, null);
      down ||= this.down;
    }
    return down;
  }

  /** Ginga until the body has been up and free for `seconds` in a row. */
  steady(seconds: number) {
    for (let up = 0; up < seconds; up += this.g.controlDt) {
      this.g.update(this.g.controlDt, this.bpm, null);
      if (this.down || this.g.busy) up = 0;
    }
  }

  /** One try: ask, wait for the action to end, then SETTLE seconds of ginga. True when it stood. */
  attempt(ask: (g: LiveGinga, run: Run) => unknown, lasts: number) {
    ask(this.g, this);
    let down = this.advance(lasts);
    for (let t = 0; this.g.busy && !down && t < 30; t += this.g.controlDt) down = this.advance(this.g.controlDt);
    return !(down || this.advance(SETTLE));
  }
}

type Counts = Record<string, Record<number, { tries: number; stood: number }>>;
const counts: Counts = {};
const random = seeded(7);

function measure(name: string, ask: (g: LiveGinga, run: Run) => unknown, lasts = 0.03) {
  if (!wanted(name)) return;
  counts[name] = {};
  for (const bpm of BPMS) {
    const run = new Run(bpm);
    let stood = 0;
    for (let i = 0; i < TRIES; i++) {
      run.steady(2 + 3 * random());
      if (run.attempt(ask, lasts)) stood++;
    }
    counts[name][bpm] = { tries: TRIES, stood };
    run.g.dispose();
  }
  report(name);
}

function report(name: string) {
  const cells = Object.entries(counts[name]).map(([bpm, c]) => `${bpm} bpm ${String(c.stood).padStart(3)}/${c.tries}`);
  console.error(`${name.padEnd(32)} ${cells.join("   ")}`);
}

// The ginga alone: the seconds, out of GINGA_SECONDS, in which the body kept its balance.
if (wanted("ginga")) {
  counts.ginga = {};
  for (const bpm of BPMS) {
    const run = new Run(bpm);
    const falls: number[] = [];
    for (let t = 0; t < GINGA_SECONDS; t += 1) {
      if (run.advance(1)) {
        falls.push(t);
        run.steady(2);
      }
    }
    counts.ginga[bpm] = { tries: GINGA_SECONDS, stood: GINGA_SECONDS - falls.length };
    if (falls.length) console.error(`ginga at ${bpm} bpm lost its balance at ${falls.join(", ")} s`);
    run.g.dispose();
  }
  report("ginga");
}

for (const move of MOVES) measure(move, (g) => g.request(move));
for (const maneuver of MANEUVERS) measure(maneuver, (g) => g.requestManeuver(maneuver));
for (const [name, [forward, side, turn]] of Object.entries(NUDGES)) measure(name, (g) => g.nudge(forward, side, turn), 1.2);
measure("orbitar (10 s)", (g) => g.toggleOrbit(), 10);
// The toque changes under the body, as the toque menu does: to each of the other tempos in turn.
measure("trocar de toque", (_, run) => (run.bpm = BPMS[(BPMS.indexOf(run.bpm) + 1) % BPMS.length]), 6);

console.log(JSON.stringify({ tries: TRIES, settle: SETTLE, counts }, null, 1));
