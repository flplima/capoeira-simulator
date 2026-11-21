import { addBandpass, addGrain, finish, rng } from "./dsp";
import type { Stroke } from "./toques";

// The berimbau's wire and gourd, rendered offline into one buffer per stroke (modal synthesis).
//
// A wooden stick strikes the steel wire near its lower end: every partial starts at once, the
// high ones die fast, and the wire's stiffness stretches them a little above the harmonics.
// Where the stick lands sets how loud each partial starts. The gourd is a resonance near the
// third partial that moves up when the player pulls it off the belly after the strike (the
// "uá") and back down as it closes, plus a fixed one around 1.9 kHz. The coin pressed on the
// wire shortens it for the high note. Resting lightly at the same point, it soaks up most of
// the ring within a few periods and buzzes against the wire: the chiado, a dead "tch" at the
// high note's partials.

/** The open note (Hz). Two recorded gungas measured 97 and 106.5 Hz. */
export const DONG_HZ = 104;
/** The coin shortens the wire by about 8 %: the high note is 1.4 semitones up (measured 1.2 and 1.6). */
export const DING_HZ = DONG_HZ * 2 ** (1.4 / 12);

/** Where the coin sits, as a fraction of the open wire from its lower end. */
const COIN = 1 - DONG_HZ / DING_HZ;
/** Where the stick lands, as a fraction of the open wire from its lower end. */
const STICK = 0.16;
/** The same point on the part of the wire above the coin, which is what sounds for the high note and the chiado. */
const STICK_OVER_COIN = (STICK - COIN) / (1 - COIN);
/** Inharmonicity: partial n sits at n·f0·√(1 + B·n²). Measured on a gunga's 3rd to 17th partials. */
const STIFFNESS = 1.5e-4;
/** Above this the caxixi covers the wire. */
const TOP_HZ = 6000;
/** Samples between updates of a partial's amplitude and pitch. */
const BLOCK = 64;
/** A partial quieter than this stops being rendered (the loudest start near 0.7, before the stroke is scaled to its peak). */
const SILENT = 1e-4;

interface Shape {
  f0: number;
  /** Where the stick lands on the vibrating part of the wire, as a fraction of it. */
  strike: number;
  /** Reverberation time (s) of a partial at the bottom, and the frequency (Hz) at which it has halved. */
  t60: number;
  halfAt: number;
  /** The coin resting on the wire: how much of each partial survives its first contacts, and how fast (T60, s) the rest goes. */
  coin?: { keeps: number; t60: number };
  /** How far the player pulls the gourd off the belly after the strike, from 0 (not at all) to 1. */
  opens: number;
  /** Length of the buffer (s), and its loudest sample. */
  seconds: number;
  peak: number;
}

const SHAPES: Record<Stroke, Shape> = {
  dong: { f0: DONG_HZ, strike: STICK, t60: 5, halfAt: 360, opens: 0.3, seconds: 1.6, peak: 0.45 },
  ding: { f0: DING_HZ, strike: STICK_OVER_COIN, t60: 4.5, halfAt: 430, opens: 1, seconds: 1.3, peak: 0.42 },
  chiado: {
    f0: DING_HZ,
    strike: STICK_OVER_COIN,
    t60: 0.8,
    halfAt: 1500,
    coin: { keeps: 0.5, t60: 0.05 },
    opens: 0,
    seconds: 0.35,
    peak: 0.27,
  },
};

/** How strongly a two-pole resonance at hz with quality q passes frequency f. */
const resonance = (f: number, hz: number, q: number) => 1 / Math.sqrt(1 + q * q * (f / hz - hz / f) ** 2);

const smooth = (x: number) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));

/**
 * The gourd's path through one stroke, one value per block. On the belly at the strike it is a
 * narrow peak near the third partial (the nasal "u"). The player pulls it off fast, and the peak
 * grows broad and moves up to 1.6 times higher (the "á"), then closes slowly. Its resonances take
 * about 10 ms to fill after the strike.
 */
function gourdPath(opens: number, blocks: number, sampleRate: number, vary: (spread: number) => number) {
  const open = opens * vary(0.2);
  const opensAt = 0.025 * vary(0.5), opensOver = 0.1 * vary(0.3), closesAt = 0.5 * vary(0.2);
  const hz = new Float64Array(blocks), q = new Float64Array(blocks), gain = new Float64Array(blocks), fill = new Float64Array(blocks);
  for (let b = 0; b < blocks; b++) {
    const t = ((b + 1) * BLOCK) / sampleRate;
    const o = open * (smooth((t - opensAt) / opensOver) - 0.4 * smooth((t - closesAt) / 0.3));
    hz[b] = 300 * 1.6 ** o;
    q[b] = 10 - 7 * o;
    fill[b] = 1 - Math.exp(-t / 0.01);
    gain[b] = fill[b] * (1 - 0.3 * o);
  }
  return { hz, q, gain, fill };
}

/** Renders one stroke of the berimbau, without the caxixi. `seed` picks one of its natural variations. */
export function renderStroke(stroke: Stroke, sampleRate: number, seed: number): Float32Array {
  const shape = SHAPES[stroke];
  const random = rng(seed);
  const vary = (spread: number) => 1 + spread * (2 * random() - 1);
  const out = new Float32Array(Math.round(shape.seconds * sampleRate));
  const blocks = Math.ceil(out.length / BLOCK);
  const gourd = gourdPath(shape.opens, blocks, sampleRate, vary);
  // The coin resting on the wire soaks up the same share of every partial.
  const coin = new Float64Array(blocks).fill(1);
  if (shape.coin) {
    const { keeps, t60 } = shape.coin;
    for (let b = 0; b < blocks; b++) coin[b] = keeps + (1 - keeps) * Math.exp((-6.91 * (b + 1) * BLOCK) / sampleRate / t60);
  }

  // A hard stick: the struck point sets each partial's level, its brief contact rolls off the top.
  const strike = shape.strike * vary(0.08);
  const glide = 0.003 * vary(0.5);
  for (let n = 1; ; n++) {
    const f = shape.f0 * n * Math.sqrt(1 + STIFFNESS * n * n);
    if (f > TOP_HZ || f > 0.45 * sampleRate) break;
    const level = (Math.abs(Math.sin(n * Math.PI * strike)) / (1 + (f / 2000) ** 2)) * vary(0.15);
    // Losses grow with frequency: at `halfAt` Hz a partial rings half as long as at the bottom.
    const fade = Math.exp(((-6.91 * (1 + f / shape.halfAt)) / shape.t60) * (BLOCK / sampleRate));
    // Neither the gourd nor the wire itself carries the lowest partials well.
    const low = (f / 300) ** 4 / (1 + (f / 300) ** 4);
    const side = 0.25 * resonance(f, 1900, 5);
    let c = 1, s = 0, a1 = 0, cw = 1, sw = 0, left = level;
    for (let b = 0; b < blocks; b++) {
      const t = (b * BLOCK) / sampleRate;
      // Past the gourd's opening, a partial that has gone quiet stays quiet.
      if (t > 0.3 && a1 < SILENT) break;
      left *= fade;
      const a0 = a1;
      a1 = left * coin[b] * low * (gourd.gain[b] * resonance(f, gourd.hz[b], gourd.q[b]) + gourd.fill[b] * side + 0.02);
      const step = (a1 - a0) / BLOCK;
      // Struck hard, the wire is stretched and sounds a few cents sharp until it settles.
      if (t < 0.3) {
        const w = (2 * Math.PI * f * (1 + glide * Math.exp(-t / 0.06))) / sampleRate;
        cw = Math.cos(w);
        sw = Math.sin(w);
      }
      const end = Math.min(out.length, (b + 1) * BLOCK);
      let a = a0;
      for (let i = b * BLOCK; i < end; i++, a += step) {
        out[i] += a * s;
        const cn = c * cw - s * sw;
        s = c * sw + s * cw;
        c = cn;
      }
    }
  }

  // The stick's click on the wire.
  const click = new Float32Array(Math.round(0.02 * sampleRate));
  addGrain(click, 0, 0.12 * vary(0.3), 0.0015, sampleRate, random);
  addBandpass(click, out, 3200, 0.9, 1, sampleRate);

  if (stroke === "chiado") {
    // The coin buzzing against the wire: a slap of noise every period, dying with the wire.
    const buzz = new Float32Array(out.length);
    for (let t = 0.002; t < 0.25; t += (1 / shape.f0) * vary(0.04)) {
      addGrain(buzz, Math.round(t * sampleRate), 0.4 * Math.exp(-t / 0.05) * (0.6 + 0.4 * random()), 0.0012, sampleRate, random);
    }
    addBandpass(buzz, out, 2200, 0.9, 1, sampleRate);
    addBandpass(buzz, out, 4000, 1.2, 0.5, sampleRate);
  }
  return finish(out, 0.03, shape.peak * vary(0.05), sampleRate);
}
