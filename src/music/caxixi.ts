import { addBandpass, addGrain, finish, rng } from "./dsp";
import type { Shake } from "./toques";

// The caxixi, a woven basket of seeds in the stick hand, rendered as a stochastic shaker (after
// Perry Cook's PhISEM): the hand's swing gives the seeds energy that dies away, every seed that
// hits the basket's hard bottom is a short click, and the basket colors the clicks with a few
// broad resonances between 1.5 and 8 kHz.

/** The swing of the hand for each shake: how fast it starts (s), how long it lasts (s), how many seeds hit per second at its peak. */
const SWINGS: Record<Shake, { attack: number; decay: number; hits: number; peak: number }> = {
  // The stick hits the wire and stops dead: the seeds all land at once.
  strike: { attack: 0.004, decay: 0.05, hits: 3500, peak: 0.95 },
  // The hand swings back between strokes: fewer seeds, a softer start.
  swing: { attack: 0.015, decay: 0.04, hits: 1600, peak: 0.55 },
};

/** The basket's resonances: center (Hz), Q and gain. */
const BASKET = [
  [1800, 1.4, 1],
  [2700, 1.6, 1],
  [4100, 1.6, 0.8],
  [6500, 1.8, 0.45],
] as const;

/** Renders one shake of the caxixi. `seed` picks one of its natural variations. */
export function renderShake(shake: Shake, sampleRate: number, seed: number): Float32Array {
  const swing = SWINGS[shake];
  const random = rng(seed);
  const seconds = swing.attack + swing.decay * 6;
  const clicks = new Float32Array(Math.round(seconds * sampleRate));
  // The hand's energy, (1 - e^(-t/attack)) e^(-t/decay), as two exponentials stepped per sample.
  const slow = Math.exp(-1 / (swing.decay * sampleRate));
  const fast = slow * Math.exp(-1 / (swing.attack * sampleRate));
  for (let i = 0, falls = 1, rises = 1; i < clicks.length; i++, falls *= slow, rises *= fast) {
    const energy = falls - rises;
    if (random() < (swing.hits * energy) / sampleRate) {
      addGrain(clicks, i, Math.sqrt(energy) * (0.3 + 0.7 * random()), 0.0006 + 0.0012 * random(), sampleRate, random);
    }
  }
  const out = new Float32Array(clicks.length);
  for (const [hz, q, gain] of BASKET) addBandpass(clicks, out, hz * (0.93 + 0.14 * random()), q, gain, sampleRate);
  return finish(out, 0.02, swing.peak * (0.92 + 0.16 * random()), sampleRate);
}
