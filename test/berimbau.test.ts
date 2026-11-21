import { describe, expect, it } from "vitest";
import { renderShake } from "../src/music/caxixi";
import { barEvents, TOQUES } from "../src/music/toques";
import { DING_HZ, DONG_HZ, renderStroke } from "../src/music/wire";

const RATE = 48000;
const SEEDS = [1, 2, 3];

/** The pitch an ear hears (Hz): the period of the waveform from its autocorrelation, which holds even when the fundamental is weak. */
function pitch(x: Float32Array) {
  const from = Math.round(0.1 * RATE), span = Math.round(0.4 * RATE);
  const r = (lag: number) => {
    let sum = 0;
    for (let i = from; i < from + span; i++) sum += x[i] * x[i + lag];
    return sum;
  };
  let lag = 0, best = -Infinity;
  for (let l = Math.floor(RATE / 200); l <= Math.ceil(RATE / 70); l++) {
    const v = r(l);
    if (v > best) [lag, best] = [l, v];
  }
  const [y0, y1, y2] = [r(lag - 1), best, r(lag + 1)];
  return RATE / (lag + (0.5 * (y0 - y2)) / (y0 - 2 * y1 + y2));
}

/** How long (s) a sound stays within `db` of its loudest 10 ms. */
function ringsFor(x: Float32Array, db: number) {
  const frame = Math.round(0.01 * RATE);
  const loudness: number[] = [];
  for (let i = 0; i + frame <= x.length; i += frame) {
    let sum = 0;
    for (let j = i; j < i + frame; j++) sum += x[j] ** 2;
    loudness.push(Math.sqrt(sum / frame));
  }
  const floor = Math.max(...loudness) * 10 ** (db / 20);
  return (loudness.findLastIndex((v) => v > floor) + 1) * 0.01;
}

const semitones = (low: number, high: number) => 12 * Math.log2(high / low);
const peak = (x: Float32Array) => x.reduce((max, v) => Math.max(max, Math.abs(v)), 0);

describe("the berimbau's voice", () => {
  it("renders every stroke and shake as audible samples that never clip", () => {
    const sounds = SEEDS.flatMap((seed) => [
      ...(["dong", "ding", "chiado"] as const).map((s) => renderStroke(s, RATE, seed)),
      ...(["strike", "swing"] as const).map((s) => renderShake(s, RATE, seed)),
    ]);
    for (const sound of sounds) {
      expect(sound.every(Number.isFinite)).toBe(true);
      expect(peak(sound)).toBeLessThanOrEqual(1);
      expect(peak(sound)).toBeGreaterThan(0.1);
    }
  });

  it("sounds the open note at its pitch, though its fundamental is weak", () => {
    for (const seed of SEEDS) expect(Math.abs(pitch(renderStroke("dong", RATE, seed)) / DONG_HZ - 1)).toBeLessThan(0.02);
  });

  it("sounds the high note 1.4 semitones above the open one", () => {
    for (const seed of SEEDS) {
      const interval = semitones(pitch(renderStroke("dong", RATE, seed)), pitch(renderStroke("ding", RATE, seed)));
      expect(interval).toBeCloseTo(semitones(DONG_HZ, DING_HZ), 0);
      expect(interval).toBeGreaterThan(1.1);
      expect(interval).toBeLessThan(1.7);
    }
  });

  it("lets the open note ring, and the chiado die within a fraction of it", () => {
    for (const seed of SEEDS) {
      const dong = ringsFor(renderStroke("dong", RATE, seed), -20);
      expect(dong).toBeGreaterThan(0.5);
      expect(ringsFor(renderStroke("chiado", RATE, seed), -20)).toBeLessThan(dong / 3);
    }
  });

  it("gives the same take for the same seed and a different one for another", () => {
    expect(renderStroke("ding", RATE, 1)).toEqual(renderStroke("ding", RATE, 1));
    expect(renderStroke("ding", RATE, 1)).not.toEqual(renderStroke("ding", RATE, 2));
    expect(renderShake("strike", RATE, 1)).not.toEqual(renderShake("strike", RATE, 2));
  });
});

describe("a bar of each toque", () => {
  it("keeps every note of the toque on its beat", () => {
    for (const toque of TOQUES) {
      const strokes = barEvents(toque).filter((e) => e.stroke);
      expect(strokes.map(({ beat, stroke }) => ({ beat, stroke }))).toEqual(toque.pattern);
    }
  });

  it("shakes the caxixi on every eighth note, hardest with a stroke or on a beat", () => {
    for (const toque of TOQUES) {
      const events = barEvents(toque);
      expect(events.map((e) => e.beat)).toEqual([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]);
      for (const e of events) expect(e.shake).toBe(e.stroke || Number.isInteger(e.beat) ? "strike" : "swing");
    }
  });
});
