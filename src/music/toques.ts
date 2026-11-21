// The berimbau toques the ginga can follow. A toque sets the tempo of the
// game: Angola is slow and close to the ground, São Bento Grande is fast and
// upright. One full ginga (both sides) takes one bar of four beats.
//
// Strokes: "chiado" is the buzz (coin resting lightly on the wire, gourd
// against the belly), "dong" the open low note, "ding" the high note with the
// coin pressed on the wire. Positions are in beats from the start of the bar.

export type Stroke = "chiado" | "dong" | "ding";
/** A caxixi shake: with a stroke (the stick stops dead on the wire) or on the hand's back-swing. */
export type Shake = "strike" | "swing";

export interface Toque {
  id: "angola" | "sao-bento-pequeno" | "sao-bento-grande";
  name: string;
  bpm: number;
  pattern: { beat: number; stroke: Stroke }[];
}

export const BEATS_PER_BAR = 4;
/** Tempo range a visitor can set. The ginga falls below 70 bpm (13 falls in 300 s at 66, 31 at 60) and from
 * 70 to 112 it does not, so the lowest toque and the range start there; 120 is the fastest toque. */
export const MIN_BPM = 70;
export const MAX_BPM = 120;

export const TOQUES: Toque[] = [
  {
    id: "angola",
    name: "Angola",
    bpm: 72,
    // tch-tch DONG ding (pausa)
    pattern: [
      { beat: 0, stroke: "chiado" },
      { beat: 0.5, stroke: "chiado" },
      { beat: 1, stroke: "dong" },
      { beat: 2, stroke: "ding" },
    ],
  },
  {
    id: "sao-bento-pequeno",
    name: "São Bento Pequeno",
    bpm: 90,
    // tch-tch DING dong (pausa)
    pattern: [
      { beat: 0, stroke: "chiado" },
      { beat: 0.5, stroke: "chiado" },
      { beat: 1, stroke: "ding" },
      { beat: 2, stroke: "dong" },
    ],
  },
  {
    id: "sao-bento-grande",
    name: "São Bento Grande",
    bpm: 120,
    // tch-tch DING dong dong — the pause becomes a second low note
    pattern: [
      { beat: 0, stroke: "chiado" },
      { beat: 0.5, stroke: "chiado" },
      { beat: 1, stroke: "ding" },
      { beat: 2, stroke: "dong" },
      { beat: 3, stroke: "dong" },
    ],
  },
];

/** The toque the app opens on. */
export const DEFAULT_TOQUE = TOQUES[1];

/** One moment of the bar: a stroke on the wire (or none) and the caxixi shake heard with it, each with its loudness. */
export interface BarEvent {
  beat: number;
  stroke: Stroke | null;
  strokeGain: number;
  shake: Shake;
  shakeGain: number;
}

/**
 * Everything the berimbau plays in one bar of a toque. The caxixi is in the stick hand, so it
 * sounds with every stroke, shakes on its own on a beat without one, and lightly on the hand's
 * back-swing at each eighth note in between. A stroke off the beat is played softer.
 */
export function barEvents(toque: Toque): BarEvent[] {
  const eighths = Array.from({ length: BEATS_PER_BAR * 2 }, (_, i) => i / 2);
  const beats = [...new Set([...eighths, ...toque.pattern.map((p) => p.beat)])].sort((a, b) => a - b);
  return beats.map((beat) => {
    const stroke = toque.pattern.find((p) => p.beat === beat)?.stroke ?? null;
    const onBeat = Number.isInteger(beat);
    if (stroke) return { beat, stroke, strokeGain: onBeat ? 1 : 0.6, shake: "strike", shakeGain: onBeat ? 1 : 0.7 };
    return { beat, stroke, strokeGain: 0, shake: onBeat ? "strike" : "swing", shakeGain: onBeat ? 1 : 0.6 };
  });
}

/** Seconds per full ginga cycle at a tempo. */
export const cyclePeriod = (bpm: number) => (BEATS_PER_BAR * 60) / bpm;
