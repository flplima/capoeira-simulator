// The berimbau toques the ginga can follow. A toque sets the tempo of the
// game: Angola is slow and close to the ground, São Bento Grande is fast and
// upright. One full ginga (both sides) takes one bar of four beats.
//
// Strokes: "chiado" is the buzz (coin resting lightly on the wire, gourd
// against the belly), "dong" the open low note, "ding" the high note with the
// coin pressed on the wire. Positions are in beats from the start of the bar.

export type Stroke = "chiado" | "dong" | "ding";

export interface Toque {
  id: string;
  name: string;
  bpm: number;
  /** pt-BR description of the game this toque calls for. */
  description: string;
  pattern: { beat: number; stroke: Stroke }[];
}

export const BEATS_PER_BAR = 4;
/** Tempo range the ginga policy was trained on. */
export const MIN_BPM = 60;
export const MAX_BPM = 130;

export const TOQUES: Toque[] = [
  {
    id: "angola",
    name: "Angola",
    bpm: 66,
    description: "Lento e cadenciado: ginga baixa, larga e fluida, perto do chão.",
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
    description: "Angola invertido, mais rápido: jogo de dentro, ginga média.",
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
    description: "Rápido e de pé: ginga alta, curta e ritmada, pronta para o golpe.",
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

/** Seconds per full ginga cycle at a tempo. */
export const cyclePeriod = (bpm: number) => (BEATS_PER_BAR * 60) / bpm;
