export interface Simulation {
  /** Directory with humanoid.xml, policy.json, policy.bin, reference.json. */
  dir: string;
}

export interface Network {
  layers: { in: number; out: number; offset: number }[];
  mean: number[];
  std: number[];
}

export interface PolicyMeta {
  obs: number;
  act: number;
  /** The ginga network and, if trained, the kick, recovery and esquiva networks (ml/brain.py). */
  networks: { ginga: Network; kicks?: Network; recover?: Network; esquiva?: Network };
  /** Seconds of ginga after getting up that the recovery network still drives (older exports: also after a kick). */
  handback: number;
  /** …and after a kick or an esquiva, that move's network (older exports: `handback`). */
  handbackKicks?: number;
  handbackEsquiva?: number;
  clip: number;
  actionScale: number;
  frameSkip: number;
  timestep: number;
  effectors: string[];
  allowedContact: string[];
  /** Never allowed to touch the floor, not even while falling. */
  neverContact: string[];
  /** Outside a recovery, hips or head below these (m) is a fall — or, where the reference goes lower (an
   * esquiva), `below` further down than the reference's (GingaEnv.low). Older exports lack it. */
  low?: { hips: number; head: number; below: number };
  minBpm: number;
  maxBpm: number;
}

export type Move =
  | "martelo"
  | "bencao"
  | "meia-lua-de-frente"
  | "queixada"
  | "ponteira"
  | "joelhada"
  | "meia-lua-de-compasso"
  | "esquiva"
  | "esquiva-baixa"
  | "cocorinha";
export const MOVES: Move[] = [
  "martelo",
  "bencao",
  "meia-lua-de-frente",
  "queixada",
  "ponteira",
  "joelhada",
  "meia-lua-de-compasso",
  "esquiva",
  "esquiva-baixa",
  "cocorinha",
];
export const EXTRA_MOVES: Move[] = [
  "meia-lua-de-frente",
  "queixada",
  "ponteira",
  "joelhada",
  "meia-lua-de-compasso",
  "esquiva-baixa",
  "cocorinha",
];
export const MOVE_NAMES: Record<Move, string> = {
  martelo: "Martelo",
  bencao: "Bênção",
  "meia-lua-de-frente": "Meia-lua de frente",
  queixada: "Queixada",
  ponteira: "Ponteira",
  joelhada: "Joelhada",
  "meia-lua-de-compasso": "Meia-lua de compasso",
  esquiva: "Esquiva lateral",
  "esquiva-baixa": "Esquiva baixa",
  cocorinha: "Cocorinha",
};
/** Which network plays a move (Brain): the kicks share one, the dodges have their own. */
export const MOVE_NETWORK: Record<Move, "kicks" | "esquiva"> = {
  martelo: "kicks",
  bencao: "kicks",
  "meia-lua-de-frente": "kicks",
  queixada: "kicks",
  ponteira: "kicks",
  joelhada: "kicks",
  "meia-lua-de-compasso": "kicks",
  esquiva: "esquiva",
  "esquiva-baixa": "esquiva",
  cocorinha: "esquiva",
};
export type Maneuver = "giro-de-base";
export const MANEUVERS: Maneuver[] = ["giro-de-base"];
export const MANEUVER_NAMES: Record<Maneuver, string> = {
  "giro-de-base": "Giro de base",
};
export type Side = "r" | "l";
/** Where a fall lands (ml/floor.py). */
export type Landing = "quatro" | "negativa" | "joelhos";

export interface KickClip {
  duration: number;
  /** Frames per second (older exports give one rate for all kicks). */
  fps?: number;
  qpos: number[];
  feet: number[];
  mask: number[];
}

/** For each move, tempo and moment of the request: the leg and the ginga phase to start from (ml/choose.py). */
export interface Choices {
  tempos: number[];
  bins: number;
  moves: Partial<Record<Move, { side: Side; entry: number; wait: number }[][]>>;
}

export interface ReferenceData {
  tempos: number[];
  samples: number;
  nq: number;
  qpos: number[][];
  feet: number[][];
  triangle: [number, number][];
  kicks: { fps?: number; blendIn: number; blendOut: number; clips: Record<string, KickClip> };
  /** Falling and getting up (ml/floor.py, env.Floor): the way up from each landing, per side. */
  floor: {
    fps: number;
    fallTime: number;
    troubleTilt: number;
    troubleDrop: number;
    /** On the floor after landing, this far (m) from the way up and the body is lost. */
    lost: number;
    /** After landing, the way up waiting at its slowest this long (s) and the body is stuck on the floor. */
    stuck: number;
    /** A trunk bent forward this far (rad, hips to head) lands on hands and knees whichever way it goes. */
    bent: number;
    /** Going down, no pose has these sites below the floor by more than their clearance (m) (env.GROUND_SITES). */
    ground: { sites: string[]; clear: number[] };
    /** The way up waits for a lagging body (GingaEnv.pace): full speed with the hips waitFrom (m) off the
     * reference's, minPace of it at waitFrom + waitSpan. */
    waitFrom: number;
    waitSpan: number;
    minPace: number;
    /** The arms brace while falling and getting up (env.STRONG_ARMS): stiffness and torque limit per joint. */
    strongArms: { shoulder: [number, number]; elbow: [number, number] };
    /** Per landing and side; phase: where in the ginga cycle the base the way up ends in is. */
    clips: Record<string, { duration: number; qpos: number[]; feet: number[]; phase: number }>;
  } | null;
  choices: Choices | null;
}

export interface LiveFiles {
  xml: string;
  policy: PolicyMeta;
  weights: Float32Array;
  reference: ReferenceData;
}

/** A placement on the floor (env.place_xz): x and z offset, and a yaw turning +Z toward +X. */
export type Placement = [number, number, number];
