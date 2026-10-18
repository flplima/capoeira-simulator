import { Quaternion, Vector3 } from "three";
import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";
import type { BoneName, Rig } from "../rig/rig";
import { cyclePeriod } from "../music/toques";

// The ginga, live: the trained policy from ml/ drives a MuJoCo simulation of
// the body right here in the browser. MuJoCo itself runs as WebAssembly (the
// same engine version the policy was trained with), and every control step
// (0.03 s) the policy network looks at the body — joint angles and speeds,
// where its hands and feet are, which feet touch the floor, where the ginga
// triangle wants them, the berimbau's tempo — and decides the next joint
// targets. Nothing is pre-recorded: the body balances, follows the music, and
// keeps going for as long as the page is open. When it loses its balance a
// third network takes over: it goes down onto its hands and knees, into the
// queda de quatro or into the negativa, gets back up and hands the ginga
// back, which carries on where the body stood up.
//
// The observation and the network are ports of ml/env.py and ml/train.py;
// test/live.test.ts checks them against values computed in Python.

export interface Simulation {
  /** Directory with humanoid.xml, policy.json, policy.bin, reference.json. */
  dir: string;
}

interface Network {
  layers: { in: number; out: number; offset: number }[];
  mean: number[];
  std: number[];
}

interface PolicyMeta {
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
const EXTRA_MOVES: Move[] = [
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
const MOVE_NETWORK: Record<Move, "kicks" | "esquiva"> = {
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
export type GingaStyle = "auto" | "angola" | "padrao" | "regional";
export type Maneuver = "troca-de-base" | "giro-de-base" | "role" | "passagem-de-negativa";
export const MANEUVERS: Maneuver[] = ["troca-de-base", "giro-de-base", "role", "passagem-de-negativa"];
export const MANEUVER_NAMES: Record<Maneuver, string> = {
  "troca-de-base": "Troca de base",
  "giro-de-base": "Giro de base",
  role: "Rolê",
  "passagem-de-negativa": "Passagem de negativa",
};
type Side = "r" | "l";
/** Where a fall lands (ml/floor.py). */
type Landing = "quatro" | "negativa" | "joelhos";
const LANDING_NAMES: Record<Landing, string> = { quatro: "queda de quatro", negativa: "negativa", joelhos: "mãos e joelhos" };
const FROM_LANDING: Record<Landing, string> = { quatro: "da queda de quatro", negativa: "da negativa", joelhos: "das mãos e joelhos" };

interface KickClip {
  duration: number;
  /** Frames per second (older exports give one rate for all kicks). */
  fps?: number;
  qpos: number[];
  feet: number[];
  mask: number[];
}

/** For each move, tempo and moment of the request: the leg and the ginga phase to start from (ml/choose.py). */
interface Choices {
  tempos: number[];
  bins: number;
  moves: Partial<Record<Move, { side: Side; entry: number; wait: number }[][]>>;
}

interface ReferenceData {
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

export async function fetchLiveFiles(dir: string): Promise<LiveFiles> {
  const get = (name: string) =>
    fetch(`${dir}${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name}: ${r.status}`);
      return r;
    });
  const [xml, policy, weights, reference] = await Promise.all([
    get("humanoid.xml").then((r) => r.text()),
    get("policy.json").then((r) => r.json()),
    get("policy.bin").then((r) => r.arrayBuffer()),
    get("reference.json").then((r) => r.json()),
  ]);
  return { xml, policy, weights: new Float32Array(weights), reference };
}

let engine: Promise<MainModule> | null = null;

/** The WebAssembly engine (10 MB, compiled once per page); call early to have it ready. */
export function loadEngine(): Promise<MainModule> {
  engine ??= Promise.all([import("@mujoco/mujoco"), import("@mujoco/mujoco/mujoco.wasm?url")]).then(([{ default: load }, wasm]) =>
    load({ locateFile: (path: string) => (path.endsWith(".wasm") ? wasm.default : path) }),
  );
  return engine;
}

/** Load the engine and the trained ginga (browser only; tests build a LiveGinga directly). */
export async function loadLive(sim: Simulation): Promise<LiveGinga> {
  const [mujoco, files] = await Promise.all([loadEngine(), fetchLiveFiles(sim.dir)]);
  return new LiveGinga(mujoco, files);
}

// ---- small math helpers (row-major 3×3, quaternions w x y z) -------------

function yawOf(w: number, x: number, y: number, z: number) {
  return Math.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y));
}

function quatMatrix(w: number, x: number, y: number, z: number, out: number[]) {
  out[0] = 1 - 2 * (y * y + z * z);
  out[1] = 2 * (x * y - w * z);
  out[2] = 2 * (x * z + w * y);
  out[3] = 2 * (x * y + w * z);
  out[4] = 1 - 2 * (x * x + z * z);
  out[5] = 2 * (y * z - w * x);
  out[6] = 2 * (x * z - w * y);
  out[7] = 2 * (y * z + w * x);
  out[8] = 1 - 2 * (x * x + y * y);
}

function nlerp(a: ArrayLike<number>, ai: number, b: ArrayLike<number>, bi: number, t: number, out: number[], oi: number) {
  const sign = a[ai] * b[bi] + a[ai + 1] * b[bi + 1] + a[ai + 2] * b[bi + 2] + a[ai + 3] * b[bi + 3] < 0 ? -1 : 1;
  let n = 0;
  for (let k = 0; k < 4; k++) {
    out[oi + k] = a[ai + k] + (sign * b[bi + k] - a[ai + k]) * t;
    n += out[oi + k] ** 2;
  }
  n = Math.sqrt(n);
  for (let k = 0; k < 4; k++) out[oi + k] /= n;
}

const wrap = (x: number) => ((x % 1) + 1) % 1;
/** Which way a move goes, for the status line: a kick names the kicking leg (the one back in the base); the
 * esquiva from the base with the right leg back drops over the front (left) leg, to the left. */
const which = (move: Move, side: Side) =>
  move === "esquiva" ? `para a ${side === "r" ? "esquerda" : "direita"}` : `perna ${side === "r" ? "direita" : "esquerda"}`;
const GRAVITY = 9.81; // m/s² (env.GRAVITY)
const smoothstep = (x: number) => {
  const t = Math.min(Math.max(x, 0), 1);
  return t * t * (3 - 2 * t);
};
// Where each leg can start a kick, as in ml/env.py: the base with that leg back.
const ENTRY_CENTRE: Record<Side, number> = { r: 0, l: 0.5 };
/** Furthest (m) a kick follows its support foot away from the corner (env.ANCHOR_LIMIT). */
const ANCHOR_LIMIT = 0.25;
const tempoInput = (bpm: number) => (bpm - 95) / 35;

/** A placement on the floor (env.place_xz): x and z offset, and a yaw turning +Z toward +X. */
export type Placement = [number, number, number];

function placeXZ(p: Placement, x: number, z: number): [number, number] {
  const c = Math.cos(p[2]);
  const s = Math.sin(p[2]);
  return [c * x + s * z + p[0], -s * x + c * z + p[1]];
}

function unplaceXZ(p: Placement, x: number, z: number): [number, number] {
  const c = Math.cos(p[2]);
  const s = Math.sin(p[2]);
  const dx = x - p[0];
  const dz = z - p[1];
  return [c * dx - s * dz, s * dx + c * dz];
}

/** A reference pose and its feet points → world, in place (env.place_pose; velocities aren't needed here). */
function placePose(p: Placement, q: number[], feet: number[]) {
  [q[0], q[2]] = placeXZ(p, q[0], q[2]);
  const hw = Math.cos(p[2] / 2);
  const hy = Math.sin(p[2] / 2);
  const [w, x, y, z] = [q[3], q[4], q[5], q[6]];
  q[3] = hw * w - hy * y;
  q[4] = hw * x + hy * z;
  q[5] = hw * y + hy * w;
  q[6] = hw * z - hy * x;
  for (let f = 0; f < 2; f++) [feet[2 * f], feet[2 * f + 1]] = placeXZ(p, feet[2 * f], feet[2 * f + 1]);
}
/** How long the body may lie on the floor before it is stood back up. */
const GET_UP_AFTER = 1.2;
/** How strongly the ginga's phase is pulled toward the music's bar each control step. */
const SYNC_GAIN = 0.08;
/** The pull never runs the cycle more than this much faster or slower: a jump in the reference topples the body. */
const SYNC_LIMIT = 0.1;

const axis = new Vector3();
const hinge = new Quaternion();

export class LiveGinga {
  readonly model: MjModel;
  readonly data: MjData;
  readonly triangle: [number, number][];
  playing = true;
  speed = 1;
  bpm = 66;
  /** 0..1 through the ginga cycle; 0 is the right foot on the back corner. */
  phase = 0;
  cycles = 0;
  fallen = false;
  /** The kick being thrown, its leg and how far into it (s); null while gingando. */
  move: Move | null = null;
  side: Side = "r";
  kickT = 0;
  /** xz shift of the kick reference onto the foot the body stands on (Choreography.anchor). */
  anchor: [number, number] = [0, 0];
  /** A requested kick waiting for its entry phase. */
  pending: { move: Move; side: Side; entry: number } | null = null;
  /** Where the ginga's triangle is (Choreography.frame): it moves to where the body gets up. */
  placement: Placement = [0, 0, 0];
  /** Active standing or floor displacement maneuver (troca-de-base, giro-de-base, role, passagem-de-negativa). */
  maneuver: { type: Maneuver; t: number; duration: number; side: Side; startYaw: number } | null = null;
  /** Continuous navigation controls (-1..1) and circular roda orbit toggle. */
  nav = { forward: 0, side: 0, turn: 0, orbit: false };
  private nudgeInput = { forward: 0, side: 0, turn: 0, timer: 0 };
  /** Quebra na paralela (finta de ginga) waiting for the next parallel pass (phase ≈ 0.25 or 0.75). */
  pendingFeintGinga = false;
  /** Modo ameaça (finta de golpe): kicks blend to 45% and retract smoothly to the ginga base. */
  feintMode = false;
  /** Ginga posture style: auto follows tempo (Angola <= 76 bpm, Regional >= 104 bpm), or explicit override. */
  gingaStyle: GingaStyle = "auto";
  /** Falling and getting up (Choreography.recover): landing, side, time since balance was lost, the pose
   * it was lost in, and where the landing is placed. */
  /** sink: the hips' vertical speed when balance was lost (m/s, up positive). */
  recover: { move: Landing; side: Side; t: number; onset: number[]; place: Placement; sink: number } | null = null;
  /** A shove in progress: horizontal force on the chest (N) and control steps left. */
  private pushForce: [number, number] = [0, 0];
  private pushLeft = 0;
  private listeners = new Set<() => void>();
  private fallenFor = 0;
  /** qpos addresses of the shoulders and elbows (Choreography.arms). */
  private readonly armQpos = new Set<number>();
  /** How long the way up has been waiting at its slowest for a body lying on the floor (env.STUCK). */
  private stuckFor = 0;
  private accumulated = 0;
  private readonly controlDt: number;
  private readonly joints: { name: string; body: BoneName; axis: number[]; adr: number }[];
  private readonly bodies: { bone: BoneName; joints: number[] }[];
  private readonly effectorIds: number[];
  private readonly headId: number;
  private readonly floor: number;
  private readonly feetGeoms: number[];
  private readonly allowed: Set<number>;
  private readonly never: Set<number>;
  private readonly chestId: number;
  private readonly floorPose: number[];
  private readonly floorFeet: number[] = [0, 0, 0, 0];
  private readonly soles: [number, number][];
  private readonly groundData: MjData;
  private readonly groundSites: number[];
  private readonly lo: Float64Array;
  private readonly hi: Float64Array;
  private readonly obs: Float64Array;
  private readonly input: Float64Array;
  private readonly layers: Float64Array[];
  private readonly refNext: number[];
  private readonly refFeet: number[] = [0, 0, 0, 0];
  private readonly ginga: number[];
  private readonly kickPose: number[];
  private readonly kickFeet: number[] = [0, 0, 0, 0];
  private readonly frame: number[] = new Array(9).fill(0);
  private readonly rot: number[] = new Array(9).fill(0);

  constructor(
    private readonly mujoco: MainModule,
    private readonly files: LiveFiles,
  ) {
    const m = (this.model = mujoco.MjModel.from_xml_string(files.xml));
    this.data = new mujoco.MjData(m);
    this.groundData = new mujoco.MjData(m);
    const { policy, reference } = files;
    this.triangle = reference.triangle;
    this.controlDt = policy.timestep * policy.frameSkip;
    this.refNext = new Array(m.nq).fill(0);
    this.ginga = new Array(m.nq).fill(0);
    this.kickPose = new Array(m.nq).fill(0);
    this.floorPose = new Array(m.nq).fill(0);
    this.obs = new Float64Array(policy.obs);
    this.input = new Float64Array(policy.obs);
    this.layers = policy.networks.ginga.layers.map((l) => new Float64Array(l.out));

    const bodyIds = m.jnt_bodyid as Int32Array;
    const axes = m.jnt_axis as Float64Array;
    const adr = m.jnt_qposadr as Int32Array;
    const trn = m.actuator_trnid as Int32Array;
    const jointNames = new Map<number, string>();
    for (let i = 0; i < m.nu; i++) {
      const actName = m.actuator(i).name;
      jointNames.set(trn[2 * i], actName);
      if (/^(shoulder|elbow)/.test(actName)) this.armQpos.add(adr[trn[2 * i]]);
    }
    this.joints = [];
    for (let j = 1; j < m.njnt; j++) {
      this.joints.push({
        name: jointNames.get(j) ?? "",
        body: m.body(bodyIds[j]).name as BoneName,
        axis: [axes[3 * j], axes[3 * j + 1], axes[3 * j + 2]],
        adr: adr[j],
      });
    }
    const byBody = new Map<BoneName, number[]>();
    this.joints.forEach((j, i) => byBody.set(j.body, [...(byBody.get(j.body) ?? []), i]));
    this.bodies = [...byBody].map(([bone, joints]) => ({ bone, joints }));

    this.effectorIds = policy.effectors.map((n) => m.body(n).id);
    this.headId = m.body("head").id;
    this.floor = m.geom("floor").id;
    this.feetGeoms = [m.geom("foot_l").id, m.geom("foot_r").id];
    this.allowed = new Set(policy.allowedContact.map((n) => m.geom(n).id));
    this.never = new Set(policy.neverContact.map((n) => m.geom(n).id));
    this.chestId = m.body("chest").id;
    this.soles = (["l", "r"] as const).map((s) => [m.site(`heel_${s}`).id, m.site(`toe_${s}`).id]);
    this.groundSites = (reference.floor?.ground.sites ?? []).map((n) => m.site(n).id);
    const range = m.actuator_ctrlrange as Float64Array;
    this.lo = new Float64Array(m.nu).map((_, i) => range[2 * i]);
    this.hi = new Float64Array(m.nu).map((_, i) => range[2 * i + 1]);
    this.reset(0);
  }

  dispose() {
    this.groundData.delete();
    this.data.delete();
    this.model.delete();
  }

  get period() {
    return cyclePeriod(this.bpm);
  }

  /** Stand the body in the reference pose at a phase, at rest. */
  reset(phase: number) {
    const { mujoco, model: m, data: d } = this;
    mujoco.mj_resetData(m, d);
    this.phase = wrap(phase);
    const busy = this.busy;
    this.move = null;
    this.pending = null;
    this.recover = null;
    this.maneuver = null;
    this.pendingFeintGinga = false;
    this.nudgeInput.timer = 0;
    this.placement = [0, 0, 0];
    this.pushLeft = 0;
    // Standing again is the ginga's (Brain.reset): no kick or recovery network is still handing back.
    this.sinceKick = Infinity;
    this.sinceRecover = Infinity;
    this.wasRecovering = false;
    if (busy) this.notify();
    (d.qpos as Float64Array).set(this.sampleReference(this.phase, this.bpm, this.refNext));
    // Moving with the reference, as GingaEnv.reset: standing still mid-cycle, the ginga topples.
    const h = 1 / this.files.reference.samples;
    const before = this.sampleReference(this.phase - h, this.bpm, new Array(m.nq));
    const after = this.sampleReference(this.phase + h, this.bpm, new Array(m.nq));
    mujoco.mj_differentiatePos(m, d.qvel, 2 * h * this.period, before, after);
    // Lift the pose out of the floor (a rolled foot's corner can sit a centimetre deep), as env.lift_out_of_floor.
    mujoco.mj_forward(m, d);
    let deepest = 0;
    const contacts = d.contact;
    for (let i = 0; i < d.ncon; i++) {
      const c = contacts.get(i)!;
      if (c.geom1 === this.floor || c.geom2 === this.floor) deepest = Math.min(deepest, c.dist);
      c.delete();
    }
    contacts.delete();
    (d.qpos as Float64Array)[1] += -deepest + 0.002;
    mujoco.mj_forward(m, d);
    this.fallen = false;
    this.fallenFor = 0;
    this.stuckFor = 0;
  }

  /** Put the simulation in an exact state (tests). */
  setState(
    qpos: number[],
    qvel: number[],
    phase: number,
    bpm: number,
    move: Move | null = null,
    side: Side = "r",
    kickT = 0,
    sinceKick = Infinity,
    anchor: [number, number] = [0, 0],
    placement: Placement = [0, 0, 0],
    recover: LiveGinga["recover"] = null,
    sinceRecover = Infinity,
  ) {
    this.sinceKick = sinceKick;
    this.sinceRecover = sinceRecover;
    this.mover = move && MOVE_NETWORK[move];
    this.anchor = [...anchor];
    this.placement = [...placement];
    this.recover = recover && { ...recover, onset: [...recover.onset], place: [...recover.place] };
    this.maneuver = null;
    this.pendingFeintGinga = false;
    this.nudgeInput.timer = 0;
    (this.data.qpos as Float64Array).set(qpos);
    (this.data.qvel as Float64Array).set(qvel);
    this.phase = phase;
    this.bpm = bpm;
    this.move = move;
    this.side = side;
    this.kickT = kickT;
    this.mujoco.mj_forward(this.model, this.data);
  }

  // ---- reference (port of env.Reference) --------------------------------

  private neighbours(bpm: number): [number, number, number] {
    const t = this.files.reference.tempos;
    const b = Math.min(Math.max(bpm, t[0]), t[t.length - 1]);
    let j = t.findIndex((x) => x >= b);
    if (j < 0) j = t.length - 1;
    const i = Math.max(j - 1, 0);
    return [i, j, i === j ? 0 : (b - t[i]) / (t[j] - t[i])];
  }

  private sampleReference(phase: number, bpm: number, out: number[]) {
    const { samples, nq, qpos } = this.files.reference;
    const [i, j, w] = this.neighbours(bpm);
    const x = wrap(phase) * samples;
    const k = Math.floor(x) % samples;
    const k1 = (k + 1) % samples;
    const a = x - Math.floor(x);
    const pair: number[][] = [];
    for (const t of [i, j]) {
      const table = qpos[t];
      const q = new Array(nq);
      for (let c = 0; c < nq; c++) q[c] = table[k * nq + c] + (table[k1 * nq + c] - table[k * nq + c]) * a;
      nlerp(table, k * nq + 3, table, k1 * nq + 3, a, q, 3);
      pair.push(q);
    }
    for (let c = 0; c < nq; c++) out[c] = pair[0][c] + (pair[1][c] - pair[0][c]) * w;
    nlerp(pair[0], 3, pair[1], 3, w, out, 3);
    return out;
  }

  private feetAt(phase: number, bpm: number, out: number[]) {
    const { samples, feet } = this.files.reference;
    const [i, j, w] = this.neighbours(bpm);
    const x = wrap(phase) * samples;
    const k = Math.floor(x) % samples;
    const k1 = (k + 1) % samples;
    const a = x - Math.floor(x);
    for (let c = 0; c < 4; c++) {
      const fi = feet[i][k * 4 + c] + (feet[i][k1 * 4 + c] - feet[i][k * 4 + c]) * a;
      const fj = feet[j][k * 4 + c] + (feet[j][k1 * 4 + c] - feet[j][k * 4 + c]) * a;
      out[c] = fi + (fj - fi) * w;
    }
    return out;
  }

  // ---- moves (port of env.Kicks and env.Choreography) -------------------

  /** Gingando, waiting for the moment to kick, kicking, or down (falling and getting up). */
  get mode(): "ginga" | "waiting" | "kick" | "down" {
    return this.recover ? "down" : this.move || (this.maneuver && !this.pending) ? "kick" : this.pending ? "waiting" : "ginga";
  }

  get busy() {
    return this.move !== null || this.pending !== null || this.recover !== null || this.maneuver !== null;
  }

  /** Whether a move can be asked for: the kicks always (when present in the reference), the dodges only with
   * their own network and reference. */
  available(move: Move) {
    return (MOVE_NETWORK[move] === "kicks" || !!this.files.policy.networks.esquiva) && !!this.clip(move, "r");
  }

  /** Whether a standing or floor displacement maneuver can be asked for. */
  availableManeuver(maneuver: Maneuver) {
    if (maneuver === "role") return this.available("cocorinha");
    if (maneuver === "passagem-de-negativa") return this.available("esquiva-baixa");
    return true;
  }

  /** Trained to fall and get up (with a recovery network and its references). */
  get canRecover() {
    return !!(this.files.policy.networks.recover && this.files.reference.floor);
  }

  /** Shove the chest for 0.2 s with a horizontal force (N, world x/z), like a partner would (GingaEnv.push). */
  push(fx: number, fz: number) {
    this.pushForce = [fx, fz];
    this.pushLeft = Math.round(0.2 / this.controlDt);
  }

  /** A shove of `force` newtons from a random direction. */
  shove(force = 350) {
    const angle = (Math.random() * 2 - 1) * Math.PI;
    this.push(force * Math.cos(angle), force * Math.sin(angle));
  }

  /** Called whenever the mode changes (React subscribes to enable and disable the move buttons). */
  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }

  private notify() {
    for (const listener of this.listeners) listener();
  }

  /** Set continuous navigation inputs (-1..1) or orbit toggle. */
  setNav(partial: Partial<typeof this.nav>) {
    Object.assign(this.nav, partial);
    this.notify();
  }

  /** Apply a timed navigation impulse (for UI buttons). */
  nudge(forward: number, side: number, turn: number, duration = 0.9) {
    this.nudgeInput = { forward, side, turn, timer: duration };
    this.notify();
  }

  /** Toggle circular roda navigation facing the center. */
  toggleOrbit() {
    this.nav.orbit = !this.nav.orbit;
    this.notify();
  }

  /** Schedule a quebra na paralela (finta de ginga) at the next parallel stance pass. */
  feintGinga() {
    if (this.busy) return false;
    this.pendingFeintGinga = !this.pendingFeintGinga;
    this.notify();
    return this.pendingFeintGinga;
  }

  /** Toggle modo ameaça (kicks chamber to 45% blend and retract to the ginga). */
  toggleFeintMode() {
    this.feintMode = !this.feintMode;
    this.notify();
    return this.feintMode;
  }

  /** Set the ginga posture style (auto, angola, padrao, regional). */
  setGingaStyle(style: GingaStyle) {
    this.gingaStyle = style;
    this.notify();
  }

  /** Request a standing or floor displacement maneuver (troca-de-base, giro-de-base, role, passagem-de-negativa). */
  requestManeuver(type: Maneuver) {
    if (this.busy || !this.availableManeuver(type)) return null;
    const side: Side = wrap(this.phase) < 0.5 ? "r" : "l";
    if (type === "troca-de-base") {
      this.maneuver = { type, t: 0, duration: 1.35, side, startYaw: this.placement[2] };
      this.notify();
      return this.maneuver;
    }
    if (type === "giro-de-base") {
      this.maneuver = { type, t: 0, duration: 1.6, side, startYaw: this.placement[2] };
      this.notify();
      return this.maneuver;
    }
    const baseMove: Move = type === "role" ? "cocorinha" : "esquiva-baixa";
    const pending = this.request(baseMove);
    if (!pending) return null;
    this.maneuver = {
      type,
      t: 0,
      duration: this.clip(baseMove, pending.side).duration,
      side: pending.side,
      startYaw: this.placement[2],
    };
    this.notify();
    return this.maneuver;
  }

  /**
   * Ask for a kick. The leg and the moment come from ml/choose.py, which tried
   * every leg and entry in simulation for this tempo and point of the cycle;
   * without that table, the leg whose base comes first is used. Ignored while
   * another move is under way.
   */
  request(move: Move) {
    if (this.busy || !this.available(move)) return null;
    const choices = this.files.reference.choices;
    const table = choices?.moves[move];
    let choice: { side: Side; entry: number };
    if (choices && table) {
      const t = choices.tempos.reduce((best, x, i) => (Math.abs(x - this.bpm) < Math.abs(choices.tempos[best] - this.bpm) ? i : best), 0);
      choice = table[t][Math.floor(this.phase * choices.bins) % choices.bins];
    } else {
      const side = (["r", "l"] as Side[]).reduce((a, b) => (wrap(ENTRY_CENTRE[a] - this.phase) <= wrap(ENTRY_CENTRE[b] - this.phase) ? a : b));
      choice = { side, entry: ENTRY_CENTRE[side] };
    }
    this.pending = { move, side: choice.side, entry: wrap(choice.entry) };
    this.notify();
    return this.pending;
  }

  private clip(move: Move, side: Side) {
    return this.files.reference.kicks.clips[`${move}_${side}`];
  }

  private blend(t: number) {
    const { blendIn, blendOut } = this.files.reference.kicks;
    const scale = this.feintMode && this.move && MOVE_NETWORK[this.move] === "kicks" ? 0.45 : 1;
    return scale * smoothstep(t / blendIn) * smoothstep((this.clip(this.move!, this.side).duration - t) / blendOut);
  }

  /** Reference pose and feet targets `ahead` seconds from now, in the world (Choreography.target). */
  private target(ahead: number): { q: number[]; feet: number[] } {
    if (this.recover) return this.recoverTarget(ahead);
    const out = this.ownTarget(ahead);
    placePose(this.placement, out.q, out.feet);
    return out;
  }

  private floorClip(move: Landing, side: Side) {
    return this.files.reference.floor!.clips[`${move}_${side}`];
  }

  /** The way up from a landing at time t, in its own frame (env.Floor.sample). */
  private sampleFloor(move: Landing, side: Side, t: number) {
    const { fps } = this.files.reference.floor!;
    const clip = this.floorClip(move, side);
    const nq = this.model.nq;
    const frames = clip.qpos.length / nq;
    const x = Math.min(Math.max(t, 0), clip.duration) * fps;
    const k = Math.min(Math.floor(x), frames - 2);
    const a = x - k;
    const q = this.floorPose;
    for (let c = 0; c < nq; c++) q[c] = clip.qpos[k * nq + c] + (clip.qpos[(k + 1) * nq + c] - clip.qpos[k * nq + c]) * a;
    nlerp(clip.qpos, k * nq + 3, clip.qpos, (k + 1) * nq + 3, a, q, 3);
    for (let c = 0; c < 4; c++) this.floorFeet[c] = clip.feet[k * 4 + c] + (clip.feet[(k + 1) * 4 + c] - clip.feet[k * 4 + c]) * a;
    return { q, feet: this.floorFeet };
  }

  /** Going down from the pose balance was lost in to the landing, then the way up (Choreography._recover_target). */
  private recoverTarget(ahead: number): { q: number[]; feet: number[] } {
    const r = this.recover!;
    const { fallTime } = this.files.reference.floor!;
    const t = r.t + ahead;
    const out = this.sampleFloor(r.move, r.side, Math.max(t - fallTime, 0));
    placePose(r.place, out.q, out.feet);
    if (t >= fallTime) return out;
    // Fast at first, settling into the landing, but never ahead of gravity: the hips drop no further than
    // they would falling freely from their speed at the start. All joints fold into the landing pose at once.
    const x = Math.min(t / fallTime, 1);
    const reach = 1 - (1 - x) ** 2;
    const q = out.q;
    const drop = r.onset[1] - q[1];
    const fallen = Math.max(-r.sink * t + 0.5 * GRAVITY * t * t, 0);
    const b = drop > 0.01 ? Math.min(reach, fallen / drop) : reach;
    const landing = q.slice(3, 7);
    for (let c = 0; c < q.length; c++) q[c] = r.onset[c] + (q[c] - r.onset[c]) * (this.armQpos.has(c) ? reach : b);
    nlerp(r.onset, 3, landing, 0, b, q, 3);
    // Blended joint by joint, the legs can swing through the floor on the way: the pose rises out of it.
    q[1] += this.sinking(q);
    return out;
  }

  /** How far (m) a pose's feet, knees or hands are below the floor (GingaEnv._sinking). */
  private sinking(q: number[]): number {
    (this.groundData.qpos as Float64Array).set(q);
    this.mujoco.mj_kinematics(this.model, this.groundData);
    const sites = this.groundData.site_xpos as Float64Array;
    const { clear } = this.files.reference.floor!.ground;
    let low = 0;
    this.groundSites.forEach((id, i) => (low = Math.min(low, sites[3 * id + 1] - clear[i])));
    return -low;
  }

  /** Losing balance: land on hands and knees (going forwards, or bent far forward), in the queda de quatro
   * (backwards) or the negativa (sideways), as the body faces (Choreography.begin_recover, GingaEnv.lean). */
  private beginRecover() {
    this.stuckFor = 0;
    const q = this.data.qpos as Float64Array;
    const v = this.data.qvel as Float64Array;
    const com = this.data.subtree_com as Float64Array;
    const sites = this.data.site_xpos as Float64Array;
    // Where the centre of mass is past the middle of the feet, plus the hips' drift.
    const lean = [3, 5].map((axis, i) => {
      const feet = this.soles.reduce((sum, [heel, toe]) => sum + (sites[3 * heel + axis - 3] + sites[3 * toe + axis - 3]) / 4, 0);
      return com[axis] - feet + 0.3 * v[2 * i];
    });
    const yaw = yawOf(q[3], q[4], q[5], q[6]);
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const sideWay = c * lean[0] - s * lean[1];
    const forward = s * lean[0] + c * lean[1];
    // How far the trunk is bent forward (GingaEnv.bent): the hips-to-head line from the vertical.
    const xpos = this.data.xpos as Float64Array;
    const up = [0, 1, 2].map((i) => xpos[3 * this.headId + i] - q[i]);
    const bent = Math.atan2(s * up[0] + c * up[2], up[1]);
    let move: Landing;
    let side: Side;
    if (bent > this.files.reference.floor!.bent) {
      move = "joelhos";
      side = this.phase < 0.25 || this.phase >= 0.75 ? "r" : "l";
    } else if (Math.abs(forward) >= Math.abs(sideWay)) {
      move = forward > 0 ? "joelhos" : "quatro";
      side = this.phase < 0.25 || this.phase >= 0.75 ? "r" : "l";
    } else {
      // Fold the leg that carries the weight: the foot nearest where the body is going over (negativa_r folds
      // the left leg).
      const feet = this.soles.map(([heel, toe]) => [0, 2].map((a) => (sites[3 * heel + a] + sites[3 * toe + a]) / 2));
      const mid = [0, 1].map((a) => (feet[0][a] + feet[1][a]) / 2);
      const [dl, dr] = feet.map((f) => Math.hypot(f[0] - mid[0] - lean[0], f[1] - mid[1] - lean[1]));
      move = "negativa";
      side = dl <= dr ? "r" : "l";
    }
    const landing = this.sampleFloor(move, side, 0).q;
    const norm = Math.hypot(lean[0], lean[1]);
    const tx = q[0] + (norm > 1e-6 ? (0.25 * lean[0]) / norm : 0);
    const tz = q[2] + (norm > 1e-6 ? (0.25 * lean[1]) / norm : 0);
    const place: Placement = [0, 0, yaw];
    const [lx, lz] = placeXZ(place, landing[0], landing[2]);
    place[0] = tx - lx;
    place[1] = tz - lz;
    this.move = null;
    this.pending = null;
    this.maneuver = null;
    this.pendingFeintGinga = false;
    this.recover = { move, side, t: 0, onset: Array.from(q), place, sink: v[1] };
    this.notify();
  }

  private braced = false;
  private arms: { id: number; relaxed: [number, number, number]; braced: [number, number, number] }[] | null = null;

  /** Arms braced (env.STRONG_ARMS) or relaxed (GingaEnv.brace): stiffness, critical damping and torque limit. */
  private brace(on: boolean) {
    const strong = this.files.reference.floor?.strongArms;
    if (on === this.braced || !strong) return;
    this.braced = on;
    const m = this.model;
    const gain = m.actuator_gainprm as Float64Array;
    const bias = m.actuator_biasprm as Float64Array;
    const force = m.actuator_forcerange as Float64Array;
    const G = gain.length / m.nu;
    const B = bias.length / m.nu;
    if (!this.arms) {
      this.arms = [];
      for (let i = 0; i < m.nu; i++) {
        const name = m.actuator(i).name;
        const joint = name.startsWith("shoulder") ? "shoulder" : name.startsWith("elbow") ? "elbow" : null;
        if (!joint) continue;
        const [kp, kv, f] = [gain[G * i], bias[B * i + 2], force[2 * i + 1]];
        const [strongKp, strongF] = strong[joint];
        this.arms.push({ id: i, relaxed: [kp, kv, f], braced: [strongKp, kv * Math.sqrt(strongKp / kp), strongF] });
      }
    }
    for (const arm of this.arms) {
      const [kp, kv, f] = on ? arm.braced : arm.relaxed;
      gain[G * arm.id] = kp;
      bias[B * arm.id + 1] = -kp;
      bias[B * arm.id + 2] = kv;
      force[2 * arm.id] = -f;
      force[2 * arm.id + 1] = f;
    }
  }

  /** How fast the reference moves on: slower on the way up while the body lags behind it (GingaEnv.pace). */
  private pace() {
    const floor = this.files.reference.floor;
    if (!this.recover || !floor || this.recover.t < floor.fallTime) return 1;
    const q = this.data.qpos as Float64Array;
    const ref = this.target(0).q;
    const lag = Math.hypot(q[0] - ref[0], q[1] - ref[1], q[2] - ref[2]);
    return Math.min(Math.max(1 - (lag - floor.waitFrom) / floor.waitSpan, floor.minPace), 1);
  }

  /** The way up starts where the body actually landed (Choreography.land). */
  private land() {
    const r = this.recover!;
    const q = this.data.qpos as Float64Array;
    const landing = this.sampleFloor(r.move, r.side, 0).q;
    const [lx, lz] = placeXZ([0, 0, r.place[2]], landing[0], landing[2]);
    r.place[0] = q[0] - lx;
    r.place[1] = q[2] - lz;
  }

  /** Balance is lost (GingaEnv.trouble): a bad contact, the head low, or (outside an active move) the hips tipped, sunk or away. */
  private trouble() {
    const q = this.data.qpos as Float64Array;
    const ref = this.target(0).q;
    const { troubleTilt, troubleDrop } = this.files.reference.floor!;
    // The angle between the hips' up axis and the reference's (heading aside).
    const R = this.rot;
    quatMatrix(q[3], q[4], q[5], q[6], R);
    const up = [R[1], R[4], R[7]];
    quatMatrix(ref[3], ref[4], ref[5], ref[6], R);
    const dot = up[0] * R[1] + up[1] * R[4] + up[2] * R[7];
    const inMove = this.move !== null || this.maneuver !== null || this.sinceKick < 0.5;
    return (
      this.contacts().bad ||
      this.low(ref) ||
      (!inMove &&
        (Math.acos(Math.min(1, Math.max(-1, dot))) > troubleTilt ||
          q[1] < ref[1] - troubleDrop ||
          Math.hypot(q[0] - ref[0], q[2] - ref[2]) > 0.5))
    );
  }

  /** Hips or head too low for anything but a fall (GingaEnv.low), against the reference pose `ref`. */
  private low(ref: ArrayLike<number>) {
    const { hips, head, below } = this.files.policy.low ?? { hips: 0.4, head: 0.8, below: 0.3 };
    const headBelow = this.move && MOVE_NETWORK[this.move] === "esquiva" ? below + 0.08 : below;
    (this.groundData.qpos as Float64Array).set(ref);
    this.mujoco.mj_kinematics(this.model, this.groundData);
    const refHead = (this.groundData.xpos as Float64Array)[3 * this.headId + 1];
    const q = this.data.qpos as Float64Array;
    const ownHead = (this.data.xpos as Float64Array)[3 * this.headId + 1];
    return q[1] < Math.min(hips, ref[1] - below) || ownHead < Math.min(head, refHead - headBelow);
  }

  /** The ginga or kick in the triangle's own frame (Choreography._target). */
  private ownTarget(ahead: number): { q: number[]; feet: number[] } {
    if (!this.move) {
      const phase = this.phase + ahead / this.period;
      return { q: this.sampleReference(phase, this.bpm, this.refNext), feet: this.feetAt(phase, this.bpm, this.refFeet) };
    }
    const g = this.sampleReference(this.phase, this.bpm, this.ginga);
    const gf = this.feetAt(this.phase, this.bpm, this.refFeet);
    const clip = this.clip(this.move, this.side);
    const fps = clip.fps ?? this.files.reference.kicks.fps!;
    const t = this.kickT + ahead;
    const nq = this.model.nq;
    const frames = clip.qpos.length / nq;
    const x = Math.min(Math.max(t, 0), clip.duration) * fps;
    const k = Math.min(Math.floor(x), frames - 2);
    const a = x - k;
    const kq = this.kickPose;
    for (let c = 0; c < nq; c++) kq[c] = clip.qpos[k * nq + c] + (clip.qpos[(k + 1) * nq + c] - clip.qpos[k * nq + c]) * a;
    nlerp(clip.qpos, k * nq + 3, clip.qpos, (k + 1) * nq + 3, a, kq, 3);
    kq[0] += this.anchor[0];
    kq[2] += this.anchor[1];
    for (let c = 0; c < 4; c++) this.kickFeet[c] = clip.feet[k * 4 + c] + (clip.feet[(k + 1) * 4 + c] - clip.feet[k * 4 + c]) * a + this.anchor[c % 2];
    const b = this.blend(t);
    const q = this.refNext;
    for (let c = 0; c < nq; c++) q[c] = g[c] + (kq[c] - g[c]) * b;
    nlerp(g, 3, kq, 3, b, q, 3);
    for (let c = 0; c < 4; c++) gf[c] += (this.kickFeet[c] - gf[c]) * b;
    return { q, feet: gf };
  }

  /** World xz of the support foot (0 = left foot while right leg is back in 0..0.5, 1 = right foot in 0.5..1). */
  private supportFootXZ(supportIndex?: 0 | 1): [number, number] {
    const idx = supportIndex ?? (wrap(this.phase) < 0.5 ? 0 : 1);
    const [heel, toe] = this.soles[idx];
    const sites = this.data.site_xpos as Float64Array;
    return [(sites[3 * heel] + sites[3 * toe]) / 2, (sites[3 * heel + 2] + sites[3 * toe + 2]) / 2];
  }

  /** Rotate the humanoid and its reference placement around a floor pivot (px, pz) by dYaw (radians). */
  private rotateAroundPivot(dYaw: number, px: number, pz: number) {
    if (Math.abs(dYaw) < 1e-7) return;
    const q = this.data.qpos as Float64Array;
    const v = this.data.qvel as Float64Array;
    const c = Math.cos(dYaw);
    const s = Math.sin(dYaw);
    const dx = q[0] - px;
    const dz = q[2] - pz;
    q[0] = px + c * dx + s * dz;
    q[2] = pz - s * dx + c * dz;
    const hw = Math.cos(dYaw / 2);
    const hy = Math.sin(dYaw / 2);
    const [w, x, y, z] = [q[3], q[4], q[5], q[6]];
    q[3] = hw * w - hy * y;
    q[4] = hw * x + hy * z;
    q[5] = hw * y + hy * w;
    q[6] = hw * z - hy * x;
    const vx = v[0];
    const vz = v[2];
    v[0] = c * vx + s * vz;
    v[2] = -s * vx + c * vz;
    const pdx = this.placement[0] - px;
    const pdz = this.placement[1] - pz;
    this.placement = [px + c * pdx + s * pdz, pz - s * pdx + c * pdz, this.placement[2] + dYaw];
    this.mujoco.mj_forward(this.model, this.data);
  }

  /** Apply continuous navigation (walk/strafe/turn/orbit) while gingando. */
  private stepNavigation(dt: number) {
    let fwd = this.nav.forward || (this.nudgeInput.timer > 0 ? this.nudgeInput.forward : 0);
    let side = this.nav.side || (this.nudgeInput.timer > 0 ? this.nudgeInput.side : 0);
    let turn = this.nav.turn || (this.nudgeInput.timer > 0 ? this.nudgeInput.turn : 0);
    if (this.nudgeInput.timer > 0) this.nudgeInput.timer = Math.max(0, this.nudgeInput.timer - dt);
    if (this.nav.orbit) {
      const q = this.data.qpos as Float64Array;
      const dist = Math.hypot(q[0], q[2]);
      const desiredYaw = dist > 0.25 ? Math.atan2(-q[0], -q[2]) : this.placement[2] + 0.65 * dt;
      const yawErr = Math.atan2(Math.sin(desiredYaw - this.placement[2]), Math.cos(desiredYaw - this.placement[2]));
      if (!side) side = 0.75;
      if (!fwd) fwd = Math.max(-0.5, Math.min(0.5, (dist - 1.0) * 0.8));
      if (!turn) turn = Math.max(-1, Math.min(1, yawErr * 1.8));
    }
    if (fwd === 0 && side === 0 && turn === 0) return;
    if (turn !== 0) {
      const swing = Math.sin(2 * Math.PI * this.phase) ** 2;
      const [px, pz] = this.supportFootXZ();
      this.rotateAroundPivot(turn * 1.15 * swing * dt, px, pz);
    }
    if (fwd !== 0 || side !== 0) {
      const base = this.reanchor();
      const yaw = base[2];
      const leadFwd = 0.055 * fwd;
      const leadSide = 0.05 * side;
      const wx = Math.sin(yaw) * leadFwd + Math.cos(yaw) * leadSide;
      const wz = Math.cos(yaw) * leadFwd - Math.sin(yaw) * leadSide;
      this.placement = [base[0] + wx, base[1] + wz, yaw];
    }
  }

  /** Move the choreography on by dt (Choreography.advance). */
  private advanceChoreography(dt: number) {
    if (this.recover) {
      const r = this.recover;
      const { fallTime } = this.files.reference.floor!;
      const falling = r.t < fallTime;
      r.t += dt;
      if (falling && r.t >= fallTime) this.land();
      if (r.t >= fallTime + this.floorClip(r.move, r.side).duration) {
        // Up in the base: the ginga carries on from here, its triangle where the body stands.
        this.placement = [...r.place];
        this.phase = this.floorClip(r.move, r.side).phase;
        this.recover = null;
        this.notify();
      }
      return;
    }
    if (this.move) {
      this.kickT += dt;
      if (this.maneuver?.type === "role" && this.kickT >= 0.25 && this.kickT <= 1.55) {
        const w = 1 - Math.cos((2 * Math.PI * (this.kickT - 0.25)) / 1.3);
        const dir = this.side === "r" ? 1 : -1;
        const [px, pz] = this.supportFootXZ(this.side === "r" ? 0 : 1);
        this.rotateAroundPivot(dir * ((1.5 * Math.PI) / 1.3) * w * dt, px, pz);
        const yaw = this.placement[2];
        this.placement = [
          this.placement[0] + Math.cos(yaw) * dir * 0.22 * w * dt,
          this.placement[1] - Math.sin(yaw) * dir * 0.22 * w * dt,
          yaw,
        ];
      } else if (this.maneuver?.type === "passagem-de-negativa" && this.kickT >= 0.3 && this.kickT <= 1.35) {
        const w = 1 - Math.cos((2 * Math.PI * (this.kickT - 0.3)) / 1.05);
        const dir = this.side === "r" ? -1 : 1;
        const [px, pz] = this.supportFootXZ(this.side === "r" ? 0 : 1);
        this.rotateAroundPivot(dir * ((0.5 * Math.PI) / 1.05) * w * dt, px, pz);
        const yaw = this.placement[2];
        this.placement = [
          this.placement[0] + Math.sin(yaw) * 0.14 * w * dt,
          this.placement[1] + Math.cos(yaw) * 0.14 * w * dt,
          yaw,
        ];
      }
      if (this.kickT >= this.clip(this.move, this.side).duration) {
        if (this.maneuver?.type === "passagem-de-negativa") this.phase = wrap(this.phase + 0.5);
        this.placement = this.reanchor();
        this.move = null;
        this.maneuver = null;
        this.notify();
      }
      return;
    }
    if (this.maneuver && !this.pending) {
      const m = this.maneuver;
      m.t += dt;
      if (m.type === "troca-de-base") {
        const w = 0.5 * (1 - Math.cos((2 * Math.PI * m.t) / m.duration));
        const effPeriod = cyclePeriod(Math.min(130, this.bpm + 38 * w));
        const next = this.phase + dt / effPeriod;
        if (next >= 1) this.cycles++;
        this.phase = wrap(next);
        const base = this.reanchor();
        const yaw = base[2];
        this.placement = [base[0] + Math.sin(yaw) * 0.035 * w, base[1] + Math.cos(yaw) * 0.035 * w, yaw];
      } else if (m.type === "giro-de-base") {
        const w = 1 - Math.cos((2 * Math.PI * m.t) / m.duration);
        const dir = m.side === "r" ? 1 : -1;
        const [px, pz] = this.supportFootXZ();
        this.rotateAroundPivot(dir * ((2 * Math.PI) / m.duration) * w * dt, px, pz);
        const next = this.phase + dt / this.period;
        if (next >= 1) this.cycles++;
        this.phase = wrap(next);
        const base = this.reanchor();
        this.placement = [base[0], base[1], this.placement[2]];
      }
      if (m.t >= m.duration) {
        this.placement = this.reanchor();
        this.maneuver = null;
        this.notify();
      }
      return;
    }
    const step = dt / this.period;
    if (this.pending && wrap(this.pending.entry - this.phase) <= step) {
      this.phase = this.pending.entry;
      this.move = this.pending.move;
      this.side = this.pending.side;
      this.kickT = 0;
      this.pending = null;
      this.anchor = this.supportAnchor();
      this.notify();
      return;
    }
    if (this.pendingFeintGinga && Math.abs((this.phase % 0.5) - 0.25) <= Math.max(step * 2, 0.025)) {
      this.phase = wrap(1 - this.phase);
      this.pendingFeintGinga = false;
      this.notify();
      return;
    }
    const next = this.phase + step;
    if (next >= 1) this.cycles++;
    this.phase = wrap(next);
    this.stepNavigation(dt);
  }

  /** Where the ginga's triangle goes when a kick or an esquiva ends: under the feet as they stand, without turning
   * (Choreography.reanchor). */
  private reanchor(): Placement {
    const local = this.feetAt(this.phase, this.bpm, [0, 0, 0, 0]);
    const sites = this.data.site_xpos as Float64Array;
    const [l0x, l0z] = placeXZ(this.placement, local[0], local[1]);
    const [l1x, l1z] = placeXZ(this.placement, local[2], local[3]);
    let fx = 0;
    let fz = 0;
    for (const [heel, toe] of this.soles) {
      fx += (sites[3 * heel] + sites[3 * toe]) / 4;
      fz += (sites[3 * heel + 2] + sites[3 * toe + 2]) / 4;
    }
    return [this.placement[0] + fx - (l0x + l1x) / 2, this.placement[1] + fz - (l0z + l1z) / 2, this.placement[2]];
  }

  /** How far the support foot stands from where the kick plans it, capped at ANCHOR_LIMIT. */
  private supportAnchor(): [number, number] {
    const support = this.side === "r" ? 0 : 1;
    const planned = this.clip(this.move!, this.side).feet;
    const [heel, toe] = this.soles[support];
    const sites = this.data.site_xpos as Float64Array;
    const [fx, fz] = unplaceXZ(this.placement, (sites[3 * heel] + sites[3 * toe]) / 2, (sites[3 * heel + 2] + sites[3 * toe + 2]) / 2);
    const dx = fx - planned[2 * support];
    const dz = fz - planned[2 * support + 1];
    const scale = Math.min(1, ANCHOR_LIMIT / (Math.hypot(dx, dz) || Infinity));
    return [dx * scale, dz * scale];
  }

  private features(): number[] {
    const r = this.recover;
    const extraZeros = EXTRA_MOVES.map(() => 0);
    if (r) {
      const total = this.files.reference.floor!.fallTime + this.floorClip(r.move, r.side).duration;
      return [0, 0, 0, 0, 0, 1, r.move === "quatro" ? 1 : 0, r.side === "r" ? 1 : -1, r.t / total, r.move === "joelhos" ? 1 : 0, 0, ...extraZeros];
    }
    if (!this.move) return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...extraZeros];
    const duration = this.clip(this.move, this.side).duration;
    const kick = [this.move === "martelo" ? 1 : 0, this.move === "bencao" ? 1 : 0, this.side === "r" ? 1 : -1, this.kickT / duration, this.blend(this.kickT)];
    return [...kick, 0, 0, 0, 0, 0, this.move === "esquiva" ? 1 : 0, ...EXTRA_MOVES.map((m) => (this.move === m ? 1 : 0))];
  }

  // ---- observation, policy, control (port of env.GingaEnv) -------------

  private contacts(): { feet: [number, number]; bad: boolean } {
    const feet: [number, number] = [0, 0];
    let bad = false;
    const list = this.data.contact;
    const n = this.data.ncon;
    for (let i = 0; i < n; i++) {
      const c = list.get(i)!;
      const other = c.geom1 === this.floor ? c.geom2 : c.geom2 === this.floor ? c.geom1 : -1;
      c.delete();
      if (other < 0) continue;
      const f = this.feetGeoms.indexOf(other);
      if (f >= 0) feet[f] = 1;
      else if (this.never.has(other) || (!this.recover && !this.allowed.has(other))) bad = true;
    }
    list.delete();
    return { feet, bad };
  }

  /** The policy's input, exactly as ml/env.py builds it. */
  observe(): Float64Array {
    const d = this.data;
    const q = d.qpos as Float64Array;
    const v = d.qvel as Float64Array;
    const xpos = d.xpos as Float64Array;
    const sites = d.site_xpos as Float64Array;
    const { nq, nv } = this.model;
    const o = this.obs;
    let n = 0;
    // Heading frame (yaw_matrix): world vectors into the body's heading.
    const yaw = yawOf(q[3], q[4], q[5], q[6]);
    const F = this.frame;
    F.fill(0);
    F[0] = F[8] = Math.cos(yaw);
    F[2] = -Math.sin(yaw);
    F[6] = Math.sin(yaw);
    F[4] = 1;
    const R = this.rot;
    quatMatrix(q[3], q[4], q[5], q[6], R);
    const { q: ref, feet: target } = this.target(this.controlDt);
    const { feet } = this.contacts();
    const rotate = (x: number, y: number, z: number, scale: number) => {
      o[n++] = (F[0] * x + F[1] * y + F[2] * z) * scale;
      o[n++] = (F[3] * x + F[4] * y + F[5] * z) * scale;
      o[n++] = (F[6] * x + F[7] * y + F[8] * z) * scale;
    };

    o[n++] = Math.sin(2 * Math.PI * this.phase);
    o[n++] = Math.cos(2 * Math.PI * this.phase);
    o[n++] = q[1];
    // (frame @ R)[:, :2], row-major.
    for (let r = 0; r < 3; r++)
      for (let col = 0; col < 2; col++) o[n++] = F[3 * r] * R[col] + F[3 * r + 1] * R[3 + col] + F[3 * r + 2] * R[6 + col];
    for (let k = 7; k < nq; k++) o[n++] = q[k];
    for (let k = 6; k < nv; k++) o[n++] = 0.1 * v[k];
    rotate(v[0], v[1], v[2], 1);
    // Freejoint angular velocity v[3:6] is already in body-local coordinates.
    o[n++] = 0.2 * v[3];
    o[n++] = 0.2 * v[4];
    o[n++] = 0.2 * v[5];
    for (const id of this.effectorIds) rotate(xpos[3 * id] - q[0], xpos[3 * id + 1] - q[1], xpos[3 * id + 2] - q[2], 1);
    o[n++] = feet[0];
    o[n++] = feet[1];
    for (let k = 7; k < nq; k++) o[n++] = ref[k] - q[k];
    rotate(ref[0] - q[0], ref[1] - q[1], ref[2] - q[2], 1);
    this.soles.forEach(([heel, toe], f) => {
      const ox = target[2 * f] - (sites[3 * heel] + sites[3 * toe]) / 2;
      const oz = target[2 * f + 1] - (sites[3 * heel + 2] + sites[3 * toe + 2]) / 2;
      o[n++] = F[0] * ox + F[2] * oz;
      o[n++] = F[6] * ox + F[8] * oz;
    });
    o[n++] = tempoInput(this.bpm);
    for (const f of this.features()) if (n < o.length) o[n++] = f;
    if (n !== o.length) throw new Error(`observation has ${n} values, the policy expects ${o.length}`);
    return o;
  }

  /** The last action taken (tests compare it with Python's). */
  readonly lastAction = new Float64Array(64);

  /** Seconds since a kick was last under way (Brain.since_kick). */
  sinceKick = Infinity;

  /** Seconds since the body was last falling or getting up (Brain.since_recover). */
  sinceRecover = Infinity;

  /** The network that played the last kick or esquiva (Brain.mover). */
  private mover: "kicks" | "esquiva" | null = null;

  /** The network in charge (Brain.network): the recovery network while down and a moment after, the kick (or
   * esquiva) network from the start of a kick (or esquiva) until a moment after it, else the ginga network. */
  get network(): "ginga" | "kicks" | "recover" | "esquiva" {
    const { networks, handback, handbackKicks, handbackEsquiva } = this.files.policy;
    if (networks.recover && this.sinceRecover < handback) return "recover";
    const moverHandback = (this.mover === "esquiva" ? handbackEsquiva : handbackKicks) ?? handback;
    return this.mover && networks[this.mover] && this.sinceKick < moverHandback ? this.mover : "ginga";
  }

  /** A network's mean action for an observation (actor MLP, ELU between layers). Each network reads as
   * many inputs as it was trained on: later skills appended theirs at the end. */
  decide(obs: Float64Array, name: "ginga" | "kicks" | "recover" | "esquiva" = this.network): Float64Array {
    const { policy, weights } = this.files;
    const net = policy.networks[name]!;
    let x = this.input;
    const inputs = net.layers[0].in;
    for (let i = 0; i < inputs; i++) x[i] = Math.min(Math.max((obs[i] - net.mean[i]) / net.std[i], -policy.clip), policy.clip);
    net.layers.forEach((layer, l) => {
      const y = this.layers[l];
      const bias = layer.offset + layer.in * layer.out;
      const last = l === net.layers.length - 1;
      for (let o = 0; o < layer.out; o++) {
        let sum = weights[bias + o];
        const row = layer.offset + o * layer.in;
        for (let i = 0; i < layer.in; i++) sum += weights[row + i] * x[i];
        y[o] = !last && sum < 0 ? Math.expm1(sum) : sum;
      }
      x = y;
    });
    return x;
  }

  /** One control step: observe, decide, set the PD targets, simulate 0.03 s, move the cycle on. */
  control() {
    const { mujoco, model: m, data: d, files } = this;
    const obs = this.observe();
    if (this.move) this.mover = MOVE_NETWORK[this.move];
    this.sinceKick = this.move ? 0 : this.sinceKick + this.controlDt;
    this.sinceRecover = this.recover ? 0 : this.sinceRecover + this.controlDt;
    const action = this.decide(obs);
    this.lastAction.set(action);
    const target = this.target(this.controlDt).q;
    const ctrl = d.ctrl as Float64Array;
    for (let i = 0; i < m.nu; i++) {
      const value = target[7 + i] + files.policy.actionScale * Math.min(Math.max(action[i], -1), 1);
      ctrl[i] = Math.min(Math.max(value, this.lo[i]), this.hi[i]);
    }
    const force = d.xfrc_applied as Float64Array;
    force[6 * this.chestId] = this.pushLeft > 0 ? this.pushForce[0] : 0;
    force[6 * this.chestId + 2] = this.pushLeft > 0 ? this.pushForce[1] : 0;
    this.pushLeft = Math.max(this.pushLeft - 1, 0);
    this.brace(this.recover !== null);
    for (let k = 0; k < files.policy.frameSkip; k++) mujoco.mj_step(m, d);
    this.wasRecovering = this.recover !== null;
    this.advanceChoreography(this.controlDt * this.pace());
  }

  private wasRecovering = false;

  /**
   * After a step: notice a loss of balance and hand over to the recovery network (or, without one, notice
   * the fall and stand the body up again after a moment), and follow the music's bar.
   */
  private settle(barPhase: number | null) {
    const q = this.data.qpos as Float64Array;
    // A fall, once noticed, holds until the body is stood up: checking again with the ginga's rules while the
    // recovery is still under way cleared it every other step, and the stand-up timer kept starting over.
    if (this.fallen) {
      // Down until GET_UP_AFTER.
    } else if (this.canRecover) {
      if (this.recover) {
        // Down and up again: a failure only if the head hits the floor, or, once landed, the body is lost or stuck.
        const ref = this.target(0).q;
        const { fallTime, lost, minPace, stuck } = this.files.reference.floor!;
        const landed = this.recover.t >= fallTime;
        const away = landed && Math.hypot(q[0] - ref[0], q[2] - ref[2]) > lost;
        this.stuckFor = landed && this.pace() <= minPace ? this.stuckFor + this.controlDt : 0;
        this.fallen = this.contacts().bad || away || q[1] < 0.05 || this.stuckFor >= stuck;
      } else if (!this.wasRecovering && this.trouble()) {
        this.beginRecover();
      }
    } else {
      this.fallen = this.contacts().bad || this.low(this.target(0).q);
    }
    this.fallenFor = this.fallen ? this.fallenFor + this.controlDt : 0;
    // Stood up on the beat, if the music plays.
    if (this.fallenFor > GET_UP_AFTER) this.reset(barPhase ?? this.phase);
    if (barPhase !== null && !this.busy) {
      // Nudge the cycle toward the bar, the shortest way round (not while a kick is coming or under way).
      const error = wrap(barPhase - this.phase + 0.5) - 0.5;
      const limit = (SYNC_LIMIT * this.controlDt) / this.period;
      this.phase = wrap(this.phase + Math.min(Math.max(SYNC_GAIN * error, -limit), limit));
    }
  }

  /** Advance real time (scaled by speed), running as many control steps as fit. */
  update(dt: number, bpm: number, barPhase: number | null) {
    this.bpm = Math.min(Math.max(bpm, this.files.policy.minBpm), this.files.policy.maxBpm);
    if (!this.playing) return;
    this.accumulated = Math.min(this.accumulated + dt * this.speed, 0.25);
    while (this.accumulated >= this.controlDt) {
      this.accumulated -= this.controlDt;
      this.control();
      this.settle(barPhase);
    }
  }

  /** One control step while paused (the transport's step button). */
  stepOnce() {
    this.control();
    this.settle(null);
  }

  get status() {
    const moveLabel = this.maneuver
      ? MANEUVER_NAMES[this.maneuver.type]
      : this.move
        ? `${this.feintMode && MOVE_NETWORK[this.move] === "kicks" ? "Finta de " : ""}${MOVE_NAMES[this.move]} · ${which(this.move, this.side)}`
        : null;
    const pendingLabel = this.pending
      ? this.maneuver
        ? `${MANEUVER_NAMES[this.maneuver.type]} · esperando o momento`
        : `${MOVE_NAMES[this.pending.move]} · ${which(this.pending.move, this.pending.side)} · esperando o momento`
      : null;
    return {
      time: this.phase * this.period,
      duration: this.period,
      keyframe: Math.floor(this.phase * 4),
      label: this.fallen
        ? "física em tempo real · caiu — recomeçando"
        : this.recover
          ? this.recover.t < this.files.reference.floor!.fallTime
            ? `caindo · ${LANDING_NAMES[this.recover.move]}`
            : `levantando · ${FROM_LANDING[this.recover.move]}`
          : pendingLabel
            ? pendingLabel
            : moveLabel
              ? moveLabel
              : `física em tempo real · ciclo ${this.cycles + 1} · ${Math.round(this.bpm)} bpm`,
      mirrored: false,
      resting: false,
    };
  }

  private effectiveGingaStyle(): "angola" | "padrao" | "regional" {
    if (this.gingaStyle !== "auto") return this.gingaStyle;
    if (this.bpm <= 76) return "angola";
    if (this.bpm >= 104) return "regional";
    return "padrao";
  }

  private styleJointOffset(jointName: string, w: number, style: "angola" | "padrao" | "regional"): number {
    if (w <= 0 || style === "padrao") return 0;
    if (style === "angola") {
      switch (jointName) {
        case "spine_flex":
          return 0.18 * w;
        case "chest_flex":
          return 0.14 * w;
        case "spine_side":
          return 0.08 * Math.sin(2 * Math.PI * this.phase) * w;
        case "neck_flex":
          return -0.28 * w;
        case "shoulder_l_abd":
        case "shoulder_r_abd":
          return 0.24 * w;
        case "elbow_l":
        case "elbow_r":
          return -0.18 * w;
        case "hip_l_flex":
        case "hip_r_flex":
          return 0.15 * w;
        case "knee_l":
        case "knee_r":
          return 0.3 * w;
        case "ankle_l":
        case "ankle_r":
          return -0.15 * w;
        default:
          return 0;
      }
    }
    // Regional: upright trunk, compact high guard.
    switch (jointName) {
      case "spine_flex":
        return -0.11 * w;
      case "chest_flex":
        return -0.08 * w;
      case "neck_flex":
        return 0.12 * w;
      case "shoulder_l_flex":
      case "shoulder_r_flex":
        return 0.18 * w;
      case "shoulder_l_abd":
      case "shoulder_r_abd":
        return -0.14 * w;
      case "elbow_l":
      case "elbow_r":
        return 0.26 * w;
      case "hip_l_flex":
      case "hip_r_flex":
        return -0.05 * w;
      case "knee_l":
      case "knee_r":
        return -0.1 * w;
      case "ankle_l":
      case "ankle_r":
        return 0.05 * w;
      default:
        return 0;
    }
  }

  /** Pose the rig from the simulation state: root from the free joint, each bone the product of its hinges. */
  apply(rig: Rig) {
    const q = this.data.qpos as Float64Array;
    for (const bone of Object.values(rig.bones)) bone.quaternion.identity();
    const style = this.effectiveGingaStyle();
    const gingaWeight = this.recover ? 0 : this.move ? Math.max(0, 1 - this.blend(this.kickT) * 1.6) : 1;
    const hipYOffset = style === "angola" ? -0.042 * gingaWeight : style === "regional" ? 0.012 * gingaWeight : 0;
    const hips = rig.bones.hips;
    hips.position.set(q[0], q[1] + hipYOffset, q[2]);
    hips.quaternion.set(q[4], q[5], q[6], q[3]).normalize();
    for (const { bone, joints } of this.bodies) {
      const quat = rig.bones[bone].quaternion.identity();
      for (const j of joints) {
        const joint = this.joints[j];
        const angle = q[joint.adr] + this.styleJointOffset(joint.name, gingaWeight, style);
        quat.multiply(hinge.setFromAxisAngle(axis.fromArray(joint.axis), angle));
      }
    }
    rig.object.updateMatrixWorld(true);
  }
}
