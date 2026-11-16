import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";
import type { BoneName, Rig } from "../rig/rig";
import { cyclePeriod } from "../music/toques";
import { sampleReference } from "./reference";
import { advanceChoreography, target } from "./choreography";
import { brace } from "./floor";
import { pace, settle } from "./balance";
import { observe } from "./observation";
import { activeNetwork, decide } from "./policy";
import { poseRig } from "./pose";
import { available, request, requestManeuver } from "./requests";
import { wrap } from "./math";
import { MOVE_NETWORK } from "./types";
import type { Landing, LiveFiles, Maneuver, Move, Placement, Side } from "./types";

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

/** How fast the body's tempo follows a new toque (bpm per second): a jump of the tempo under the feet topples it. */
const TEMPO_GLIDE = 6;

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
  /** The standing maneuver under way (giro-de-base). */
  maneuver: { type: Maneuver; t: number; duration: number; side: Side; startYaw: number } | null = null;
  /** Continuous navigation controls (-1..1) and circular roda orbit toggle. */
  nav = { forward: 0, side: 0, turn: 0, orbit: false };
  nudgeInput = { forward: 0, side: 0, turn: 0, timer: 0 };
  /** Falling and getting up (Choreography.recover): landing, side, time since balance was lost, the pose
   * it was lost in, and where the landing is placed. */
  /** sink: the hips' vertical speed when balance was lost (m/s, up positive). */
  recover: { move: Landing; side: Side; t: number; onset: number[]; place: Placement; sink: number } | null = null;
  /** A push in progress: horizontal force on the chest (N) and control steps left. */
  pushForce: [number, number] = [0, 0];
  pushLeft = 0;
  listeners = new Set<() => void>();
  fallenFor = 0;
  /** qpos addresses of the shoulders and elbows (Choreography.arms). */
  readonly armQpos = new Set<number>();
  /** How long the way up has been waiting at its slowest for a body lying on the floor (env.STUCK). */
  stuckFor = 0;
  accumulated = 0;
  readonly controlDt: number;
  readonly joints: { name: string; body: BoneName; axis: number[]; adr: number }[];
  readonly bodies: { bone: BoneName; joints: number[] }[];
  readonly effectorIds: number[];
  readonly headId: number;
  readonly floor: number;
  readonly feetGeoms: number[];
  readonly allowed: Set<number>;
  readonly never: Set<number>;
  readonly chestId: number;
  readonly floorPose: number[];
  readonly floorFeet: number[] = [0, 0, 0, 0];
  readonly soles: [number, number][];
  readonly groundData: MjData;
  readonly groundSites: number[];
  readonly lo: Float64Array;
  readonly hi: Float64Array;
  readonly obs: Float64Array;
  readonly input: Float64Array;
  readonly layers: Float64Array[];
  readonly refNext: number[];
  readonly refFeet: number[] = [0, 0, 0, 0];
  readonly ginga: number[];
  readonly kickPose: number[];
  readonly kickFeet: number[] = [0, 0, 0, 0];
  readonly frame: number[] = new Array(9).fill(0);
  readonly rot: number[] = new Array(9).fill(0);

  constructor(
    readonly mujoco: MainModule,
    readonly files: LiveFiles,
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
    this.nudgeInput.timer = 0;
    this.placement = [0, 0, 0];
    this.pushLeft = 0;
    // Standing again is the ginga's (Brain.reset): no kick or recovery network is still handing back.
    this.sinceKick = Infinity;
    this.sinceRecover = Infinity;
    this.wasRecovering = false;
    if (busy) this.notify();
    (d.qpos as Float64Array).set(sampleReference(this, this.phase, this.bpm, this.refNext));
    // Moving with the reference, as GingaEnv.reset: standing still mid-cycle, the ginga topples.
    const h = 1 / this.files.reference.samples;
    const before = sampleReference(this, this.phase - h, this.bpm, new Array(m.nq));
    const after = sampleReference(this, this.phase + h, this.bpm, new Array(m.nq));
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

  // ---- moves (port of env.Kicks and env.Choreography) -------------------

  /** Gingando, waiting for the moment to kick, kicking, or down (falling and getting up). */
  get mode(): "ginga" | "waiting" | "kick" | "down" {
    return this.recover ? "down" : this.move || this.maneuver ? "kick" : this.pending ? "waiting" : "ginga";
  }

  get busy() {
    return this.move !== null || this.pending !== null || this.recover !== null || this.maneuver !== null;
  }

  /** Trained to fall and get up (with a recovery network and its references). */
  get canRecover() {
    return !!(this.files.policy.networks.recover && this.files.reference.floor);
  }

  /** Push the chest for 0.2 s with a horizontal force (N, world x/z), like a partner would (GingaEnv.push). */
  push(fx: number, fz: number) {
    this.pushForce = [fx, fz];
    this.pushLeft = Math.round(0.2 / this.controlDt);
  }

  /** Called whenever the mode changes (React subscribes to enable and disable the move buttons). */
  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }

  notify() {
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

  braced = false;
  arms: { id: number; relaxed: [number, number, number]; braced: [number, number, number] }[] | null = null;

  // ---- observation, policy, control (port of env.GingaEnv) -------------

  /** The last action taken (tests compare it with Python's). */
  readonly lastAction = new Float64Array(64);

  /** Seconds since a kick was last under way (Brain.since_kick). */
  sinceKick = Infinity;

  /** Seconds since the body was last falling or getting up (Brain.since_recover). */
  sinceRecover = Infinity;

  /** The network that played the last kick or esquiva (Brain.mover). */
  mover: "kicks" | "esquiva" | null = null;

  /** One control step: observe, decide, set the PD targets, simulate 0.03 s, move the cycle on. */
  control() {
    const { mujoco, model: m, data: d, files } = this;
    const obs = this.observe();
    if (this.move) this.mover = MOVE_NETWORK[this.move];
    this.sinceKick = this.move ? 0 : this.sinceKick + this.controlDt;
    this.sinceRecover = this.recover ? 0 : this.sinceRecover + this.controlDt;
    const action = this.decide(obs);
    this.lastAction.set(action);
    const aim = target(this, this.controlDt).q;
    const ctrl = d.ctrl as Float64Array;
    for (let i = 0; i < m.nu; i++) {
      const value = aim[7 + i] + files.policy.actionScale * Math.min(Math.max(action[i], -1), 1);
      ctrl[i] = Math.min(Math.max(value, this.lo[i]), this.hi[i]);
    }
    const force = d.xfrc_applied as Float64Array;
    force[6 * this.chestId] = this.pushLeft > 0 ? this.pushForce[0] : 0;
    force[6 * this.chestId + 2] = this.pushLeft > 0 ? this.pushForce[1] : 0;
    this.pushLeft = Math.max(this.pushLeft - 1, 0);
    brace(this, this.recover !== null);
    for (let k = 0; k < files.policy.frameSkip; k++) mujoco.mj_step(m, d);
    this.wasRecovering = this.recover !== null;
    advanceChoreography(this, this.controlDt * pace(this));
  }

  wasRecovering = false;

  /** Whether `update` has set the tempo once yet. */
  private tempoSet = false;

  /** Advance real time (scaled by speed), running as many control steps as fit. */
  update(dt: number, bpm: number, barPhase: number | null) {
    if (!this.playing) return;
    const wanted = Math.min(Math.max(bpm, this.files.policy.minBpm), this.files.policy.maxBpm);
    // The body starts at the wanted tempo; after that a new toque is followed gradually.
    const glide = this.tempoSet ? TEMPO_GLIDE * dt * this.speed : Infinity;
    this.tempoSet = true;
    this.bpm += Math.min(Math.max(wanted - this.bpm, -glide), glide);
    this.accumulated = Math.min(this.accumulated + dt * this.speed, 0.25);
    while (this.accumulated >= this.controlDt) {
      this.accumulated -= this.controlDt;
      this.control();
      settle(this, barPhase);
    }
  }

  /** One control step while paused (the transport's step button). */
  stepOnce() {
    this.control();
    settle(this, null);
  }

  observe() {
    return observe(this);
  }

  get network() {
    return activeNetwork(this);
  }

  decide(obs: Float64Array, name: "ginga" | "kicks" | "recover" | "esquiva" = this.network) {
    return decide(this, obs, name);
  }

  apply(rig: Rig) {
    poseRig(this, rig);
  }

  available(move: Move) {
    return available(this, move);
  }

  requestManeuver(type: Maneuver) {
    return requestManeuver(this, type);
  }

  request(move: Move) {
    return request(this, move);
  }
}
