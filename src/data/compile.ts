import { CH, JOINTS, JOINT_NAMES, mirrorPose, restPose, type Pose } from "./channels";
import type { Movement, PoseBody, PoseLibrary } from "./schema";

/** A movement compiled to dense keyframes, ready to sample. */
export interface Clip {
  id: string;
  times: number[];
  frames: Pose[];
  labels: string[];
  /** Clip length in seconds; for a loop, the time at which it wraps to frame 0. */
  duration: number;
  loop: boolean;
  /** Seconds to hold the last frame before a non-looping clip repeats. */
  rest: number;
  /** Flight phases [takeoff, landing] whose height follows gravity (filled by bakeDynamics). */
  airborne?: [number, number][];
  /** Spins re-timed by the body's moment of inertia (filled by bakeDynamics). */
  spins?: { t0: number; t1: number; yaw: Float64Array }[];
}

// Placement carries over from the previous keyframe even when a named pose is
// applied: a named pose says how the body is shaped, not where it stands.
const PLACEMENT = ["root.x", "root.z", "root.yaw"].map((k) => CH[k]);

function applyOverrides(target: Pose, body: PoseBody) {
  for (const joint of JOINT_NAMES) {
    const value = body[joint];
    if (value === undefined) continue;
    if (typeof value === "number") {
      target[CH[`${joint}.${JOINTS[joint][0]}`]] = value;
      continue;
    }
    for (const [field, v] of Object.entries(value)) if (v !== undefined) target[CH[`${joint}.${field}`]] = v;
  }
}

export class PoseResolver {
  private cache = new Map<string, Pose>();
  constructor(private library: PoseLibrary) {}

  resolve(name: string, stack: string[] = []): Pose {
    const cached = this.cache.get(name);
    if (cached) return cached;
    const body = this.library.poses[name];
    if (!body) throw new Error(`unknown pose "${name}"`);
    if (stack.includes(name)) throw new Error(`pose cycle: ${[...stack, name].join(" → ")}`);
    const parent = body.pose ?? body.mirror;
    const pose = parent ? Float64Array.from(this.resolve(parent, [...stack, name])) : restPose();
    if (body.mirror) mirrorPose(pose, pose);
    applyOverrides(pose, body);
    this.cache.set(name, pose);
    return pose;
  }
}

export function compileMovement(movement: Movement, resolver: PoseResolver): Clip {
  let previous = restPose();
  const frames = movement.keyframes.map((kf) => {
    const frame = kf.pose ? Float64Array.from(resolver.resolve(kf.pose)) : Float64Array.from(previous);
    if (kf.pose) for (const i of PLACEMENT) frame[i] = previous[i];
    applyOverrides(frame, kf);
    previous = frame;
    return frame;
  });
  const times = movement.keyframes.map((k) => k.t);
  return {
    id: movement.id,
    times,
    frames,
    labels: movement.keyframes.map((k, i) => k.label ?? k.pose ?? `#${i + 1}`),
    duration: movement.loop ? movement.duration! : times[times.length - 1],
    loop: movement.loop,
    rest: movement.rest,
  };
}
