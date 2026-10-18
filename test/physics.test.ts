import { describe, expect, it } from "vitest";
import { CH, restPose } from "../src/data/channels";
import type { Clip } from "../src/data/compile";
import { G, sampleClip } from "../src/anim/interp";
import { applyPose, buildRig } from "../src/rig/rig";
import { bakeDynamics } from "../src/physics/bake";
import { closestInHull, convexHull } from "../src/physics/hull";
import { SEGMENTS, centerOfMass, verticalInertia } from "../src/physics/mass";

const pose = (values: Record<string, number>) => {
  const p = restPose();
  for (const [k, v] of Object.entries(values)) p[CH[k]] = v;
  return p;
};

const clip = (frames: Record<string, number>[], times: number[]): Clip => ({
  id: "test",
  times,
  frames: frames.map(pose),
  labels: times.map((_, i) => `#${i}`),
  duration: times[times.length - 1],
  loop: false,
  rest: 0.2,
});

describe("mass model", () => {
  it("segment fractions add up to the whole body", () => {
    expect(SEGMENTS.reduce((sum, s) => sum + s[2], 0)).toBeCloseTo(1, 6);
  });

  it("puts the standing centre of mass just above the hips, over the feet", () => {
    const rig = buildRig();
    applyPose(rig, restPose());
    const com = centerOfMass(rig);
    // About 55–57 % of standing height for an adult.
    expect(com.y).toBeGreaterThan(0.93);
    expect(com.y).toBeLessThan(1.02);
    expect(Math.abs(com.x)).toBeLessThan(0.01);
  });

  it("has a larger moment of inertia with a leg out than tucked", () => {
    const rig = buildRig();
    applyPose(rig, pose({ "hip_r.flex": 90 }));
    const out = verticalInertia(rig);
    applyPose(rig, pose({ "hip_r.flex": 120, "knee_r.flex": 140 }));
    expect(out).toBeGreaterThan(verticalInertia(rig) * 1.3);
  });
});

describe("support polygon", () => {
  const square: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]];

  it("drops interior points from the hull", () => {
    expect(convexHull(square)).toHaveLength(4);
  });

  it("leaves a point inside where it is and clamps one outside to the edge", () => {
    const hull = convexHull(square);
    expect(closestInHull([0.3, 0.6], hull)).toEqual([0.3, 0.6]);
    const [x, z] = closestInHull([2, 0.5], hull);
    expect(x).toBeCloseTo(1);
    expect(z).toBeCloseTo(0.5);
  });
});

describe("baked dynamics", () => {
  it("flies a jump on a ballistic arc set by its flight time", () => {
    const jump = bakeDynamics(clip([{}, { "root.lift": 0.3 }, {}], [0, 0.25, 0.5]));
    expect(jump.airborne).toEqual([[0, 0.5]]);
    // Peak of a 0.5 s flight: g·T²/8.
    expect(sampleClip(jump, 0.25)[CH["root.lift"]]).toBeCloseTo((G * 0.5 * 0.5) / 8, 6);
    expect(sampleClip(jump, 0.5)[CH["root.lift"]]).toBeCloseTo(0, 6);
  });

  it("re-times spins without changing the turn at the keyframes", () => {
    const spin = bakeDynamics(clip([{ "root.yaw": 0 }, { "root.yaw": -180, "hip_r.flex": 90 }, { "root.yaw": -360 }], [0, 0.6, 1.2]));
    expect(spin.spins?.length).toBeGreaterThan(0);
    spin.times.forEach((t, k) => expect(sampleClip(spin, t)[CH["root.yaw"]]).toBeCloseTo(spin.frames[k][CH["root.yaw"]], 6));
  });

  it("spins faster while tucked than with a leg out", () => {
    const frames: Record<string, number>[] = [
      { "root.yaw": 0, "hip_r.flex": 120, "knee_r.flex": 140 },
      { "root.yaw": -180, "hip_r.flex": 90 },
      { "root.yaw": -360, "hip_r.flex": 90 },
    ];
    const authored = clip(frames, [0, 1, 2]);
    const baked = bakeDynamics(clip(frames, [0, 1, 2]));
    const rate = (c: Clip, t: number) => Math.abs(sampleClip(c, t + 0.01)[CH["root.yaw"]] - sampleClip(c, t - 0.01)[CH["root.yaw"]]);
    // In the first turn the leg unfolds from tucked to straight out: compared
    // with the keyframed curve, the start turns faster and the end slower.
    expect(rate(baked, 0.2) / rate(authored, 0.2)).toBeGreaterThan(rate(baked, 0.8) / rate(authored, 0.8));
    expect(baked.spins).toHaveLength(2);
  });
});

