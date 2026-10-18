import { describe, expect, it } from "vitest";
import { CH, mirrorPose, restPose, wrapAngle } from "../src/data/channels";
import type { Clip } from "../src/data/compile";
import { sampleClip } from "../src/anim/interp";
import { applyPose, buildRig } from "../src/rig/rig";
import { Vector3 } from "three";

const pose = (values: Record<string, number>) => {
  const p = restPose();
  for (const [k, v] of Object.entries(values)) p[CH[k]] = v;
  return p;
};

const clip = (frames: Record<string, number>[], times: number[], loop = false, duration?: number): Clip => ({
  id: "test",
  times,
  frames: frames.map(pose),
  labels: times.map((_, i) => `#${i}`),
  duration: duration ?? times[times.length - 1],
  loop,
  rest: 0.2,
});

describe("interpolation", () => {
  it("holds between two equal keyframes instead of overshooting", () => {
    const c = clip([{ "knee_l.flex": 0 }, { "knee_l.flex": 90 }, { "knee_l.flex": 90 }, { "knee_l.flex": 0 }], [0, 1, 2, 3]);
    for (let t = 1; t <= 2; t += 0.05) expect(sampleClip(c, t)[CH["knee_l.flex"]]).toBeCloseTo(90, 6);
  });

  it("never leaves the range of its keyframes", () => {
    const c = clip([{ "hip_r.flex": 0 }, { "hip_r.flex": 100 }, { "hip_r.flex": 20 }, { "hip_r.flex": 0 }], [0, 0.3, 0.5, 1.4]);
    for (let t = 0; t <= 1.4; t += 0.01) {
      const v = sampleClip(c, t)[CH["hip_r.flex"]];
      expect(v).toBeGreaterThanOrEqual(-1e-9);
      expect(v).toBeLessThanOrEqual(100 + 1e-9);
    }
  });

  it("wraps a loop back to its first frame", () => {
    const c = clip([{ "torso.flex": 10 }, { "torso.flex": 30 }], [0, 1], true, 2);
    expect(sampleClip(c, 2)[CH["torso.flex"]]).toBeCloseTo(10);
    expect(sampleClip(c, 1.5)[CH["torso.flex"]]).toBeCloseTo(20);
  });

  it("interpolates a full turn numerically", () => {
    const c = clip([{ "root.yaw": 0 }, { "root.yaw": -360 }], [0, 1]);
    expect(sampleClip(c, 0.5)[CH["root.yaw"]]).toBeCloseTo(-180);
  });
});

describe("mirroring", () => {
  it("swaps sides and flips lateral channels", () => {
    const m = mirrorPose(
      pose({
        "hip_r.flex": 80,
        "root.x": 0.5,
        "root.yaw": 30,
        "torso.side": 10,
      }),
    );
    expect(m[CH["hip_l.flex"]]).toBe(80);
    expect(m[CH["hip_r.flex"]]).toBe(0);
    expect(m[CH["root.x"]]).toBe(-0.5);
    expect(m[CH["root.yaw"]]).toBe(-30);
    expect(m[CH["torso.side"]]).toBe(-10);
  });

  it("is its own inverse", () => {
    const p = pose({ "shoulder_l.abd": 40, "root.roll": 90, "head.twist": 15 });
    expect(Array.from(mirrorPose(mirrorPose(p)))).toEqual(Array.from(p));
  });

  it("wraps angle differences to the short way round", () => {
    expect(wrapAngle(350)).toBe(-10);
    expect(wrapAngle(-360)).toBe(0);
    expect(wrapAngle(180)).toBe(180);
  });
});

describe("rig", () => {
  it("stands its lowest point on the floor", () => {
    const rig = buildRig();
    const poses: Record<string, number>[] = [
      {},
      { "knee_l.flex": 120, "hip_l.flex": 110 },
      { "root.roll": 180, "shoulder_l.abd": 170, "shoulder_r.abd": 170 },
    ];
    for (const values of poses) {
      applyPose(rig, pose(values));
      const lowest = Math.min(...rig.contacts.map((c) => c.point.getWorldPosition(new Vector3()).y - c.radius));
      expect(lowest).toBeCloseTo(0, 6);
    }
  });

  it("is about 1.75 m tall standing", () => {
    const rig = buildRig();
    applyPose(rig, restPose());
    const head = rig.bones.head.getWorldPosition(new Vector3()).y;
    expect(head).toBeGreaterThan(1.5);
    expect(head).toBeLessThan(1.65);
  });

  it("lifts off the floor by root.lift", () => {
    const rig = buildRig();
    applyPose(rig, restPose());
    const grounded = rig.bones.hips.position.y;
    applyPose(rig, pose({ "root.lift": 0.3 }));
    expect(rig.bones.hips.position.y - grounded).toBeCloseTo(0.3);
  });
});

describe("ginga poses", () => {
  it("stand on both feet", async () => {
    const { readFile } = await import("node:fs/promises");
    const { parse } = await import("yaml");
    const { PoseResolver } = await import("../src/data/compile");
    const library = parse(await readFile(new URL("../public/data/poses.yaml", import.meta.url), "utf8"));
    const resolver = new PoseResolver(library);
    const rig = buildRig();
    const v = new Vector3();
    for (const name of ["parallel", "base_r", "base_l", "ginga_r", "ginga_l"]) {
      applyPose(rig, resolver.resolve(name));
      for (const foot of ["foot_l", "foot_r"] as const) {
        const clearance = Math.min(...rig.effectors[foot].probes.map((p) => p.point.getWorldPosition(v).y - p.radius));
        expect(clearance, `${name} ${foot}`).toBeLessThan(0.015);
      }
    }
  });

  it("put the feet on the corners of the ginga triangle", async () => {
    const { readFile } = await import("node:fs/promises");
    const { loadLibrary } = await import("../src/data/load");
    const library = await loadLibrary((path) => readFile(new URL(`../public/data/${path}`, import.meta.url), "utf8"), ["ginga"]);
    const [frontLeft, frontRight, back] = library.movements[0].triangle!;
    const frames = library.clips.ginga.frames;
    const rig = buildRig();
    const at = (foot: "foot_l" | "foot_r") => {
      const [a, b] = rig.effectors[foot].probes.map((p) => p.point.getWorldPosition(new Vector3()));
      return [(a.x + b.x) / 2, (a.z + b.z) / 2];
    };
    const expectNear = ([x, z]: number[], [cx, cz]: number[], what: string) => expect(Math.hypot(x - cx, z - cz), what).toBeLessThan(0.05);
    const corners: [number[], number[]][] = [
      [frontLeft, frontRight],
      [frontLeft, back],
      [frontLeft, frontRight],
      [back, frontRight],
    ];
    frames.forEach((frame, i) => {
      applyPose(rig, frame);
      expectNear(at("foot_l"), corners[i][0], `keyframe ${i + 1} left`);
      expectNear(at("foot_r"), corners[i][1], `keyframe ${i + 1} right`);
    });
  });
});
