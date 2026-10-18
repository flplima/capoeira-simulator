import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { CH, CHANNELS, mirrorPose } from "../src/data/channels";
import { loadLibrary } from "../src/data/load";
import { sampleClip } from "../src/anim/interp";
import { applyPose, buildRig, type BoneName } from "../src/rig/rig";
import { Vector3 } from "three";

const library = await loadLibrary((path) => readFile(new URL(`../public/data/${path}`, import.meta.url), "utf8"));

// Generous anatomical limits: wide enough for splits and high kicks, tight
// enough to catch a sign error or a typo (knee: -90, elbow: 400).
const LIMITS: Record<string, [number, number]> = {
  "knee_*.flex": [0, 165],
  "elbow_*.flex": [0, 165],
  "ankle_*.flex": [-50, 80],
  "hip_*.flex": [-70, 170],
  "hip_*.abd": [-50, 120],
  "hip_*.rot": [-70, 70],
  "shoulder_*.flex": [-90, 200],
  "shoulder_*.abd": [-45, 200],
  "shoulder_*.rot": [-100, 100],
  "torso.flex": [-60, 110],
  "torso.side": [-60, 60],
  "torso.twist": [-80, 80],
  "head.flex": [-70, 80],
  "head.side": [-45, 45],
  "head.twist": [-80, 80],
};
const limitFor = (key: string) => LIMITS[key.replace(/_[lr]\./, "_*.")];

describe("movement library", () => {
  it("loads every movement without errors", () => {
    expect(library.errors).toEqual([]);
    expect(library.movements).toHaveLength(17);
  });

  it("has unique ids and names", () => {
    expect(new Set(library.movements.map((m) => m.id)).size).toBe(17);
    expect(new Set(library.movements.map((m) => m.name)).size).toBe(17);
  });

  for (const m of library.movements) {
    it(`${m.id}: keeps every joint inside anatomical limits`, () => {
      const clip = library.clips[m.id];
      clip.frames.forEach((frame, k) => {
        for (const [i, c] of CHANNELS.entries()) {
          expect(Number.isFinite(frame[i]), `${c.key} @ keyframe ${k}`).toBe(true);
          const limit = limitFor(c.key);
          if (!limit) continue;
          expect(frame[i], `${c.key} @ keyframe ${k} (${clip.labels[k]})`).toBeGreaterThanOrEqual(limit[0]);
          expect(frame[i], `${c.key} @ keyframe ${k} (${clip.labels[k]})`).toBeLessThanOrEqual(limit[1]);
        }
      });
    });

    // A loop wraps back to its first keyframe, so only one-shot movements need this.
    if (!m.loop)
      it(`${m.id}: ends facing the way it started`, () => {
        const clip = library.clips[m.id];
        const yaw = clip.frames[clip.frames.length - 1][CH["root.yaw"]] - clip.frames[0][CH["root.yaw"]];
        const roll = clip.frames[clip.frames.length - 1][CH["root.roll"]] - clip.frames[0][CH["root.roll"]];
        expect(Math.abs(yaw % 360)).toBeLessThan(1);
        expect(Math.abs(roll % 360)).toBeLessThan(1);
      });

    if (!m.loop)
      it(`${m.id}: comes back to the start in z (travel only sideways)`, () => {
        const clip = library.clips[m.id];
        const z = clip.frames[clip.frames.length - 1][CH["root.z"]] - clip.frames[0][CH["root.z"]];
        expect(Math.abs(z)).toBeLessThan(0.15);
      });
  }

  it("samples a clip exactly at its keyframes", () => {
    const clip = library.clips.ginga;
    clip.times.forEach((t, k) => sampleClip(clip, t).forEach((v, c) => expect(v).toBeCloseTo(clip.frames[k][c], 9)));
  });

  it("mirrors the right base into the left one", () => {
    const right = library.clips.ginga.frames[1];
    const left = library.clips.ginga.frames[3];
    expect(Array.from(mirrorPose(right), (v) => v + 0)).toEqual(Array.from(left, (v) => v + 0));
  });
});

describe("solid body", () => {
  // Distance from a point to a segment, for capsule-vs-sphere checks.
  const toSegment = (p: Vector3, a: Vector3, b: Vector3) => {
    const ab = b.clone().sub(a);
    const t = Math.min(1, Math.max(0, p.clone().sub(a).dot(ab) / ab.lengthSq()));
    return a.clone().addScaledVector(ab, t).distanceTo(p);
  };

  it("never passes an arm through the head", () => {
    const rig = buildRig();
    const at = (bone: BoneName, y: number) => rig.bones[bone].localToWorld(new Vector3(0, y, 0));
    for (const m of library.movements) {
      const clip = library.clips[m.id];
      for (let t = 0; t <= clip.duration; t += 0.02) {
        applyPose(rig, sampleClip(clip, t));
        const head = rig.bones.head.localToWorld(new Vector3(0, 0.1, 0.005));
        for (const s of ["l", "r"] as const) {
          // Capsule radii from rig.ts plus the head's 0.105 m.
          const gaps = [
            toSegment(head, at(`upperarm_${s}`, -0.05), at(`upperarm_${s}`, -0.24)) - 0.046,
            toSegment(head, at(`forearm_${s}`, 0), at(`forearm_${s}`, -0.21)) - 0.038,
            toSegment(head, at(`hand_${s}`, -0.03), at(`hand_${s}`, -0.11)) - 0.032,
          ].map((d) => d - 0.105);
          expect(Math.min(...gaps), `${m.id} arm ${s} at ${t.toFixed(2)} s`).toBeGreaterThan(0);
        }
      }
    }
  });
});
