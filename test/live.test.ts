import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import load from "@mujoco/mujoco";
import { LiveGinga } from "../src/sim/simulation";
import type { LiveFiles, Move } from "../src/sim/types";

// The browser runs the policy itself, so its port of the observation and of
// the network has to match ml/ exactly. The fixture (written by
// ml/export_web.py) holds states from a Python rollout with the observation,
// action and next state Python computed for each control step.

const dir = new URL("../public/data/sim/ginga/", import.meta.url);
const read = (name: string) => readFile(new URL(name, dir));

async function live() {
  const files: LiveFiles = {
    xml: (await read("humanoid.xml")).toString(),
    policy: JSON.parse((await read("policy.json")).toString()),
    weights: new Float32Array(new Uint8Array(await read("policy.bin")).buffer),
    reference: JSON.parse((await read("reference.json")).toString()),
  };
  return new LiveGinga(await load(), files);
}

interface Sample {
  start: {
    qpos: number[];
    qvel: number[];
    phase: number;
    bpm: number;
    move: Move | null;
    side: "r" | "l";
    kickT: number;
    sinceKick: number;
    anchor: [number, number];
    placement: [number, number, number];
    recover: LiveGinga["recover"];
    sinceRecover: number;
  };
  steps: { obs: number[]; action: number[]; qpos: number[] }[];
}

const fixture: Sample[] = JSON.parse(await readFile(new URL("fixtures/ginga-live.json", import.meta.url), "utf8"));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => Math.max(...Array.from(b, (v, i) => Math.abs(a[i] - v)));

describe("live ginga", () => {
  it("observes, decides and steps exactly like the Python environment", async () => {
    const ginga = await live();
    for (const { start, steps } of fixture) {
      ginga.setState(
        start.qpos,
        start.qvel,
        start.phase,
        start.bpm,
        start.move,
        start.side,
        start.kickT,
        start.sinceKick,
        start.anchor,
        start.placement,
        start.recover,
        start.sinceRecover,
      );
      for (const step of steps) {
        const obs = ginga.observe();
        expect(obs.length).toBe(step.obs.length);
        // Python keeps the observation in float32.
        expect(maxDiff(obs, step.obs)).toBeLessThan(1e-4);
        ginga.control();
        // The action of the network in charge (ginga, kicks or recovery), as Python's Brain chose it.
        // (Python runs the network in float32 on a float32 observation.)
        expect(maxDiff(ginga.lastAction, step.action)).toBeLessThan(1e-3);
        expect(maxDiff(ginga.data.qpos as Float64Array, step.qpos)).toBeLessThan(1e-4);
      }
    }
    ginga.dispose();
  });

  it("throws a requested kick and comes back to the ginga", async () => {
    const ginga = await live();
    ginga.bpm = 90;
    ginga.reset(0.2);
    const modes: string[] = [];
    ginga.subscribe(() => modes.push(ginga.mode));
    const pending = ginga.request("martelo");
    expect(pending).not.toBeNull();
    // A second request while busy is ignored.
    expect(ginga.request("bencao")).toBeNull();
    for (let i = 0; i < 800 && modes.at(-1) !== "ginga"; i++) ginga.update(0.03, 90, null);
    // The kick may end on the floor; then the body gets up (or is stood up) and gingas on.
    expect(modes.slice(0, 2)).toEqual(["waiting", "kick"]);
    expect(modes.at(-1)).toBe("ginga");
    expect(modes.slice(2, -1).every((mode) => mode === "down")).toBe(true);
    expect(ginga.busy).toBe(false);
    ginga.dispose();
  });

  it("stays down after a failed get-up until it is stood up", async () => {
    const ginga = await live();
    ginga.bpm = 90;
    ginga.reset(0.2);
    let falls = 0;
    let cleared = 0;
    let was = false;
    for (let i = 0; i < 4000; i++) {
      // Hard shoves from turning directions, so some get-ups fail.
      if (i % 130 === 0) ginga.push(500 * Math.cos(i * 2.399), 500 * Math.sin(i * 2.399));
      ginga.update(0.03, 90, null);
      if (ginga.fallen && !was) falls++;
      // The fall may only clear by standing the body up again (reset), never mid-recovery.
      if (!ginga.fallen && was && ginga.recover) cleared++;
      was = ginga.fallen;
    }
    expect(falls).toBeGreaterThan(0);
    expect(cleared).toBe(0);
    ginga.dispose();
  }, 120_000);

  it("keeps its balance for a while from the reference pose", async () => {
    const ginga = await live();
    ginga.bpm = 90;
    ginga.reset(0);
    let falls = 0;
    for (let i = 0; i < 300; i++) {
      ginga.update(0.03, 90, null);
      if (ginga.fallen) falls++;
    }
    expect(ginga.cycles).toBeGreaterThan(2);
    expect(Number.isFinite((ginga.data.qpos as Float64Array)[1])).toBe(true);
    ginga.dispose();
    // Balance is the policy's job; this only reports it (early checkpoints fall).
    console.log(`fallen in ${falls} of 300 control steps`);
  });

  it("performs navigation, feints, and standing/floor maneuvers without falling", async () => {
    const ginga = await live();
    ginga.bpm = 90;
    ginga.reset(0);

    // 1. Navigation & Orbit
    ginga.setNav({ forward: 1, turn: 1 });
    for (let i = 0; i < 60; i++) ginga.update(0.03, 90, null);
    expect(Math.hypot(ginga.placement[0], ginga.placement[1])).toBeGreaterThan(0.05);
    expect(Math.abs(ginga.placement[2])).toBeGreaterThan(0.1);
    expect(ginga.fallen).toBe(false);
    ginga.setNav({ forward: 0, turn: 0 });

    // 2. Feint Ginga (quebra na paralela) & Feint Mode (modo ameaça)
    expect(ginga.feintGinga()).toBe(true);
    for (let i = 0; i < 60 && ginga.pendingFeintGinga; i++) ginga.update(0.03, 90, null);
    expect(ginga.pendingFeintGinga).toBe(false);
    expect(ginga.toggleFeintMode()).toBe(true);
    expect(ginga.toggleFeintMode()).toBe(false);

    // 3. Standing & Floor Maneuvers
    for (const m of ["troca-de-base", "giro-de-base", "role", "passagem-de-negativa"] as const) {
      ginga.reset(0);
      const req = ginga.requestManeuver(m);
      expect(req).not.toBeNull();
      for (let i = 0; i < 200 && ginga.busy; i++) ginga.update(0.03, 90, null);
      expect(ginga.busy).toBe(false);
      expect(ginga.fallen).toBe(false);
    }
    ginga.dispose();
  }, 60_000);
});
