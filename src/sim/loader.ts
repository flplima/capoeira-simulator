import type { MainModule } from "@mujoco/mujoco";
import { LiveGinga } from "./simulation";
import type { LiveFiles, Simulation } from "./types";

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
