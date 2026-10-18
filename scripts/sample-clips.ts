// Samples movement clips at 30 fps for the ML pipeline (ml/kicks.py), right
// side and mirrored left side, so the simulated kicks start from the same
// keyframes the app shows (tuned against the reference video).
//
//     npx --yes tsx@4.19.2 scripts/sample-clips.ts martelo bencao > ml/data/clips.json

import { readFile } from "node:fs/promises";
import { CHANNELS, mirrorPose } from "../src/data/channels";
import { loadLibrary } from "../src/data/load";
import { sampleClip } from "../src/anim/interp";

const FPS = 30;
const ids = process.argv.slice(2);
const library = await loadLibrary((path) => readFile(new URL(`../public/data/${path}`, import.meta.url), "utf8"), ids);
if (library.errors.length) throw new Error(library.errors.join("\n"));

const out: Record<string, unknown> = { fps: FPS, channels: CHANNELS.map((c) => c.key) };
for (const id of ids) {
  const clip = library.clips[id];
  const count = Math.round(clip.duration * FPS) + 1;
  const right = Array.from({ length: count }, (_, i) => Array.from(sampleClip(clip, Math.min(i / FPS, clip.duration))));
  const left = right.map((frame) => Array.from(mirrorPose(Float64Array.from(frame))));
  out[id] = { duration: clip.duration, side: library.movements.find((m) => m.id === id)!.side, frames: { r: right, l: left } };
}
process.stdout.write(JSON.stringify(out));
