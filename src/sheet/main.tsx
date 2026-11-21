// Contact sheet for authoring poses: every keyframe of one movement side by
// side, from several orthographic views. Dev-only — open
// /sheet.html?m=<id>[&n=<samples>][&views=front,side,top][&mirror=1][&travel=1]
// Without n it shows the keyframes; with n it shows n evenly spaced samples.

import { useLayoutEffect, useMemo } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useThree } from "@react-three/fiber";
import { Vector3 } from "three";
import { CH, mirrorPose, type Pose } from "../data/channels";
import { fetchText, loadLibrary } from "../data/load";
import { sampleClip } from "../anim/interp";
import { dict, t } from "../i18n/lang";
import { applyPose, buildRig } from "../rig/rig";

const SPACING = 1.9;
const VIEWS: Record<string, [number, number, number]> = {
  front: [0, 0.15, 1],
  side: [1, 0.15, 0],
  back: [0, 0.15, -1],
  top: [0.0001, 1, 0.25],
  "3/4": [1, 0.35, 1.2],
};

function Figure({ pose, at }: { pose: Pose; at: Vector3 }) {
  const rig = useMemo(() => buildRig(), []);
  useLayoutEffect(() => {
    rig.object.position.copy(at);
    rig.object.updateMatrixWorld(true);
    applyPose(rig, pose);
  }, [rig, pose, at]);
  return <primitive object={rig.object} />;
}

function Camera({ dir, width, center }: { dir: Vector3; width: number; center: Vector3 }) {
  const { camera, size } = useThree();
  useLayoutEffect(() => {
    camera.position.copy(center).addScaledVector(dir, 20);
    camera.up.set(0, 1, 0);
    camera.lookAt(center);
    camera.zoom = size.width / width;
    camera.updateProjectionMatrix();
  }, [camera, size, dir, width, center]);
  return null;
}

function Row({ name, poses }: { name: string; poses: Pose[] }) {
  const dir = new Vector3(...VIEWS[name]).normalize();
  // Lay the figures out across the screen: perpendicular to the view direction.
  const across = new Vector3(0, 1, 0).cross(dir).normalize();
  if (across.lengthSq() < 0.5) across.set(1, 0, 0);
  const width = poses.length * SPACING;
  const at = poses.map((_, i) => across.clone().multiplyScalar((i - (poses.length - 1) / 2) * SPACING));
  const center = new Vector3(0, name === "top" ? 0 : 0.85, 0);
  const height = name === "top" ? 2.2 : 2.3;
  return (
    <div className="row">
      <div className="view">{(dict().sheetViews as Record<string, string>)[name] ?? name}</div>
      <div style={{ height: `calc(${height / width} * min(100vw - 40px, 1800px))` }}>
        <Canvas orthographic flat dpr={1}>
          <color attach="background" args={["#e9e6df"]} />
          <ambientLight intensity={1.2} />
          <directionalLight position={[3, 5, 4]} intensity={2} />
          <gridHelper args={[width + 4, Math.round(width + 4) * 2, "#c9c4b8", "#d9d5cc"]} />
          {poses.map((pose, i) => (
            <Figure key={i} pose={pose} at={at[i]} />
          ))}
          <Camera dir={dir} width={width} center={center} />
        </Canvas>
      </div>
    </div>
  );
}

async function main() {
  const params = new URLSearchParams(location.search);
  const id = params.get("m") ?? "ginga";
  const library = await loadLibrary(fetchText, [id]);
  if (library.errors.length) throw new Error(library.errors.join("\n\n"));
  const clip = library.clips[id];
  if (!clip) throw new Error(`no movement "${id}"`);
  const n = Number(params.get("n")) || 0;
  const times = n ? Array.from({ length: n }, (_, i) => (clip.duration * i) / (clip.loop ? n : n - 1)) : clip.times;
  const labels = n ? times.map((t) => `${t.toFixed(2)}s`) : clip.labels.map((l, i) => `${clip.times[i].toFixed(2)}s · ${l}`);
  const poses = times.map((t) => {
    const pose = sampleClip(clip, t);
    if (params.get("mirror")) mirrorPose(pose, pose);
    if (!params.get("travel")) pose[CH["root.x"]] = pose[CH["root.z"]] = 0;
    return pose;
  });
  const views = (params.get("views") ?? "front,side,3/4").split(",");

  document.title = t("sheetTitle", { id });
  const style = document.createElement("style");
  style.textContent = `
    body { margin: 0; padding: 12px 20px; background: #e9e6df; font: 13px/1.3 ui-sans-serif, system-ui; color: #222; }
    h1 { font-size: 16px; margin: 0 0 6px; }
    .labels { display: grid; max-width: 1800px; }
    .labels div { text-align: center; padding: 0 4px; }
    .row { max-width: 1800px; position: relative; border-top: 1px solid #cfcabf; }
    .view { position: absolute; left: 4px; top: 4px; z-index: 1; color: #777; }
  `;
  document.head.append(style);
  createRoot(document.getElementById("root")!).render(
    <>
      <h1>{id}</h1>
      <div className="labels" style={{ gridTemplateColumns: `repeat(${poses.length}, 1fr)` }}>
        {labels.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
      {views.map((v) => (
        <Row key={v} name={v} poses={poses} />
      ))}
    </>,
  );
}

// The message can carry the ?m= the visitor typed, so it goes in as text, never as HTML.
main().catch((e) => {
  const pre = document.createElement("pre");
  pre.style.cssText = "color:#b00;white-space:pre-wrap";
  pre.textContent = String(e?.message ?? e);
  document.body.replaceChildren(pre);
});
