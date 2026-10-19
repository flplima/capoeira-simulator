import { useEffect, useMemo, useRef, useSyncExternalStore } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Grid, OrbitControls } from "@react-three/drei";
import { Vector3, type PerspectiveCamera } from "three";
import type { OrbitControls as OrbitControlsImpl } from "three/examples/jsm/controls/OrbitControls.js";
import { CapoeiraContext } from "../context";
import type { Playback } from "../playback";
import { loadLive } from "../sim/loader";
import { berimbau } from "../music/berimbau";
import { buildRig, type Rig } from "../rig/rig";
import { Markers } from "../physics/markers";
import { FloorMarkers } from "./FloorMarkers";
import { FloorTriangle } from "./FloorTriangle";

const RODA_RADIUS = 2.4;
/** Height the camera orbits around; below the hips so the figure sits above the control dock. */
const TARGET_Y = 0.5;
/** The trained ginga, kicks, esquiva and recovery (ml/export_web.py). */
const SIMULATION = `${import.meta.env.BASE_URL}data/sim/ginga/`;

/** Loads the live simulation and drives the rig from it every frame, in step with the machine. */
function Capoeirista({ playback, rig, markers }: { playback: Playback; rig: Rig; markers: Markers }) {
  const actor = CapoeiraContext.useActorRef();
  const speed = CapoeiraContext.useSelector((s) => s.context.speed);
  const playing = CapoeiraContext.useSelector((s) => s.matches("playing"));
  const bpm = CapoeiraContext.useSelector((s) => s.context.bpm);
  const bpmRef = useRef(bpm);
  bpmRef.current = bpm;
  const settings = useRef({ speed, playing });
  settings.current = { speed, playing };
  useEffect(() => {
    // The engine (MuJoCo as WebAssembly) and the networks load once, when the page opens.
    let cancelled = false;
    loadLive({ dir: SIMULATION })
      .then((live) => {
        if (cancelled) return live.dispose();
        Object.assign(live, { ...settings.current, bpm: bpmRef.current });
        live.reset(0);
        playback.sim = live;
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("física em tempo real indisponível", error);
        playback.error = String((error as Error)?.message ?? error);
      });
    return () => {
      cancelled = true;
      playback.sim?.dispose();
      playback.sim = null;
    };
  }, [playback]);
  useEffect(() => {
    if (playback.sim) Object.assign(playback.sim, { speed, playing });
  }, [playback, speed, playing]);
  useEffect(() => {
    // A live simulation can't go back in time: stepping runs one control step.
    const sub = actor.on("step", () => playback.sim?.stepOnce());
    return () => sub.unsubscribe();
  }, [actor, playback]);

  useFrame((_, delta) => {
    if (!playback.sim) return;
    // The policy and the physics run live; the music's bar keeps the ginga on the beat.
    playback.sim.update(Math.min(delta, 0.1), bpmRef.current, berimbau.barPhase());
    playback.sim.apply(rig);
    markers.observe();
  });
  return <primitive object={rig.object} />;
}

/** Keeps the orbit target on the capoeirista as travelling moves carry them around the roda. */
function Follow({ rig }: { rig: Rig }) {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  const hips = useMemo(() => new Vector3(), []);
  const delta = useMemo(() => new Vector3(), []);
  useFrame(({ camera }, dt) => {
    if (!controls) return;
    rig.bones.hips.getWorldPosition(hips);
    hips.y = TARGET_Y;
    delta.copy(hips).sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 3));
    controls.target.add(delta);
    camera.position.add(delta);
    controls.update();
  });
  return null;
}

/** Widen the lens on portrait screens so the whole figure, kicks included, stays in frame. */
function Lens() {
  const camera = useThree((s) => s.camera) as PerspectiveCamera;
  const aspect = useThree((s) => s.size.width / s.size.height);
  useEffect(() => {
    camera.fov = aspect < 0.8 ? 62 : aspect < 1.2 ? 46 : 36;
    camera.updateProjectionMatrix();
  }, [camera, aspect]);
  return null;
}

function Roda() {
  return (
    <group rotation-x={-Math.PI / 2}>
      <mesh position-z={-0.004}>
        <circleGeometry args={[14, 64]} />
        <meshStandardMaterial color="#1a1b1f" roughness={1} />
      </mesh>
      <mesh>
        <ringGeometry args={[RODA_RADIUS - 0.012, RODA_RADIUS + 0.012, 128]} />
        <meshBasicMaterial color="#3a3a40" />
      </mesh>
      <mesh>
        <ringGeometry args={[0.04, 0.055, 32]} />
        <meshBasicMaterial color="#34343a" />
      </mesh>
    </group>
  );
}

export function Stage({ playback }: { playback: Playback }) {
  const rig = useMemo(() => buildRig(), []);
  const markers = useMemo(() => new Markers(rig), [rig]);
  const sim = useSyncExternalStore(playback.subscribe, () => playback.sim);
  return (
    <Canvas camera={{ position: [2.9, 1.5, 4.6], fov: 36 }} dpr={[1, 2]}>
      <color attach="background" args={["#141518"]} />
      <fog attach="fog" args={["#141518", 7, 16]} />
      <hemisphereLight args={["#f3efe6", "#26262b", 1.1]} />
      <directionalLight position={[3, 6, 4]} intensity={2.2} />
      <directionalLight position={[-4, 3, -3]} intensity={0.6} color="#b9c4ff" />
      <Roda />
      {/* 25 cm cells, 1 m sections. */}
      <Grid
        position={[0, 0.001, 0]}
        infiniteGrid
        cellSize={0.25}
        cellThickness={0.6}
        cellColor="#2b2c32"
        sectionSize={1}
        sectionThickness={1.1}
        sectionColor="#3d3e46"
        fadeDistance={16}
        fadeStrength={1.4}
      />
      <ContactShadows position={[0, 0.001, 0]} scale={10} blur={2.2} opacity={0.65} far={2.5} resolution={512} />
      <Capoeirista playback={playback} rig={rig} markers={markers} />
      {sim && <FloorTriangle points={sim.triangle} markers={markers} placement={() => sim.placement} />}
      {sim && <FloorMarkers rig={rig} markers={markers} />}
      <OrbitControls
        makeDefault
        target={[0, TARGET_Y, 0]}
        minDistance={1.8}
        maxDistance={9}
        maxPolarAngle={Math.PI / 2 - 0.05}
        enablePan={false}
      />
      <Follow rig={rig} />
      <Lens />
    </Canvas>
  );
}
