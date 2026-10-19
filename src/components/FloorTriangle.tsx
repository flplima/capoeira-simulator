import { useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import {
  BufferAttribute,
  BufferGeometry,
  CircleGeometry,
  Color,
  Group,
  LineDashedMaterial,
  LineLoop,
  Mesh,
  MeshBasicMaterial,
  RingGeometry,
} from "three";
import type { Markers } from "../physics/markers";
import type { Placement } from "../sim/types";

const LIFT = 0.002;
const REACH = 0.14; // a planted foot this close to a corner is on it
const LINE = new Color("#5f8fc4");
const ON = new Color("#8ec1f2");

/**
 * The ginga triangle on the floor: the two front corners are where the feet
 * stand in the paralela, the back corner is where either foot steps back to.
 * A corner fills in while a planted foot is on it. In the live simulation the
 * triangle follows the game: after a fall it is wherever the body got up.
 */
export function FloorTriangle({
  points,
  markers,
  placement,
}: {
  points: [number, number][];
  markers: Markers;
  placement?: () => Placement | null;
}) {
  const scene = useMemo(() => {
    const group = new Group();
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(new Float32Array(points.flatMap(([x, z]) => [x, LIFT, z])), 3));
    const outline = new LineLoop(
      geometry,
      new LineDashedMaterial({
        color: LINE,
        dashSize: 0.06,
        gapSize: 0.035,
        transparent: true,
        opacity: 0.9,
      }),
    );
    outline.computeLineDistances();
    group.add(outline);
    const corners = points.map(([x, z]) => {
      const ring = new Mesh(new RingGeometry(0.075, 0.09, 40), new MeshBasicMaterial({ color: LINE, transparent: true, opacity: 0.9 }));
      const fill = new Mesh(new CircleGeometry(0.075, 40), new MeshBasicMaterial({ color: ON, transparent: true, opacity: 0.28 }));
      for (const m of [ring, fill]) {
        m.rotation.x = -Math.PI / 2;
        m.position.set(x, LIFT, z);
        group.add(m);
      }
      return { x, z, ring, fill };
    });
    return { group, corners };
  }, [points]);

  useFrame(() => {
    const [px, pz, yaw] = placement?.() ?? [0, 0, 0];
    scene.group.position.set(px, 0, pz);
    scene.group.rotation.y = yaw;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const feet = markers.state.effectors.filter((e) => e.planted && e.name.startsWith("foot"));
    for (const c of scene.corners) {
      const x = cos * c.x + sin * c.z + px;
      const z = -sin * c.x + cos * c.z + pz;
      const on = feet.some((f) => Math.hypot(f.point.x - x, f.point.z - z) < REACH);
      c.fill.visible = on;
      (c.ring.material as MeshBasicMaterial).color.copy(on ? ON : LINE);
    }
  });

  return <primitive object={scene.group} />;
}
