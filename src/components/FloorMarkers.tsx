import { useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import {
  BufferAttribute,
  BufferGeometry,
  CircleGeometry,
  Color,
  Group,
  Line,
  LineBasicMaterial,
  LineDashedMaterial,
  LineLoop,
  Mesh,
  MeshBasicMaterial,
  RingGeometry,
} from "three";
import type { Markers } from "../physics/markers";
import type { Rig } from "../rig/rig";

const LIFT = 0.003;
const COLORS = {
  air: new Color("#8d8a83"),
  planted: new Color("#e0a82e"),
  com: new Color("#f1eee6"),
  balanced: new Color("#6fbf8a"),
  unbalanced: new Color("#e0704f"),
};

function dashed(color: Color, dash = 0.03, gap = 0.025) {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(6), 3));
  const line = new Line(geometry, new LineDashedMaterial({ color, dashSize: dash, gapSize: gap, transparent: true, opacity: 0.85 }));
  line.frustumCulled = false;
  return line;
}

function drop(line: Line, x: number, y: number, z: number) {
  const position = line.geometry.attributes.position as BufferAttribute;
  position.setXYZ(0, x, y, z);
  position.setXYZ(1, x, LIFT, z);
  position.needsUpdate = true;
  line.computeLineDistances();
  line.visible = y > LIFT + 0.01;
}

/**
 * Plumb lines from each hand and foot to the floor, ending in a small circle
 * (filled when the limb is planted), plus the centre of mass projected on the
 * floor and the support polygon it has to stay inside — green while balanced.
 */
export function FloorMarkers({ rig, markers }: { rig: Rig; markers: Markers }) {
  const scene = useMemo(() => {
    const group = new Group();
    const limbs = markers.state.effectors.map(() => {
      const line = dashed(COLORS.air);
      const ring = new Mesh(new RingGeometry(0.03, 0.04, 28), new MeshBasicMaterial({ color: COLORS.air, transparent: true, opacity: 0.9 }));
      const dot = new Mesh(new CircleGeometry(0.03, 28), new MeshBasicMaterial({ color: COLORS.planted }));
      for (const m of [ring, dot]) m.rotation.x = -Math.PI / 2;
      group.add(line, ring, dot);
      return { line, ring, dot };
    });

    const comLine = dashed(COLORS.com, 0.04, 0.03);
    const comMark = new Mesh(new RingGeometry(0.035, 0.05, 4), new MeshBasicMaterial({ color: COLORS.com }));
    comMark.rotation.x = -Math.PI / 2;
    const supportGeometry = new BufferGeometry();
    supportGeometry.setAttribute("position", new BufferAttribute(new Float32Array(3 * 64), 3));
    const support = new LineLoop(supportGeometry, new LineBasicMaterial({ color: COLORS.balanced, transparent: true, opacity: 0.8 }));
    support.frustumCulled = false;
    group.add(comLine, comMark, support);
    return { group, limbs, comLine, comMark, support };
  }, [markers]);

  useFrame(() => {
    // Markers live in the rig's local space, which is the world here.
    scene.group.position.copy(rig.object.position);
    const { effectors, com, support: hull, balanced } = markers.state;
    effectors.forEach((e, i) => {
      const { line, ring, dot } = scene.limbs[i];
      drop(line, e.point.x, e.point.y, e.point.z);
      ring.position.set(e.point.x, LIFT, e.point.z);
      dot.position.set(e.point.x, LIFT + 0.0005, e.point.z);
      dot.visible = e.planted;
      (ring.material as MeshBasicMaterial).color.copy(e.planted ? COLORS.planted : COLORS.air);
      (line.material as LineDashedMaterial).color.copy(e.planted ? COLORS.planted : COLORS.air);
    });

    drop(scene.comLine, com.x, com.y, com.z);
    scene.comMark.position.set(com.x, LIFT, com.z);
    const color = balanced ? COLORS.balanced : COLORS.unbalanced;
    (scene.comMark.material as MeshBasicMaterial).color.copy(color);

    const position = scene.support.geometry.attributes.position as BufferAttribute;
    const n = Math.min(hull.length, 64);
    hull.slice(0, n).forEach(([x, z], i) => position.setXYZ(i, x, LIFT, z));
    position.needsUpdate = true;
    scene.support.geometry.setDrawRange(0, n);
    scene.support.visible = n >= 2;
    (scene.support.material as LineBasicMaterial).color.copy(color);
  });

  return <primitive object={scene.group} />;
}
