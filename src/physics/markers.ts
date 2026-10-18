import { Vector3 } from "three";
import { EFFECTOR_NAMES, type EffectorName, type Rig } from "../rig/rig";
import { SEGMENTS, centerOfMass, segmentPositions } from "./mass";
import { closestInHull, convexHull, type P2 } from "./hull";

// What the floor markers draw, measured from the posed rig every frame after
// the simulation has moved it: which hands and feet touch the floor, the
// centre of mass, and the support polygon it has to stay over. Everything is
// in the rig object's local space, whose floor is y = 0.

/** A probe this close to the floor (m) touches it. */
const TOUCH = 0.015;
/** The floor contact of a hand or foot is a disc of this radius around each probe. */
const CONTACT_RADIUS = 0.045;

export interface EffectorState {
  name: EffectorName;
  /** Lowest point of the hand or foot (rig-local). */
  point: Vector3;
  /** Touching the floor. */
  planted: boolean;
}

export interface BodyState {
  effectors: EffectorState[];
  com: Vector3;
  /** Support polygon on the floor, [x, z] in hull order. */
  support: P2[];
  /** Centre of mass over the support polygon (or too little touching the floor to tell). */
  balanced: boolean;
}

const tmp = new Vector3();
const DISC = Array.from({ length: 8 }, (_, i): P2 => [Math.cos((i * Math.PI) / 4) * CONTACT_RADIUS, Math.sin((i * Math.PI) / 4) * CONTACT_RADIUS]);

export class Markers {
  readonly state: BodyState = {
    effectors: EFFECTOR_NAMES.map((name) => ({ name, point: new Vector3(), planted: false })),
    com: new Vector3(),
    support: [],
    balanced: true,
  };
  private points = SEGMENTS.map(() => new Vector3());

  constructor(private rig: Rig) {}

  /** Measure the pose as it is now. */
  observe() {
    const { rig } = this;
    rig.object.updateMatrixWorld(true);
    const support: P2[] = [];
    for (const e of this.state.effectors) {
      const effector = rig.effectors[e.name];
      e.point.set(0, 0, 0);
      let lowest = Infinity;
      for (const probe of effector.probes) {
        this.local(probe.point, tmp);
        e.point.add(tmp);
        lowest = Math.min(lowest, tmp.y - probe.radius);
        if (tmp.y - probe.radius < TOUCH) for (const [dx, dz] of DISC) support.push([tmp.x + dx, tmp.z + dz]);
      }
      e.point.divideScalar(effector.probes.length).setY(Math.max(lowest, 0));
      e.planted = lowest < TOUCH;
    }
    const com = rig.object.worldToLocal(centerOfMass(rig, this.state.com, segmentPositions(rig, this.points)));
    this.state.support = convexHull(support);
    const [cx, cz] = closestInHull([com.x, com.z], this.state.support);
    this.state.balanced = this.state.support.length < 3 || Math.hypot(cx - com.x, cz - com.z) < 0.03;
  }

  /** World position of an object, in the rig's local space. */
  private local(object: { getWorldPosition(v: Vector3): Vector3 }, out = new Vector3()): Vector3 {
    return this.rig.object.worldToLocal(object.getWorldPosition(out));
  }
}
