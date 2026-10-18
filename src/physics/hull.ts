// 2D helpers on the floor plane (x, z) for the support polygon.

export type P2 = [number, number];

const cross = (o: P2, a: P2, b: P2) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

/** Convex hull, counter-clockwise (monotone chain). */
export function convexHull(points: P2[]): P2[] {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const lower: P2[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: P2[] = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function closestOnSegment(p: P2, a: P2, b: P2): P2 {
  const dx = b[0] - a[0];
  const dz = b[1] - a[1];
  const len = dx * dx + dz * dz;
  const t = len === 0 ? 0 : Math.min(Math.max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / len, 0), 1);
  return [a[0] + dx * t, a[1] + dz * t];
}

/** The point of a convex polygon (hull order) nearest to p — p itself when inside. */
export function closestInHull(p: P2, hull: P2[]): P2 {
  if (hull.length === 0) return p;
  if (hull.length >= 3 && hull.every((a, i) => cross(a, hull[(i + 1) % hull.length], p) >= 0)) return p;
  let best: P2 = hull[0];
  let bestD = Infinity;
  for (let i = 0; i < hull.length; i++) {
    const q = closestOnSegment(p, hull[i], hull[(i + 1) % hull.length]);
    const d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2;
    if (d < bestD) {
      bestD = d;
      best = q;
    }
  }
  return best;
}
