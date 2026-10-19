"""The esquiva's timing, the hips' fall and rise, the centre of mass's path and the specs of each dodge."""

import numpy as np
from scipy.optimize import brentq

MOVE = "esquiva-lateral"
# Keyframe times (s, public/data/movements/esquiva-lateral.yaml): the back foot leaves its corner, is
# down on the wide spot, the weight has settled at the bottom, the body starts back up, the foot leaves
# the wide spot, is back on its corner. (The hips reach the bottom when DROP lets them: see drop.)
LIFT, WIDE, SETTLE, RISE, HOME, BACK = 0.18, 0.45, 0.62, 0.85, 1.3, 1.7
# Going down the hips fall no faster than DROP[0] (m/s², under gravity's 9.81: the legs give way, they
# can't pull the body down) and are caught at no more than DROP[1] (on one deeply bent knee).
DROP = (7.0, 9.0)
# Coming back up the hips speed up at no more than LIFT_UP[0] and slow down at no more than LIFT_UP[1] (m/s²):
# rising off one deeply bent leg is the hard part, and a body that can't follow the rise sits down.
LIFT_UP = (4.0, 6.0)
# The wide spot: this far (m) outside the back foot's own front corner, and this far behind it.
WIDEN, BEHIND = 0.36, 0.1
# At the bottom the centre of mass is this share of the way from the front foot to the wide one.
SHARE = 0.25
# At the bottom the hips are this high (m), the bent thigh about level with the floor; the head this
# high, and this far outside the front foot, over the bent knee, out of the kick's way.
HIPS_LOW, HEAD_LOW, HEAD_OUT = 0.48, 0.75, 0.08
# Legs and hands stay this far (m) off the floor (only the soles touch it).
CLEAR = 0.02
# The back foot's step is this high (m) at its middle: low, almost a slide, but off the floor.
ARC = 0.045
# At the bottom the neck lifts this far (rad) from the ginga's, so the eyes stay on the partner as the
# trunk lies over the knee.
NECK_UP = -0.6
# The keyframes blend in from the ginga's base over this much keyframe time (s), and back out.
FROM_GINGA = 0.15
# How long a foot's floor target blends in or out (s) as it arrives or leaves.
RELEASE = 0.05
# The simulated body plays the esquiva this much slower than the keyframes.
SLOW = 1.0


def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def smoother(x: float) -> float:
    """0 to 1 with no jump in speed or acceleration at either end."""
    x = min(max(x, 0.0), 1.0)
    return x**3 * (x * (6 * x - 15) + 10)


def hermite(p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray, span: float, u: float) -> np.ndarray:
    """Cubic from (p0, v0) to (p1, v1) over `span` seconds, at fraction u."""
    u = min(max(u, 0.0), 1.0)
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * p0 + h10 * span * v0 + h01 * p1 + h11 * span * v1


def drop(times: np.ndarray, depth: float, limits: tuple[float, float] = DROP) -> np.ndarray:
    """How far along (0–1 of `depth` m) the hips are, going down as fast as DROP allows (or, with LIFT_UP,
    coming up): a smooth push of acceleration, then a smooth stop, with the speed zero at both ends."""
    fall, catch = limits
    # Half-sine accelerations: down for t1, up for t2; equal impulses (fall·t1 = catch·t2); the depth fixes t1.
    def depth_for(t1: float) -> float:
        t2 = fall * t1 / catch
        v = 2 * fall * t1 / np.pi
        return fall * t1**2 / np.pi + v * t2 - catch * t2**2 / np.pi
    t1 = brentq(lambda t: depth_for(t) - depth, 1e-4, 2.0) if depth > 1e-4 else 1e-4
    t2 = fall * t1 / catch
    dt = 1e-3
    grid = np.arange(0.0, t1 + t2 + dt, dt)
    acc = np.where(grid < t1, fall * np.sin(np.pi * grid / t1), -catch * np.sin(np.pi * np.clip(grid - t1, 0, t2) / t2))
    speed = np.cumsum(acc) * dt
    down = np.cumsum(speed) * dt
    return np.clip(np.interp(times, grid, down / max(down[-1], 1e-9)), 0.0, 1.0)


def com_path(
    times: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    foot: np.ndarray,
    bottom: np.ndarray,
    height: float,
    stage_times: tuple[float, float, float, float, float, float] = (LIFT, WIDE, SETTLE, RISE, HOME, BACK),
) -> np.ndarray:
    """The centre of mass (xz per sample). In the base it sits between the feet, well off the front foot. A
    short push from the back foot sets it moving toward the front one; with the back foot in the air it rolls
    on toward the front foot like a pendulum that just reaches it (the zero-moment point stays on that foot);
    with both feet down it goes a little toward the wide foot, holds, comes back, and leaves the same way in
    reverse, coming to rest where the base has it."""
    t_lift, t_wide, t_settle, t_rise, t_home, t_back = stage_times
    w = np.sqrt(9.81 / height)
    first = start + 0.15 * (foot - start)  # where the back foot leaves, moving toward the front foot at…
    first_v = -w * (first - foot)  # …just the speed to arrive over it
    last = end + 0.15 * (foot - end)
    last_v = w * (last - foot)
    landed = foot + (first - foot) * np.exp(-w * (t_wide - t_lift))
    leaving = foot + (last - foot) * np.exp(-w * (t_back - t_home))
    zero = np.zeros(2)
    path = np.zeros((len(times), 2))
    for k, t in enumerate(times):
        if t < t_lift:
            path[k] = hermite(start, zero, first, first_v, t_lift, t / t_lift)
        elif t < t_wide:
            path[k] = foot + (first - foot) * np.exp(-w * (t - t_lift))
        elif t < t_rise:
            path[k] = hermite(landed, -w * (landed - foot), bottom, zero, t_settle - t_wide, (t - t_wide) / (t_settle - t_wide))
        elif t < t_home:
            path[k] = hermite(bottom, zero, leaving, w * (leaving - foot), t_home - t_rise, (t - t_rise) / (t_home - t_rise))
        elif t < t_back:
            path[k] = foot + (last - foot) * np.exp(-w * (t_back - t))
        else:
            path[k] = hermite(last, last_v, end, zero, times[-1] - t_back, (t - t_back) / (times[-1] - t_back))
    return path


DODGE_SPECS = {
    "esquiva": {
        "clip": "esquiva-lateral",
        "times": (LIFT, WIDE, SETTLE, RISE, HOME, BACK),
        "wide": lambda front, back, side, out: front[side] + np.array([out * WIDEN, -BEHIND]),
        "flat_wide": 1.0,
        "share": SHARE,
        "hips_low": HIPS_LOW,
        "head_low": HEAD_LOW,
        "head_x": lambda foot, out: foot[0] - out * HEAD_OUT,
    },
    "esquiva-baixa": {
        "clip": "esquiva-baixa",
        "times": (0.18, 0.45, 0.62, 0.85, 1.30, 1.65),
        "wide": lambda front, back, side, out: back + np.array([0.0, -0.10]),
        "flat_wide": 0.0,
        "share": 0.20,
        "hips_low": 0.55,
        "head_low": 0.78,
        "head_x": lambda foot, out: foot[0],
    },
    "cocorinha": {
        "clip": "cocorinha",
        "times": (LIFT, WIDE, SETTLE, RISE, HOME, BACK),
        "wide": lambda front, back, side, out: front[side],
        "flat_wide": 1.0,
        "share": 0.45,
        "hips_low": 0.52,
        "head_low": 0.80,
        "head_x": lambda foot, out: 0.0,
    },
}
