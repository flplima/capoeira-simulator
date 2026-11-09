"""What both trainers write to log.jsonl each iteration: why the last episodes ended, and the tracking tolerances."""

import numpy as np

from env_consts import ENDS, LANDINGS, MOVES, SIGMA_INIT, TERMS


def ends_stats(ended: list[tuple[int, int, float, bool]]) -> dict:
    """Of the last episodes (why each ended, its landing or 0, whether it got up, whether it began from a recorded
    fall): the share per ending; of those that fell, the share per landing and how many got up (all of them, and
    those that began from a recorded fall — the ones like a real fall)."""
    stats: dict = {}
    if ended:
        stats.update({f"end_{name}": round(sum(e == k for e, *_ in ended) / len(ended), 3) for k, name in enumerate(ENDS)})
        fell = [(landing, up) for _, landing, up, _ in ended if landing]
        real = [up for _, landing, up, from_fall in ended if landing and from_fall]
        if fell:
            stats.update({f"fell_{name}": round(sum(l == k + 1 for l, _ in fell) / len(fell), 3) for k, name in enumerate(LANDINGS)})
            stats["got_up"] = round(float(np.mean([up for _, up in fell])), 3)
        if real:
            stats["got_up_falls"] = round(float(np.mean(real)), 3)
    return stats


def tolerance_stats(sigma: np.ndarray) -> dict:
    """The tolerances already set, as root-mean-square errors per term (rad, m): the ginga's, then each move's on
    each leg (env_consts.SIGMA_ROWS); a row nothing has visited yet is left out."""
    names = ["ginga"] + [f"{move}_{leg}" for move in MOVES for leg in "lr"]
    rows = {name: {term: round(float(np.sqrt(s)), 4) for term, s in zip(TERMS, row) if s < SIGMA_INIT} for name, row in zip(names, sigma)}
    return {name: row for name, row in rows.items() if row}
