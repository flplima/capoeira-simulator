"""The trained networks that drive the body, and which one is in charge.

The ginga and the kicks are learned by two networks with the same inputs and
outputs ("experts", see train.py --skill): one practised only the ginga at
every tempo, the other practised kicks — from the moment the kick starts,
through it, and back into the ginga. The ginga network waits for the right
moment; the kick network takes over as the kick starts and hands back to
the ginga network a moment after the body is gingando again. A third network practised losing
balance: it takes over the moment the body is falling (GingaEnv.trouble),
lands it on its hands and knees, in the queda de quatro or in the negativa, gets it up and hands back
to the ginga the same way. The esquiva lateral has a network of its own, which takes over from the
ginga at the base and hands back like a kick network. A kick may also have a network of its own (kick_networks):
a network that practises every kick loses one as it gains another, and one that plays a single kick has nothing
to lose. The browser does the same (src/sim/simulation.ts).

The inputs grew as skills were added (new ones are appended at the end), so
each network reads only as many as it was trained on.
"""

import numpy as np
import torch

from env_consts import DODGES, KICK_MOVES
from train import Agent, RunningNorm

HANDBACK = 1.0  # s of ginga after getting up that the recovery network still drives
MOVE_HANDBACK = 2.5  # …and after a kick or an esquiva, that move's network (it practises 3 s: env.MOVE_EPISODE_TAIL)


class Network:
    def __init__(self, checkpoint: str):
        data = torch.load(checkpoint, weights_only=False)
        self.agent = Agent(data["obs"], data["act"])
        self.agent.load_state_dict(data["agent"])
        self.norm = RunningNorm(data["obs"])
        self.norm.mean, self.norm.var = data["mean"], data["var"]
        self.obs, self.act, self.stats = data["obs"], data["act"], data["stats"]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(self.norm(obs[None, : self.obs]).astype(np.float32))
            return self.agent.distribution(x).mean[0].numpy()


def kick_networks(spec: str) -> tuple[dict[str, str], dict[str, str]]:
    """What --kicks asks for: one checkpoint for every kick, then `move=checkpoint` for each kick played by another
    ("a.pt,martelo=b.pt,queixada=b.pt"). Returns the checkpoint of each network by name ("kicks", then "kicks-2",
    "kicks-3"… one per further checkpoint) and the network name of each kick."""
    shared, *own = spec.split(",")
    paths, names = {"kicks": shared}, {shared: "kicks"}
    moves = dict.fromkeys(KICK_MOVES, "kicks")
    for pair in own:
        move, path = pair.split("=")
        if move not in moves:
            raise ValueError(f"--kicks: {move} is not a kick ({', '.join(KICK_MOVES)})")
        moves[move] = names.setdefault(path, f"kicks-{len(names) + 1}")
        paths[moves[move]] = path
    return paths, moves


class Brain:
    def __init__(self, ginga: str, kicks: str | None = None, recover: str | None = None, esquiva: str | None = None):
        """kicks: a checkpoint, or several with the kicks each one plays (kick_networks)."""
        kick_paths, kick_moves = kick_networks(kicks) if kicks else ({}, {})
        paths = {"ginga": ginga, **kick_paths, "recover": recover, "esquiva": esquiva}
        # Every network by the name the app loads it under, and the network that plays each move.
        self.networks = {name: Network(path) for name, path in paths.items() if path}
        self.move_network = {**kick_moves, **({move: "esquiva" for move in DODGES} if esquiva else {})}
        self.ginga, self.recover = self.networks["ginga"], self.networks.get("recover")
        self.since_kick = np.inf
        self.since_recover = np.inf
        self.mover: Network | None = None  # the network that played the last kick or esquiva

    def reset(self) -> None:
        self.since_kick = np.inf
        self.since_recover = np.inf
        self.mover = None

    def network(self, env) -> Network:
        if self.recover and self.since_recover < HANDBACK:
            return self.recover
        return self.mover if self.mover and self.since_kick < MOVE_HANDBACK else self.ginga

    def __call__(self, env, obs: np.ndarray) -> np.ndarray:
        """The action for this control step; call once per step."""
        c = env.choreo
        kicking = c.move is not None  # the ginga network waits for the entry; kick practice starts at the kick
        if kicking:
            self.mover = self.networks.get(self.move_network.get(c.move))
        self.since_kick = 0.0 if kicking else self.since_kick + env.control_dt
        self.since_recover = 0.0 if c.recover is not None else self.since_recover + env.control_dt
        return self.network(env)(obs)
