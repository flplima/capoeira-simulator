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
ginga at the base and hands back like a kick network. The browser does the same (src/sim/simulation.ts).

The inputs grew as skills were added (new ones are appended at the end), so
each network reads only as many as it was trained on.
"""

import numpy as np
import torch

from env import DODGES
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


class Brain:
    def __init__(self, ginga: str, kicks: str | None = None, recover: str | None = None, esquiva: str | None = None):
        self.ginga = Network(ginga)
        self.kicks = Network(kicks) if kicks else None
        self.recover = Network(recover) if recover else None
        self.esquiva = Network(esquiva) if esquiva else None
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
            self.mover = self.esquiva if c.move in DODGES else self.kicks
        self.since_kick = 0.0 if kicking else self.since_kick + env.control_dt
        self.since_recover = 0.0 if c.recover is not None else self.since_recover + env.control_dt
        return self.network(env)(obs)
