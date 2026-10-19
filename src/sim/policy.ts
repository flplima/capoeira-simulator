import type { LiveGinga } from "./simulation";

/** The network in charge (Brain.network): the recovery network while down and a moment after, the kick (or
 * esquiva) network from the start of a kick (or esquiva) until a moment after it, else the ginga network. */
export function activeNetwork(g: LiveGinga): "ginga" | "kicks" | "recover" | "esquiva" {
  const { networks, handback, handbackKicks, handbackEsquiva } = g.files.policy;
  if (networks.recover && g.sinceRecover < handback) return "recover";
  const moverHandback = (g.mover === "esquiva" ? handbackEsquiva : handbackKicks) ?? handback;
  return g.mover && networks[g.mover] && g.sinceKick < moverHandback ? g.mover : "ginga";
}

/** A network's mean action for an observation (actor MLP, ELU between layers). Each network reads as
 * many inputs as it was trained on: later skills appended theirs at the end. */
export function decide(g: LiveGinga, obs: Float64Array, name: "ginga" | "kicks" | "recover" | "esquiva" = g.network): Float64Array {
  const { policy, weights } = g.files;
  const net = policy.networks[name]!;
  let x = g.input;
  const inputs = net.layers[0].in;
  for (let i = 0; i < inputs; i++) x[i] = Math.min(Math.max((obs[i] - net.mean[i]) / net.std[i], -policy.clip), policy.clip);
  net.layers.forEach((layer, l) => {
    const y = g.layers[l];
    const bias = layer.offset + layer.in * layer.out;
    const last = l === net.layers.length - 1;
    for (let o = 0; o < layer.out; o++) {
      let sum = weights[bias + o];
      const row = layer.offset + o * layer.in;
      for (let i = 0; i < layer.in; i++) sum += weights[row + i] * x[i];
      y[o] = !last && sum < 0 ? Math.expm1(sum) : sum;
    }
    x = y;
  });
  return x;
}
