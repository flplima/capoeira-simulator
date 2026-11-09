"""The PPO update of train_mjx.py in pure JAX: generalised advantages over the rollout, the batch normalised for the
networks, and the clipped policy and value losses over epochs of shuffled minibatches. With --mirror a symmetry
loss joins them: the policy must act on a mirrored observation with the mirrored action (env_mirror.py)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from env_mirror import Mirror
from train_nets import NormState, apply_mlp, apply_norm, gaussian_log_prob

SYMMETRY_WEIGHT = 4.0


class Batch(NamedTuple):
    obs: jax.Array
    act: jax.Array
    logp: jax.Array
    adv: jax.Array
    ret: jax.Array
    obs_mirror: jax.Array | None


def advantages(rewards: jax.Array, dones: jax.Array, values: jax.Array, boots: jax.Array, next_value: jax.Array, gamma: float, lam: float) -> jax.Array:
    """Generalised advantage estimates over a rollout (steps × envs); `boots` is the value of the state an episode
    was cut off at by time, and 0 otherwise."""

    def step(carry, inputs):
        last, following = carry
        rew, done, val, boot = inputs
        nonterminal = 1.0 - done
        gae = rew + gamma * (following * nonterminal + boot) - val + gamma * lam * nonterminal * last
        return (gae, val), gae

    _, adv = jax.lax.scan(step, (jnp.zeros_like(next_value), next_value), (rewards, dones, values, boots), reverse=True)
    return adv


def batch(norm: NormState, obs: jax.Array, act: jax.Array, logp: jax.Array, adv: jax.Array, ret: jax.Array, mirror: Mirror | None) -> Batch:
    """The flat training batch: normalised observations, actions, the old policy's log probabilities, normalised
    advantages and returns, and with `mirror` the normalised left-right mirror of every observation."""
    adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    obs_mirror = apply_norm(norm, obs[:, mirror.obs_perm] * mirror.obs_sign) if mirror is not None else None
    return Batch(apply_norm(norm, obs), act, logp, adv, ret, obs_mirror)


def asymmetry(actor: list, mean: jax.Array, obs_mirror: jax.Array, mirror: Mirror) -> jax.Array:
    """Mean squared gap between the policy's mean action at the mirrored observation and the mirror of its mean
    action `mean` at the original: zero for a left-right symmetric policy. A mirrored body state is worth as much
    as the original, so this pulls what one leg learned onto the other (Abdolhosseini et al. 2019, the LOSS method).
    The original's mean is held fixed, so the states the rollout visited keep what the reward taught them and only
    their mirrors move: pulling both ends met in the middle and undid the trained leg. Duplicating the batch with
    mirrored samples instead puts them hundreds of nats off a one-legged policy and overflows the PPO ratio."""
    target = jax.lax.stop_gradient(mean)[:, mirror.act_perm] * mirror.act_sign
    return jnp.mean((apply_mlp(actor, obs_mirror) - target) ** 2)


def update(tx, params: dict, opt_state, rng: jax.Array, data: Batch, mirror: Mirror | None, epochs: int, minibatches: int, actor_lr: jax.Array, critic_lr: jax.Array, clip_eps: float):
    """PPO's clipped policy loss plus half the value loss (plus the weighted symmetry loss with `mirror`), over
    `epochs` passes of `minibatches` shuffled minibatches, the actor and the critic stepped at their own learning
    rates."""
    size = data.obs.shape[0] // minibatches

    def mb_step(carry, idx):
        p, ost = carry
        mb = jax.tree.map(lambda x: x[idx], data)

        def loss_fn(prms):
            mean = apply_mlp(prms["actor"], mb.obs)
            logp = gaussian_log_prob(mean, prms["log_std"], mb.act)
            ratio = jnp.exp(logp - mb.logp)
            p_loss = -jnp.mean(jnp.minimum(ratio * mb.adv, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb.adv))
            val = jnp.squeeze(apply_mlp(prms["critic"], mb.obs), axis=-1)
            loss = p_loss + 0.5 * jnp.mean((val - mb.ret) ** 2)
            if mirror is not None:
                loss += SYMMETRY_WEIGHT * asymmetry(prms["actor"], mean, mb.obs_mirror, mirror)
            return loss

        grads = jax.grad(loss_fn)(p)
        updates, ost = tx.update(grads, ost, p)
        scaled = {
            "actor": jax.tree.map(lambda u: u * actor_lr, updates["actor"]),
            "critic": jax.tree.map(lambda u: u * critic_lr, updates["critic"]),
            "log_std": updates["log_std"] * actor_lr,
        }
        return (optax.apply_updates(p, scaled), ost), None

    def epoch(carry, _):
        p, ost, key = carry
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, data.obs.shape[0])[: minibatches * size].reshape((minibatches, size))
        (p, ost), _ = jax.lax.scan(mb_step, (p, ost), perm)
        return (p, ost, key), None

    (params, opt_state, rng), _ = jax.lax.scan(epoch, (params, opt_state, rng), None, length=epochs)
    return params, opt_state, rng
