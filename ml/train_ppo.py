"""The PPO update of train_mjx.py in pure JAX: generalised advantages over the rollout, the batch normalised for the
networks (and, with --mirror, its left-right twin: env_mirror.py), and the clipped policy and value losses over
epochs of shuffled minibatches."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from env_mirror import Mirror
from train_nets import NormState, apply_mlp, apply_norm, gaussian_log_prob


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


def batch(params: dict, norm: NormState, obs: jax.Array, act: jax.Array, logp: jax.Array, adv: jax.Array, ret: jax.Array, mirror: Mirror | None) -> tuple[jax.Array, ...]:
    """The flat training batch: normalised observations, actions, the old policy's log probabilities, normalised
    advantages and returns. With `mirror`, the left-right twin of every sample follows: the mirrored observation
    and action, the old policy's log probability of that pair, and the same advantage and return, since a mirrored
    state is as likely and worth as much as the original (Abdolhosseini et al. 2019, duplicated data)."""
    adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
    parts = [(apply_norm(norm, obs), act, logp, adv, ret)]
    if mirror is not None:
        obs_m = apply_norm(norm, obs[:, mirror.obs_perm] * mirror.obs_sign)
        act_m = act[:, mirror.act_perm] * mirror.act_sign
        logp_m = gaussian_log_prob(apply_mlp(params["actor"], obs_m), params["log_std"], act_m)
        parts.append((obs_m, act_m, logp_m, adv, ret))
    return tuple(jnp.concatenate(x) for x in zip(*parts))


def update(tx, params: dict, opt_state, rng: jax.Array, data: tuple[jax.Array, ...], epochs: int, minibatches: int, actor_lr: jax.Array, critic_lr: jax.Array, clip_eps: float):
    """PPO's clipped policy loss plus half the value loss, over `epochs` passes of `minibatches` shuffled
    minibatches, the actor and the critic stepped at their own learning rates."""
    obs, act, logp_old, adv, ret = data
    size = obs.shape[0] // minibatches

    def mb_step(carry, idx):
        p, ost = carry

        def loss_fn(prms):
            logp = gaussian_log_prob(apply_mlp(prms["actor"], obs[idx]), prms["log_std"], act[idx])
            ratio = jnp.exp(logp - logp_old[idx])
            p_loss = -jnp.mean(jnp.minimum(ratio * adv[idx], jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[idx]))
            val = jnp.squeeze(apply_mlp(prms["critic"], obs[idx]), axis=-1)
            return p_loss + 0.5 * jnp.mean((val - ret[idx]) ** 2)

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
        perm = jax.random.permutation(k_perm, obs.shape[0])[: minibatches * size].reshape((minibatches, size))
        (p, ost), _ = jax.lax.scan(mb_step, (p, ost), perm)
        return (p, ost, key), None

    (params, opt_state, rng), _ = jax.lax.scan(epoch, (params, opt_state, rng), None, length=epochs)
    return params, opt_state, rng
