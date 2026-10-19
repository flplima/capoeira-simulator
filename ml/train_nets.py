"""The pure-JAX actor-critic for train_mjx.py: observation normaliser, MLPs, and the PyTorch `.pt` checkpoint round trip."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch


# ---- Running observation normaliser (matches train.py RunningNorm) ----------


class NormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def init_norm(size: int) -> NormState:
    return NormState(
        mean=jnp.zeros(size, jnp.float32),
        var=jnp.ones(size, jnp.float32),
        count=jnp.full(size, 1e-4, jnp.float32),
    )


def update_norm(norm: NormState, x: jax.Array) -> NormState:
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    n = jnp.array(x.shape[0], jnp.float32)
    delta = batch_mean - norm.mean
    total = norm.count + n
    w_old = norm.count / total
    w_new = n / total
    mean = norm.mean + delta * w_new
    var = norm.var * w_old + batch_var * w_new + (delta**2) * w_old * w_new
    return NormState(mean=mean, var=var, count=total)


def apply_norm(norm: NormState, x: jax.Array) -> jax.Array:
    return jnp.clip((x - norm.mean) / jnp.sqrt(norm.var + 1e-8), -10.0, 10.0)


# ---- Pure-JAX Actor-Critic MLP (512 -> ELU -> 256 -> ELU -> out) ------------


def init_mlp(rng: jax.Array, in_size: int, out_size: int, hidden=(512, 256), out_scale: float = 1.0) -> list[dict[str, jax.Array]]:
    sizes = [in_size, *hidden, out_size]
    keys = jax.random.split(rng, len(sizes) - 1)
    layers = []
    for i, (k, din, dout) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        bound = 1.0 / np.sqrt(din)
        kw, kb = jax.random.split(k)
        w = jax.random.uniform(kw, (din, dout), jnp.float32, minval=-bound, maxval=bound)
        b = jax.random.uniform(kb, (dout,), jnp.float32, minval=-bound, maxval=bound)
        if i == len(sizes) - 2 and out_scale != 1.0:
            w = w * out_scale
            b = jnp.zeros_like(b)
        layers.append({"w": w, "b": b})
    return layers


def apply_mlp(layers: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
    for layer in layers[:-1]:
        x = jax.nn.elu(x @ layer["w"] + layer["b"])
    last = layers[-1]
    return x @ last["w"] + last["b"]


def init_params(rng: jax.Array, obs_size: int, act_size: int, log_std: float = -1.2) -> dict:
    k_act, k_crit = jax.random.split(rng)
    return {
        "actor": init_mlp(k_act, obs_size, act_size, out_scale=0.01),
        "critic": init_mlp(k_crit, obs_size, 1, out_scale=1.0),
        "log_std": jnp.full((act_size,), log_std, jnp.float32),
    }


def load_pt_checkpoint(
    path: str, obs_size: int, act_size: int, log_std: float
) -> tuple[dict, NormState, dict]:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    state = ckpt["agent"]
    old_obs = int(ckpt["obs"])

    def _load_net(prefix: str) -> list[dict[str, jax.Array]]:
        layers = []
        for idx in (0, 2, 4):
            w_pt = state[f"{prefix}.{idx}.weight"].detach().cpu().numpy().astype(np.float32)  # (out, in)
            b_pt = state[f"{prefix}.{idx}.bias"].detach().cpu().numpy().astype(np.float32)    # (out,)
            if idx == 0 and w_pt.shape[1] < obs_size:
                grown = np.zeros((w_pt.shape[0], obs_size), np.float32)
                grown[:, :old_obs] = w_pt
                w_pt = grown
            layers.append({"w": jnp.asarray(w_pt.T), "b": jnp.asarray(b_pt)})
        return layers

    params = {
        "actor": _load_net("actor"),
        "critic": _load_net("critic"),
        "log_std": jnp.full((act_size,), log_std, jnp.float32),
    }
    mean = np.zeros(obs_size, np.float32)
    var = np.ones(obs_size, np.float32)
    count = np.full(obs_size, 1e4, np.float32)
    old_mean = np.asarray(ckpt["mean"], np.float32)[:old_obs]
    old_var = np.asarray(ckpt["var"], np.float32)[:old_obs]
    active = old_var >= 1e-4
    mean[:old_obs] = np.where(active, old_mean, 0.0)
    var[:old_obs] = np.where(active, old_var, 1.0)
    count[:old_obs] = np.where(active, 1e8, 1e4)
    norm = NormState(mean=jnp.asarray(mean), var=jnp.asarray(var), count=jnp.asarray(count))
    return params, norm, ckpt.get("stats", {})


def save_pt_checkpoint(
    path: Path, params: dict, norm: NormState, obs_size: int, act_size: int, stats: dict
) -> None:
    state = {}
    for prefix in ("actor", "critic"):
        for layer_i, pt_idx in enumerate((0, 2, 4)):
            w_jax = np.asarray(params[prefix][layer_i]["w"], np.float32)  # (in, out)
            b_jax = np.asarray(params[prefix][layer_i]["b"], np.float32)  # (out,)
            state[f"{prefix}.{pt_idx}.weight"] = torch.from_numpy(w_jax.T.copy())
            state[f"{prefix}.{pt_idx}.bias"] = torch.from_numpy(b_jax.copy())
    state["log_std"] = torch.from_numpy(np.asarray(params["log_std"], np.float32).copy())
    payload = {
        "agent": state,
        "mean": np.asarray(norm.mean, np.float64),
        "var": np.asarray(norm.var, np.float64),
        "obs": obs_size,
        "act": act_size,
        "stats": stats,
    }
    torch.save(payload, path)


def gaussian_log_prob(mean: jax.Array, log_std: jax.Array, action: jax.Array) -> jax.Array:
    std = jnp.exp(log_std)
    var = std**2
    log_scale = log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.sum(-0.5 * ((action - mean) ** 2) / var - log_scale, axis=-1)
