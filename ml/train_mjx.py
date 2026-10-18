"""Train the Capoeira imitation policy with PPO in pure JAX + MJX (MuJoCo XLA).

    uv run python train_mjx.py [--steps 40e6] [--envs 2048] [--horizon 64] [--skill kicks] [--init runs/ginga_tri/last.pt]

Runs the entire rollout (T steps across N parallel MJX environments), GAE
advantage calculation, and PPO minibatch updates on the accelerator (GPU/TPU or
CPU), while reading and writing the exact same PyTorch `.pt` checkpoint format
(`best.pt`, `last.pt`, `log.jsonl`) used by `train.py`, `check.py`, `choose.py`,
and `export_web.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from env import ENDS, FALL_SPEEDS, LANDINGS
from env_mjx import EnvState, MjxGingaEnv


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


# ---- Main PPO loop ----------------------------------------------------------


def main() -> None:
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=float, default=40e6)
    parser.add_argument("--envs", type=int, default=0, help="parallel envs (default: 2048 on GPU, 64 on CPU)")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--minutes", type=float, default=90.0)
    parser.add_argument("--out", default="runs/ginga_mjx")
    parser.add_argument("--init", help="PyTorch .pt checkpoint to warm-start from")
    parser.add_argument("--skill", default="all", choices=["all", "ginga", "kicks", "esquiva", "recover"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-std", type=float, default=-2.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--start-size", type=int, default=-1, help="initial curriculum size (-1: infer from --init)")
    parser.add_argument("--patience", type=int, default=100, help="early-stop iterations without improvement at max curriculum (0 disables)")
    parser.add_argument("--cache-dir", default="", help="persistent JAX XLA compilation cache directory")
    parser.add_argument("--impl", default="jax", choices=["jax", "warp"])
    args = parser.parse_args()

    cache_dir = args.cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

    on_gpu = any(d.platform == "gpu" for d in jax.devices())
    num_envs = args.envs if args.envs > 0 else (2048 if on_gpu else 64)
    horizon = args.horizon
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = MjxGingaEnv(skill=args.skill, max_seconds=40.0, impl=args.impl)
    obs_size, act_size = env.obs_size, env.act_size

    rng = jax.random.PRNGKey(0)
    rng, k_init, k_envs = jax.random.split(rng, 3)
    init_stats: dict = {}
    if args.init:
        params, norm, init_stats = load_pt_checkpoint(args.init, obs_size, act_size, args.log_std)
    else:
        params = init_params(k_init, obs_size, act_size, log_std=-1.2)
        norm = init_norm(obs_size)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale(-1.0),
    )
    opt_state = tx.init(params)

    start_cur, start_fall = 0, 0
    if args.start_size >= 0:
        if args.skill == "recover":
            start_fall = min(args.start_size, len(FALL_SPEEDS) - 1)
        else:
            start_cur = min(args.start_size, 3)
    elif init_stats:
        if args.skill in ("kicks", "esquiva") and (init_stats.get("kicking") or 0.0) > 0.05:
            start_cur = min(int(init_stats.get("size") or 0), 3)
        elif args.skill == "recover" and (init_stats.get("down") or 0.0) > 0.05:
            start_fall = min(int(init_stats.get("size") or 0), len(FALL_SPEEDS) - 1)

    curriculum = jnp.array(start_cur, jnp.int32)
    fall_level = jnp.array(start_fall, jnp.int32)

    v_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None, None)))
    v_step = jax.vmap(env.step, in_axes=(0, 0, None, None))

    env_keys = jax.random.split(k_envs, num_envs)
    env_state = v_reset(env_keys, curriculum, fall_level)

    gamma, lam, clip_eps = 0.95, 0.95, 0.2
    batch_size = horizon * num_envs
    num_minibatches = min(args.minibatches, batch_size)
    minibatch_size = batch_size // num_minibatches

    @jax.jit
    def train_iteration(
        params: dict,
        opt_state: optax.OptState,
        norm: NormState,
        env_state: EnvState,
        rng: jax.Array,
        actor_lr: jax.Array,
        critic_lr: jax.Array,
        curriculum: jax.Array,
        fall_level: jax.Array,
    ):
        def rollout_step(carry, _):
            st, nrm, key = carry
            nrm = update_norm(nrm, st.obs)
            o_norm = apply_norm(nrm, st.obs)
            mean = apply_mlp(params["actor"], o_norm)
            val = jnp.squeeze(apply_mlp(params["critic"], o_norm), axis=-1)
            key, k_act = jax.random.split(key)
            noise = jax.random.normal(k_act, mean.shape)
            action = mean + noise * jnp.exp(params["log_std"])
            logp = gaussian_log_prob(mean, params["log_std"], action)

            next_st, rew, term, trunc, info = v_step(st, action, curriculum, fall_level)
            done = term | trunc
            final_o_norm = apply_norm(nrm, info["final_obs"])
            final_val = jnp.squeeze(apply_mlp(params["critic"], final_o_norm), axis=-1)
            boot = jnp.where(trunc & (~term), final_val, 0.0)

            transition = (o_norm, action, logp, rew, done.astype(jnp.float32), val, boot, info)
            return (next_st, nrm, key), transition

        (env_state, norm, rng), traj = jax.lax.scan(
            rollout_step, (env_state, norm, rng), None, length=horizon
        )
        b_obs, b_act, b_logp, b_rew, b_done, b_val, b_boot, b_info = traj

        next_o_norm = apply_norm(norm, env_state.obs)
        next_value = jnp.squeeze(apply_mlp(params["critic"], next_o_norm), axis=-1)

        def gae_step(carry, t_inputs):
            last_gae, following = carry
            rew_t, done_t, val_t, boot_t = t_inputs
            nonterminal = 1.0 - done_t
            delta = rew_t + gamma * (following * nonterminal + boot_t) - val_t
            gae = delta + gamma * lam * nonterminal * last_gae
            return (gae, val_t), gae

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros(num_envs, jnp.float32), next_value),
            (b_rew, b_done, b_val, b_boot),
            reverse=True,
        )
        returns = advantages + b_val

        flat_obs = b_obs.reshape((batch_size, obs_size))
        flat_act = b_act.reshape((batch_size, act_size))
        flat_logp = b_logp.reshape((batch_size,))
        flat_adv = advantages.reshape((batch_size,))
        flat_ret = returns.reshape((batch_size,))
        flat_adv = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

        def epoch_step(carry, _):
            p, ost, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, batch_size)[: num_minibatches * minibatch_size]
            perm = perm.reshape((num_minibatches, minibatch_size))

            def mb_step(p_ost, idx):
                p_cur, ost_cur = p_ost
                mb_o = flat_obs[idx]
                mb_a = flat_act[idx]
                mb_lp = flat_logp[idx]
                mb_adv = flat_adv[idx]
                mb_ret = flat_ret[idx]

                def loss_fn(prms):
                    mean = apply_mlp(prms["actor"], mb_o)
                    logp = gaussian_log_prob(mean, prms["log_std"], mb_a)
                    ratio = jnp.exp(logp - mb_lp)
                    p_loss = -jnp.mean(
                        jnp.minimum(ratio * mb_adv, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv)
                    )
                    val = jnp.squeeze(apply_mlp(prms["critic"], mb_o), axis=-1)
                    v_loss = jnp.mean((val - mb_ret) ** 2)
                    return p_loss + 0.5 * v_loss

                grads = jax.grad(loss_fn)(p_cur)
                updates, ost_next = tx.update(grads, ost_cur, p_cur)
                scaled_updates = {
                    "actor": jax.tree.map(lambda u: u * actor_lr, updates["actor"]),
                    "critic": jax.tree.map(lambda u: u * critic_lr, updates["critic"]),
                    "log_std": updates["log_std"] * actor_lr,
                }
                p_next = optax.apply_updates(p_cur, scaled_updates)
                return (p_next, ost_next), None

            (p, ost), _ = jax.lax.scan(mb_step, (p, ost), perm)
            return (p, ost, key), None

        (params, opt_state, rng), _ = jax.lax.scan(
            epoch_step, (params, opt_state, rng), None, length=args.epochs
        )

        metrics = {
            "reward": jnp.mean(b_rew),
            "pose": jnp.mean(b_info["pose"]),
            "vel": jnp.mean(b_info["vel"]),
            "ee": jnp.mean(b_info["ee"]),
            "root": jnp.mean(b_info["root"]),
            "feet": jnp.mean(b_info["feet"]),
            "kicking": jnp.mean(b_info["kicking"]),
            "strike": jnp.mean(b_info["strike"]),
            "pivot": jnp.mean(b_info["pivot"]),
            "down": jnp.mean(b_info["down"]),
            "up": jnp.mean(b_info["up"]),
            "done": b_done,
            "episode_steps": b_info["episode_steps"],
            "end": b_info["end"],
            "landing": b_info["landing"],
            "got_up": b_info["got_up"],
            "from_fall": b_info["from_fall"],
            "kick_attempts": jnp.sum(b_info["kick_attempt"]),
            "kick_oks": jnp.sum(b_info["kick_ok"]),
        }
        return params, opt_state, norm, env_state, rng, metrics

    total_steps, iteration, start = 0, 0, time.time()
    post_jit_start: float | None = None
    post_jit_steps = 0
    log = open(out / "log.jsonl", "a")
    best = -1.0
    finished_lengths: list[float] = []
    finished_ends: list[tuple[int, int, float, bool]] = []
    kick_success_ema, kick_count, iters_at_level = 0.0, 0, 0
    fall_success_ema, fall_count = 0.0, 0
    stale_iters = 0

    in_cooldown = False
    while total_steps < args.steps and (time.time() - start) < args.minutes * 60:
        iteration += 1
        iters_at_level += 1
        elapsed_frac = (time.time() - start) / max(args.minutes * 60, 1.0)
        if not in_cooldown and elapsed_frac >= 0.75:
            in_cooldown = True
            stale_iters = 0
            params["log_std"] = jnp.minimum(params["log_std"], -2.6)
            if args.skill in ("kicks", "esquiva") and int(curriculum) < 3:
                curriculum = jnp.array(3, jnp.int32)
                best = -1.0
            elif args.skill == "recover" and int(fall_level) < len(FALL_SPEEDS) - 1:
                fall_level = jnp.array(len(FALL_SPEEDS) - 1, jnp.int32)
                best = -1.0
        frac = 0.3 if in_cooldown else max(1.0 - total_steps / args.steps, 0.3)
        critic_lr = jnp.array(args.lr * frac, jnp.float32)
        warmup = min((iteration - 1) / 5.0, 1.0) if args.init else 1.0
        actor_lr = jnp.array(args.lr * frac * warmup, jnp.float32)

        params, opt_state, norm, env_state, rng, m = train_iteration(
            params, opt_state, norm, env_state, rng, actor_lr, critic_lr, curriculum, fall_level
        )
        jax.block_until_ready(m["reward"])
        total_steps += batch_size
        if post_jit_start is None:
            post_jit_start = time.time()
        else:
            post_jit_steps += batch_size

        done_np = np.asarray(m["done"]) > 0.5
        if np.any(done_np):
            ep_steps = np.asarray(m["episode_steps"])[done_np]
            ends = np.asarray(m["end"])[done_np]
            lands = np.asarray(m["landing"])[done_np]
            ups = np.asarray(m["got_up"])[done_np]
            ffalls = np.asarray(m["from_fall"])[done_np]
            finished_lengths.extend((ep_steps * env.control_dt).tolist()[-200:])
            finished_lengths = finished_lengths[-200:]
            for e, l, u, ff in zip(ends[-200:], lands[-200:], ups[-200:], ffalls[-200:]):
                finished_ends.append((int(e), int(l), float(u), bool(ff)))
            finished_ends = finished_ends[-200:]

        # Curriculum progression for kicks/esquiva
        att = int(m["kick_attempts"])
        if att > 0 and args.skill in ("kicks", "esquiva"):
            rate = float(m["kick_oks"]) / att
            for _ in range(min(att, 20)):
                kick_count += 1
                kick_success_ema += 0.1 * (rate - kick_success_ema)
        if args.skill in ("kicks", "esquiva") and int(curriculum) < 3:
            due = elapsed_frac >= 0.18 * (int(curriculum) + 1)
            passed = iters_at_level >= 3 and kick_count >= 20 and kick_success_ema > 0.70
            if passed or due:
                curriculum = jnp.array(int(curriculum) + 1, jnp.int32)
                kick_count, kick_success_ema, iters_at_level, stale_iters = 0, 0.0, 0, 0
                finished_lengths.clear()
                finished_ends.clear()
                best = -1.0

        # Curriculum progression for recover
        if args.skill == "recover" and int(fall_level) < len(FALL_SPEEDS) - 1:
            rec_real = [u for _, l, u, ff in finished_ends[-50:] if l > 0 and ff]
            if len(rec_real) >= 20:
                fall_count += len(rec_real)
                fall_success_ema = float(np.mean(rec_real))
            due = elapsed_frac >= 0.18 * (int(fall_level) + 1)
            passed = iters_at_level >= 3 and fall_count >= 20 and fall_success_ema > 0.45
            if passed or due:
                fall_level = jnp.array(int(fall_level) + 1, jnp.int32)
                fall_count, fall_success_ema, iters_at_level, stale_iters = 0, 0.0, 0, 0
                finished_lengths.clear()
                finished_ends.clear()
                best = -1.0

        kicking_share = float(m["kicking"])
        sps = (
            int(post_jit_steps / max(time.time() - post_jit_start, 1e-3))
            if post_jit_start is not None and post_jit_steps > 0
            else int(total_steps / max(time.time() - start, 1e-3))
        )
        stats = {
            "iteration": iteration,
            "steps": total_steps,
            "minutes": round((time.time() - start) / 60, 2),
            "sps": sps,
            "reward": round(float(m["reward"]), 4),
            "episode_s": round(float(np.mean(finished_lengths[-50:])), 2) if finished_lengths else None,
            "std": round(float(jnp.mean(jnp.exp(params["log_std"]))), 3),
            "pose": round(float(m["pose"]), 3),
            "vel": round(float(m["vel"]), 3),
            "ee": round(float(m["ee"]), 3),
            "root": round(float(m["root"]), 3),
            "feet": round(float(m["feet"]), 3),
            "kicking": round(kicking_share, 3),
            "strike": round(float(m["strike"]) / kicking_share, 3) if kicking_share > 1e-4 else None,
            "pivot": round(float(m["pivot"]) / kicking_share, 3) if kicking_share > 1e-4 else None,
            "size": int(fall_level) if args.skill == "recover" else int(curriculum),
            "down": round(float(m["down"]), 3),
            "up": round(float(m["up"]), 3),
        }
        if finished_ends:
            stats.update(
                {
                    f"end_{name}": round(sum(e == k for e, *_ in finished_ends) / len(finished_ends), 3)
                    for k, name in enumerate(ENDS)
                }
            )
            fell = [(l, u) for _, l, u, _ in finished_ends if l > 0]
            real = [u for _, l, u, ff in finished_ends if l > 0 and ff]
            if fell:
                stats.update(
                    {
                        f"fell_{name}": round(sum(l == k + 1 for l, _ in fell) / len(fell), 3)
                        for k, name in enumerate(LANDINGS)
                    }
                )
                stats["got_up"] = round(float(np.mean([u for _, u in fell])), 3)
            if real:
                stats["got_up_falls"] = round(float(np.mean(real)), 3)

        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()

        score = stats["reward"] * (stats["episode_s"] or 1.0)
        if iteration == 1:
            save_pt_checkpoint(out / "best.pt", params, norm, obs_size, act_size, stats)
            save_pt_checkpoint(out / "last.pt", params, norm, obs_size, act_size, stats)
        elif iteration >= 3:
            if score >= best:
                best = score
                stale_iters = 0
                save_pt_checkpoint(out / "best.pt", params, norm, obs_size, act_size, stats)
                save_pt_checkpoint(out / "last.pt", params, norm, obs_size, act_size, stats)
            else:
                stale_iters += 1
                if iteration % 10 == 0:
                    save_pt_checkpoint(out / "last.pt", params, norm, obs_size, act_size, stats)

        at_max_level = (
            (args.skill in ("kicks", "esquiva") and int(curriculum) >= 3)
            or (args.skill == "recover" and int(fall_level) >= len(FALL_SPEEDS) - 1)
            or args.skill in ("ginga", "all")
        )
        if args.patience > 0 and at_max_level and iters_at_level >= 30:
            if not in_cooldown and stale_iters >= args.patience:
                in_cooldown = True
                stale_iters = 0
                params["log_std"] = jnp.minimum(params["log_std"], -2.6)
            elif in_cooldown and stale_iters >= max(30, args.patience // 3):
                print(json.dumps({"early_stop": True, "iteration": iteration, "minutes": stats["minutes"]}), flush=True)
                break

    save_pt_checkpoint(out / "last.pt", params, norm, obs_size, act_size, stats)
    log.close()


if __name__ == "__main__":
    main()
