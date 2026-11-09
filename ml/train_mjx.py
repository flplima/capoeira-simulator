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

import jax
import jax.numpy as jnp
import numpy as np
import optax

from env_consts import FALL_SPEEDS, SIGMA_INIT, SIGMA_ROWS, SIGMA_WARMUP, TERMS
from env_math import tighten
from env_mirror import mirror_maps
from env_mjx import MjxGingaEnv
from mjx_eval import report
from mjx_math import error_sums
from mjx_state import EnvState
from train_log import ends_stats, tolerance_stats
from train_nets import (
    NormState,
    apply_mlp,
    apply_norm,
    gaussian_log_prob,
    init_norm,
    init_params,
    load_pt_checkpoint,
    save_pt_checkpoint,
    update_norm,
)
from train_ppo import advantages, batch, update


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
    parser.add_argument("--focus", default="", help="a move that gets half of the practice (a new or changed one)")
    parser.add_argument("--only", default="", help="the one move practised (a new kick learned on its own)")
    parser.add_argument("--gated", action="store_true", help="sizes grow only on success, never on the clock")
    parser.add_argument("--mirror", action="store_true", help="also train on the left-right mirror of every batch: each move learns both legs at once")
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

    env = MjxGingaEnv(skill=args.skill, max_seconds=40.0, impl=args.impl, focus=args.focus, only=args.only)
    obs_size, act_size = env.obs_size, env.act_size
    mirror = mirror_maps(env.m) if args.mirror else None

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
    v_step = jax.vmap(env.step, in_axes=(0, 0, None, None, None))

    env_keys = jax.random.split(k_envs, num_envs)
    env_state = v_reset(env_keys, curriculum, fall_level)

    gamma, lam, clip_eps = 0.95, 0.95, 0.2
    batch_size = horizon * num_envs
    num_minibatches = min(args.minibatches, batch_size)

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
        sigma: jax.Array,
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

            next_st, rew, term, trunc, info = v_step(st, action, curriculum, fall_level, sigma)
            done = term | trunc
            final_o_norm = apply_norm(nrm, info["final_obs"])
            final_val = jnp.squeeze(apply_mlp(params["critic"], final_o_norm), axis=-1)
            boot = jnp.where(trunc & (~term), final_val, 0.0)

            # The observation is kept raw: the update normalises it (and its mirror) with the rollout's final statistics
            transition = (st.obs, action, logp, rew, done.astype(jnp.float32), val, boot, info)
            return (next_st, nrm, key), transition

        (env_state, norm, rng), traj = jax.lax.scan(
            rollout_step, (env_state, norm, rng), None, length=horizon
        )
        b_obs, b_act, b_logp, b_rew, b_done, b_val, b_boot, b_info = traj

        next_o_norm = apply_norm(norm, env_state.obs)
        next_value = jnp.squeeze(apply_mlp(params["critic"], next_o_norm), axis=-1)
        adv = advantages(b_rew, b_done, b_val, b_boot, next_value, gamma, lam)
        flat = batch(
            params,
            norm,
            b_obs.reshape((batch_size, obs_size)),
            b_act.reshape((batch_size, act_size)),
            b_logp.reshape((batch_size,)),
            adv.reshape((batch_size,)),
            (adv + b_val).reshape((batch_size,)),
            mirror,
        )
        params, opt_state, rng = update(tx, params, opt_state, rng, flat, args.epochs, num_minibatches, actor_lr, critic_lr, clip_eps)

        # Each tolerance row's mean squared error per term and how many frames it saw: the next tolerances (env_math.tighten)
        err_sum, err_count = error_sums(b_info["err"], b_info["row"], b_info["kick"])
        metrics = {
            "err_mean": err_sum / jnp.maximum(err_count, 1.0),
            "err_count": err_count,
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

    # Tracking tolerances (env_consts.TERMS): loose at first, then each term's running mean error, only ever
    # tightened; a new size is a new task, and its tolerances start loose again.
    def loose() -> np.ndarray:
        return np.full((SIGMA_ROWS, len(TERMS)), SIGMA_INIT)

    sigma, ema, since_loose = loose(), loose(), 0

    def loosen() -> None:
        nonlocal sigma, ema, since_loose
        sigma, ema, since_loose = loose(), loose(), 0

    in_cooldown = False
    while total_steps < args.steps and (time.time() - start) < args.minutes * 60:
        iteration += 1
        iters_at_level += 1
        elapsed_frac = (time.time() - start) / max(args.minutes * 60, 1.0)
        if not in_cooldown and elapsed_frac >= 0.75:
            in_cooldown = True
            stale_iters = 0
            params["log_std"] = jnp.minimum(params["log_std"], -2.6)
            # A gated run keeps the size its successes reached.
            if not args.gated and args.skill in ("kicks", "esquiva") and int(curriculum) < 3:
                curriculum = jnp.array(3, jnp.int32)
                best = -1.0
                loosen()
            elif not args.gated and args.skill == "recover" and int(fall_level) < len(FALL_SPEEDS) - 1:
                fall_level = jnp.array(len(FALL_SPEEDS) - 1, jnp.int32)
                best = -1.0
                loosen()
        frac = 0.3 if in_cooldown else max(1.0 - total_steps / args.steps, 0.3)
        critic_lr = jnp.array(args.lr * frac, jnp.float32)
        warmup = min((iteration - 1) / 5.0, 1.0) if args.init else 1.0
        actor_lr = jnp.array(args.lr * frac * warmup, jnp.float32)

        params, opt_state, norm, env_state, rng, m = train_iteration(
            params, opt_state, norm, env_state, rng, actor_lr, critic_lr, curriculum, fall_level, jnp.asarray(sigma, jnp.float32)
        )
        jax.block_until_ready(m["reward"])
        since_loose += 1
        sigma, ema = tighten(sigma, np.asarray(m["err_mean"]), np.asarray(m["err_count"]), ema, since_loose > SIGMA_WARMUP)
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
            due = not args.gated and elapsed_frac >= 0.18 * (int(curriculum) + 1)
            passed = iters_at_level >= 3 and kick_count >= 20 and kick_success_ema > 0.70
            if passed or due:
                curriculum = jnp.array(int(curriculum) + 1, jnp.int32)
                kick_count, kick_success_ema, iters_at_level, stale_iters = 0, 0.0, 0, 0
                finished_lengths.clear()
                finished_ends.clear()
                best = -1.0
                loosen()

        # Curriculum progression for recover
        if args.skill == "recover" and int(fall_level) < len(FALL_SPEEDS) - 1:
            rec_real = [u for _, l, u, ff in finished_ends[-50:] if l > 0 and ff]
            if len(rec_real) >= 20:
                fall_count += len(rec_real)
                fall_success_ema = float(np.mean(rec_real))
            due = not args.gated and elapsed_frac >= 0.18 * (int(fall_level) + 1)
            passed = iters_at_level >= 3 and fall_count >= 20 and fall_success_ema > 0.45
            if passed or due:
                fall_level = jnp.array(int(fall_level) + 1, jnp.int32)
                fall_count, fall_success_ema, iters_at_level, stale_iters = 0, 0.0, 0, 0
                finished_lengths.clear()
                finished_ends.clear()
                best = -1.0
                loosen()

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
        stats.update(ends_stats(finished_ends))
        stats["tol"] = tolerance_stats(sigma)

        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n")
        log.flush()

        # best.pt: for kicks and esquivas the share of this iteration's moves finished standing (what `eval` and
        # check.py count; thousands a run), otherwise the reward over the episode length
        if args.skill in ("kicks", "esquiva"):
            score = float(m["kick_oks"]) / att if att > 0 else 0.0
        else:
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
    if args.skill in ("kicks", "esquiva"):
        report(env, out, log, params, norm, (obs_size, act_size, args.log_std), num_envs, rng, curriculum, fall_level, jnp.asarray(sigma, jnp.float32))
    log.close()


if __name__ == "__main__":
    main()
