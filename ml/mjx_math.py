"""Pure JAX quaternion, heading and floor-placement helpers (the jnp counterparts of env_math.py)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_angle(a: jax.Array, b: jax.Array) -> jax.Array:
    dot = jnp.clip(jnp.abs(jnp.sum(a * b, axis=-1)), 0.0, 1.0 - 1e-7)
    return 2.0 * jnp.arccos(dot)


def nlerp(a: jax.Array, b: jax.Array, t: jax.Array) -> jax.Array:
    dot = jnp.sum(a * b, axis=-1, keepdims=True)
    b = jnp.where(dot < 0.0, -b, b)
    q = a + (b - a) * t
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


def yaw_of(q: jax.Array) -> jax.Array:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fx = 2.0 * (x * z + w * y)
    fz = 1.0 - 2.0 * (x * x + y * y)
    return jnp.arctan2(fx, fz)


def yaw_matrix(yaw: jax.Array) -> jax.Array:
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    z = jnp.zeros_like(yaw)
    o = jnp.ones_like(yaw)
    return jnp.stack(
        [
            jnp.stack([c, z, -s], axis=-1),
            jnp.stack([z, o, z], axis=-1),
            jnp.stack([s, z, c], axis=-1),
        ],
        axis=-2,
    )


def quat_matrix(q: jax.Array) -> jax.Array:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def place_xz(frame: jax.Array, xz: jax.Array) -> jax.Array:
    c, s = jnp.cos(frame[2]), jnp.sin(frame[2])
    x, z = xz[..., 0], xz[..., 1]
    return jnp.stack([c * x + s * z + frame[0], -s * x + c * z + frame[1]], axis=-1)


def unplace_xz(frame: jax.Array, xz: jax.Array) -> jax.Array:
    c, s = jnp.cos(frame[2]), jnp.sin(frame[2])
    x, z = xz[..., 0] - frame[0], xz[..., 1] - frame[1]
    return jnp.stack([c * x - s * z, s * x + c * z], axis=-1)


def place_pose(frame: jax.Array, q: jax.Array, v: jax.Array, feet: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    xz = place_xz(frame, jnp.stack([q[0], q[2]]))
    q = q.at[0].set(xz[0]).at[2].set(xz[1])
    yaw_q = jnp.array([jnp.cos(frame[2] / 2.0), 0.0, jnp.sin(frame[2] / 2.0), 0.0], dtype=q.dtype)
    q = q.at[3:7].set(quat_mul(yaw_q, q[3:7]))
    c, s = jnp.cos(frame[2]), jnp.sin(frame[2])
    vx, vz = c * v[0] + s * v[2], -s * v[0] + c * v[2]
    v = v.at[0].set(vx).at[2].set(vz)
    return q, v, place_xz(frame, feet)


def smoothstep(x: jax.Array) -> jax.Array:
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
