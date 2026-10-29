"""Per-sample loss weights reduced over the global batch: the `global_ratio` reducer.

A SignalBundle may carry a float32 weight per corpus row. The AXIS loader (axis_bundle) puts each
planned sample's weight into the batch under LOSS_WEIGHT_KEY, and `combine` turns per-chunk losses
and weights into the scalar objective. The weights are first renormalised so that the batch mean
weight is 1, which keeps the gradient scale of plain behaviour cloning (so the same learning rate
and weight decay apply), and then

    L = sum_i w_i * l_i / (sum_i w_i + eps),   eps = 1e-6,

over the whole global batch: under JAX the batch axis is sharded across devices, but both sums run
over all of it. `combine(chunked_loss, None)` returns exactly `jnp.mean(chunked_loss)`, so a run
without weights is bit-for-bit the plain loss.

Online tier (openpi venv, jax).
"""

from __future__ import annotations

import jax.numpy as jnp

import openpi.shared.array_typing as at

# The batch key the per-sample weight travels under, from the loader to train_step.
LOSS_WEIGHT_KEY = "axis_loss_weight"

# The epsilon of both normalisations; also the all-zero guard (0.0 instead of NaN).
EPS = 1e-6


def renormalize(weights: at.Array, *, eps: float = EPS) -> at.Array:
    """Scale weights so that (1/B) sum_i w_i == 1 (a per-batch renormalisation).

    Keeping the batch mean weight at 1 lets a weighted run use an unweighted run's
    learning rate and weight decay. ``+ eps`` in the denominator is the all-zero guard.
    """
    b = weights.shape[0]
    return b * weights / (jnp.sum(weights) + eps)


def weighted_mean(
    per_example_loss: at.Float[at.Array, " b"],
    weights: at.Float[at.Array, " b"],
    *,
    eps: float = EPS,
) -> at.Float[at.Array, ""]:
    """``L = sum_i w_i l_i / (sum_i w_i + eps)``, on renormalised weights.

    With equal weights this collapses to ``mean(per_example_loss)`` (up to eps), so a
    weighted and an unweighted run train at the same loss scale.
    """
    w = renormalize(jnp.asarray(weights, dtype=per_example_loss.dtype), eps=eps)
    return jnp.sum(w * per_example_loss) / (jnp.sum(w) + eps)


def combine(chunked_loss: at.Array, weights: at.Array | None = None, *, eps: float = EPS) -> at.Array:
    """Reduce a ``[*b ah]`` per-chunk loss to the scalar training objective.

    ``weights is None`` is the vanilla-BC path and returns literally
    ``jnp.mean(chunked_loss)``. Otherwise the chunk axes are averaged into one
    per-example loss l_i and the weighted mean above is taken over the batch axis.
    """
    if weights is None:
        return jnp.mean(chunked_loss)
    per_example = jnp.mean(chunked_loss, axis=tuple(range(1, chunked_loss.ndim)))
    return weighted_mean(per_example, weights, eps=eps)
