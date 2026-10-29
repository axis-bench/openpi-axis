"""The global_ratio reducer: plain mean without weights, Eq. L = sum w l / (sum w + eps) with them."""

import jax.numpy as jnp
import numpy as np

from openpi.training import weighted_loss


def test_no_weights_is_exactly_the_mean():
    loss = jnp.asarray(np.random.default_rng(0).random((8, 10)), dtype=jnp.float32)
    assert weighted_loss.combine(loss, None) == jnp.mean(loss)


def test_all_ones_weights_reproduce_the_mean_up_to_reduction_order():
    loss = jnp.asarray(np.random.default_rng(1).random((64, 10)), dtype=jnp.float32)
    np.testing.assert_allclose(weighted_loss.combine(loss, jnp.ones(64)), jnp.mean(loss), rtol=1e-6)


def test_weights_are_renormalised_over_the_batch():
    w = jnp.asarray([1.0, 2.0, 3.0, 2.0])
    np.testing.assert_allclose(jnp.mean(weighted_loss.renormalize(w)), 1.0, rtol=1e-6)
    per_example = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(weighted_loss.weighted_mean(per_example, w), 1.0 / 8.0, rtol=1e-5)
    assert weighted_loss.combine(jnp.ones((4, 2)), jnp.zeros(4)) == 0.0
