"""Classifier-free guidance in Pi0.sample_actions.

WHY THE VELOCITY HEAD IS STUBBED
    The `dummy` gemma variant cannot be used to test input-dependence: its embed_prefix
    returns bit-identical tokens even when the images differ (measured: images differing by
    0.5 give max|d embed_prefix| == 0.0), so cond and uncond branches are indistinguishable
    and every guidance scale looks like a no-op. That is also why pi0_test.py only ever uses
    nnx.eval_shape. Stubbing action_out_proj so the two batch halves differ BY CONSTRUCTION
    makes the guidance arithmetic exactly checkable, and pins the cond/uncond split order --
    the property most likely to be silently wrong.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0_config as _pi0_config

_AH, _AD, _B, _STEPS = 4, 8, 2, 4


def _model_and_obs():
    cfg = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=_AH,
        action_dim=_AD,
        max_token_len=16,
        pi05=True,
    )
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=_B)
    # fake_obs fills the prompt with ONE repeated token id, so permuting it is a no-op --
    # the uncond prompt must use different ids to be a genuinely different input.
    uncond = dataclasses.replace(obs, tokenized_prompt=obs.tokenized_prompt.at[:, :6].set(7))
    return model, obs, uncond


def _stub_velocity(model):
    """cond half -> v=1, uncond half -> v=0, so v_guided = 0 + (1+w)(1-0) = 1+w."""

    def fake_proj(x):
        b = x.shape[0]
        if b == 2 * _B:
            half = b // 2
            return jnp.concatenate([jnp.ones((half, _AH, _AD)), jnp.zeros((half, _AH, _AD))], axis=0)
        return jnp.ones((b, _AH, _AD))

    model.action_out_proj = fake_proj


@pytest.mark.parametrize("guidance_scale", [1.0, 3.0, 10.0])
def test_identical_uncond_is_a_noop(guidance_scale):
    """v_cond == v_uncond => guidance must vanish at ANY scale."""
    model, obs, _ = _model_and_obs()
    noise = jax.random.normal(jax.random.key(2), (_B, _AH, _AD))
    plain = model.sample_actions(jax.random.key(1), obs, num_steps=_STEPS, noise=noise)
    guided = model.sample_actions(
        jax.random.key(1),
        obs,
        num_steps=_STEPS,
        noise=noise,
        guidance_scale=guidance_scale,
        uncond_observation=obs,
    )
    assert jnp.max(jnp.abs(plain - guided)) < 1e-5


def test_scale_zero_short_circuits():
    """guidance_scale=0 must reproduce conditional sampling even with a different uncond."""
    model, obs, uncond = _model_and_obs()
    noise = jax.random.normal(jax.random.key(2), (_B, _AH, _AD))
    plain = model.sample_actions(jax.random.key(1), obs, num_steps=_STEPS, noise=noise)
    guided = model.sample_actions(
        jax.random.key(1),
        obs,
        num_steps=_STEPS,
        noise=noise,
        guidance_scale=0.0,
        uncond_observation=uncond,
    )
    assert jnp.array_equal(plain, guided)


@pytest.mark.parametrize("guidance_scale", [0.0, 1.0, 2.0, 5.0])
def test_guidance_arithmetic_and_split_order(guidance_scale):
    """Closed form: with x_0=0 and v=1+w held constant, x_0 integrates to -(1+w).

    If jnp.split returned [uncond, cond] instead of [cond, uncond], w=1 would give +0.0
    rather than -2.0, so this also pins the split order.
    """
    model, obs, uncond = _model_and_obs()
    _stub_velocity(model)
    noise = jnp.zeros((_B, _AH, _AD))
    out = model.sample_actions(
        jax.random.key(1),
        obs,
        num_steps=_STEPS,
        noise=noise,
        guidance_scale=guidance_scale,
        uncond_observation=uncond,
    )
    assert float(out[0, 0, 0]) == pytest.approx(-(1.0 + guidance_scale), abs=1e-5)
