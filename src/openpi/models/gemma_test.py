import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma


def _build(*, insulate_expert0: bool) -> _gemma.Module:
    config = _gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4)
    return _gemma.Module(configs=[config, config], embed_dtype="float32", insulate_expert0=insulate_expert0)


def _expert0_grad_norm(grads) -> float:
    flat = jax.tree_util.tree_leaves_with_path(grads)
    return sum(float(jnp.sum(jnp.square(v))) for path, v in flat if "_1" not in jax.tree_util.keystr(path))


def test_insulate_expert0_blocks_cross_expert_gradient():
    key = jax.random.key(0)
    len0, len1, batch = 3, 2, 1
    x0 = jax.random.normal(jax.random.key(1), (batch, len0, 8))
    x1 = jax.random.normal(jax.random.key(2), (batch, len1, 8))
    positions = jnp.arange(len0 + len1)[None, :]
    mask = jnp.ones((batch, len0 + len1, len0 + len1), dtype=bool)

    def loss_fn(params, module, target_expert):
        (out0, out1), _ = module.apply(params, [x0, x1], positions, mask)
        return jnp.sum(out1) if target_expert == 1 else jnp.sum(out0)

    for insulate in (False, True):
        module = _build(insulate_expert0=insulate)
        params = nn.Module.init(module, key, [x0, x1], positions, mask)

        # Expert 1's own loss should always reach expert 0's own gradient path unaffected -
        # insulation only cuts the path FROM other experts TO expert 0, not expert 0's own.
        grad_from_expert0_loss = jax.grad(loss_fn)(params, module, 0)
        assert _expert0_grad_norm(grad_from_expert0_loss) > 0

        # Expert 1's loss reaching into expert 0's params is exactly what insulation should
        # control: present when disabled, exactly zero when enabled.
        grad_from_expert1_loss = jax.grad(loss_fn)(params, module, 1)
        expert0_grad_norm = _expert0_grad_norm(grad_from_expert1_loss)
        if insulate:
            assert expert0_grad_norm == 0
        else:
            assert expert0_grad_norm > 0
