from dataclasses import dataclass

import jax
from jax import lax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Float, jaxtyped

TYPE_CHECKER = beartype


@dataclass
class ModelArgs:
    dim: int
    depth: int
    heads: int
    patch: int
    channels: int
    time_dim: int
    context_dim: int
    timestep_dim: int
    timestep_std: float = 1.0


def unstack(x, axis=0):
    return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


class FourierFeatures(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs) -> None:
        self.weight = nnx.Param(
            jax.random.uniform(rngs.params(), (args.timestep_dim // 2, 1)) * args.timestep_std
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x: Float[Array, ""]) -> Float[Array, " D"]:
        f = 2 * jnp.pi * x @ self.weight.T
        return jnp.concatenate([f.cos(), f.sin()], dim=-1)


# TODO: move to a generic library
class GLU(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs, use_bias: bool = True) -> None:
        self.proj = nnx.Linear(
            in_features=in_features,
            out_features=out_features * 2,
            use_bias=use_bias,
            rngs=rngs,
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x: Float[Array, "B N D"]) -> Float[Array, "B N D"]:
        x, gate = unstack(self.proj(x), axis=-1)
        return x + nnx.silu(gate)


class FeedForward(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs) -> None:
        self.linear_in = GLU(args=args, rngs=rngs)
        self.linear_out = nnx.Linear(in_features=args.dim, out_features=args.dim, use_bias=True, rngs=rngs)

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x):
        x = self.linear_in(x)
        # silu act inside linear_in
        x = self.linear_out(x)
        return x


class TransformerBlock(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs) -> None:
        self.pre_norm = nnx.LayerNorm(num_features=args.dim, rngs=rngs)

        self.attn = nnx.MultiHeadAttention(
            num_heads=args.heads,
            in_features=args.dim,
            out_features=args.dim,
            decode=False,
            normalize_qk=False,
            rngs=rngs,
        )

        self.ff_norm = nnx.LayerNorm(num_features=args.dim, rngs=rngs)
        self.ff = FeedForward(args=args, rngs=rngs)

        self.global_proj = nnx.Linear(
            in_features=args.dim,
            out_features=args.dim * 6,
            use_bias=False,
            rngs=rngs,
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x):
        pass


class ContinuousTransformer(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs) -> None:
        self.proj_in = nnx.Linear(
            in_features=args.channels * args.patch,
            out_features=args.dim,
            use_bias=False,
            rngs=rngs,
        )
        self.proj_out = nnx.Linear(
            in_features=args.dim,
            out_features=args.channels * args.patch,
            use_bias=False,
            rngs=rngs,
        )

        self.layers = [TransformerBlock(args=args, rngs=rngs) for _ in range(args.depth)]

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x: Float[Array, "x_seq model_dim"], ctx: Float[Array, "ctx_seq model_dim"]):
        x = self.proj_in(x)

        # TODO: rotary emb, cached?

        for layer in self.layers:
            x = layer(x=x, ctx=ctx)

        x = self.proj_out(x)

        return x


class DiT(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs) -> None:
        self.args = ModelArgs

        # patching conv
        self.patch = nnx.Conv(
            in_features=args.channels,
            out_features=args.channels,
            kernel_size=1,
            strides=1,
        )

        # timestep and projections
        self.to_timestep = FourierFeatures(args, rngs)
        self.time_proj_1 = nnx.Linear(
            in_features=args.timestep_dim,
            out_features=args.dim,
            use_bias=True,
            rngs=rngs,
        )
        self.time_proj_2 = nnx.Linear(in_features=args.dim, out_features=args.dim, use_bias=True, rngs=rngs)

        # global cond projections
        self.global_proj_1 = nnx.Linear(
            in_features=args.dim, out_features=args.dim, use_bias=False, rngs=rngs
        )
        self.global_proj_2 = nnx.Linear(
            in_features=args.dim, out_features=args.dim, use_bias=False, rngs=rngs
        )

        # context projections
        self.context_proj_1 = nnx.Linear(
            in_features=args.context_dim,
            out_features=args.context_dim,
            use_bias=False,
            rngs=rngs,
        )
        self.context_proj_2 = nnx.Linear(
            in_features=args.context_dim,
            out_features=args.context_dim,
            use_bias=False,
            rngs=rngs,
        )

        self.transformer = ContinuousTransformer(args, rngs=rngs)

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(
        self,
        x: Float[Array, "x_chan x_dim"],
        t: Float[Array, " 1"],
        g: Float[Array, " global_dim"],
        ctx: Float[Array, "ctx_seq ctx_dim"],
    ):
        """
        The Stable Audio 1.0 DiT

        - Doesn't have any prepend cond but DOES apply global cond as a prepend.
        - Uses cross attention for the prompt
        """
        # patch x
        x = self.patch(x) + x

        # project timesteps
        t = self.to_timestep(t)
        t = self.time_proj_1(t)
        t = nnx.silu(t)
        t = self.time_proj_2(t)

        # project global cond
        g = self.global_proj_1(g)
        g = nnx.silu(g)
        g = self.global_proj_2(g)

        # project context
        ctx = self.context_proj_1(ctx)
        ctx = nnx.silu(ctx)
        ctx = self.context_proj_2(ctx)

        # combine global and timestep, add extra dimension
        g = jnp.expand_dims(g + t, axis=1)

        # combine g and x
        x = jnp.concat([g, x], axis=1)

        # oh yeah, attend it ;)
        out = self.transformer(x=x, ctx=ctx)

        return out
