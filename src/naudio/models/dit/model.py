from dataclasses import dataclass
from naudio.models.activations import GLU
import jax
from jax import lax
import jax.numpy as jnp
from beartype import beartype
from einops import rearrange
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
    context_dim: int
    timestep_dim: int = 256
    timestep_std: float = 1.0


def unstack(x, axis=0):
    return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


class FourierFeatures(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.weight = nnx.Linear(
            in_features=1,
            out_features=args.timestep_dim // 2,
            rngs=rngs,
            use_bias=False,
            dtype=dtype
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x: Float[Array, " 1"]) -> Float[Array, " timestep_dim"]:
        f = 2 * jnp.pi * self.weight(x)
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class FeedForward(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.linear_in = GLU(in_features=args.dim, out_features=args.dim * 4, rngs=rngs, dtype=dtype)
        self.linear_out = nnx.Linear(
            in_features=args.dim * 4, out_features=args.dim, use_bias=True, rngs=rngs, dtype=dtype
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x):
        x = self.linear_in(x)
        # silu act inside linear_in
        x = self.linear_out(x)
        return x


# TODO: make generic
class Attention(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(
        self, q_in_features, k_in_features, v_in_features, d_model, num_heads, rngs: nnx.Rngs, dtype=jnp.float32
    ) -> None:
        self.num_heads = num_heads
        self.to_q = nnx.Linear(in_features=q_in_features, out_features=d_model, rngs=rngs, dtype=dtype)
        self.to_k = nnx.Linear(in_features=k_in_features, out_features=d_model, rngs=rngs, dtype=dtype)
        self.to_v = nnx.Linear(in_features=v_in_features, out_features=d_model, rngs=rngs, dtype=dtype)

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, q, k, v):
        # projections
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # reshape
        q = rearrange(q, "tokens (heads head_dim) -> tokens heads head_dim", heads=self.num_heads)
        k = rearrange(k, "tokens (heads head_dim) -> tokens heads head_dim", heads=self.num_heads)
        v = rearrange(v, "tokens (heads head_dim) -> tokens heads head_dim", heads=self.num_heads)

        # TODO: rotary embeddings

        attn = nnx.dot_product_attention(q, k, v)
        attn = rearrange(attn, "tokens heads head_dim -> tokens (heads head_dim)", heads=self.num_heads)

        return attn


class TransformerBlock(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.pre_norm = nnx.LayerNorm(num_features=args.dim, rngs=rngs, dtype=dtype)
        self.ctx_norm = nnx.LayerNorm(num_features=args.context_dim, rngs=rngs, dtype=dtype)

        self.self_attn = Attention(
            q_in_features=args.dim,
            k_in_features=args.dim,
            v_in_features=args.dim,
            d_model=args.dim,
            num_heads=args.heads,
            rngs=rngs,
            dtype=dtype
        )
        self.ctx_attn = Attention(
            q_in_features=args.dim,
            k_in_features=args.context_dim,
            v_in_features=args.context_dim,
            d_model=args.dim,
            num_heads=args.heads,
            rngs=rngs,
            dtype=dtype
        )

        self.ff_norm = nnx.LayerNorm(num_features=args.dim, rngs=rngs, dtype=dtype)
        self.ff = FeedForward(args=args, rngs=rngs, dtype=dtype)

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x, ctx):
        # self attention
        pre_norm_x = self.pre_norm(x)
        x = x + self.self_attn(q=pre_norm_x, k=pre_norm_x, v=pre_norm_x)
        # cross attention
        ctx_norm_x = self.ctx_norm(ctx)
        x = x + self.ctx_attn(q=x, k=ctx_norm_x, v=ctx_norm_x)
        # feed forward norm & projection
        x = x + self.ff(self.ff_norm(x))

        return x


class DiT(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.args = args

        # patching conv
        self.proj_x_1 = nnx.Conv(
            in_features=args.channels,
            out_features=args.channels,
            kernel_size=1,
            strides=1,
            use_bias=True,
            rngs=rngs,
            dtype=dtype
        )
        self.proj_x_2 = nnx.Linear(in_features=args.channels, out_features=args.dim, use_bias=True, rngs=rngs, dtype=dtype)

        # timestep and projections
        self.to_timestep = FourierFeatures(args, rngs)
        self.time_proj_1 = nnx.Linear(
            in_features=args.timestep_dim,
            out_features=args.dim,
            use_bias=True,
            rngs=rngs,
            dtype=dtype
        )
        self.time_proj_2 = nnx.Linear(in_features=args.dim, out_features=args.dim, use_bias=True, rngs=rngs, dtype=dtype)

        # global cond projections
        self.global_proj_1 = nnx.Linear(
            in_features=args.dim, out_features=args.dim, use_bias=False, rngs=rngs, dtype=dtype
        )
        self.global_proj_2 = nnx.Linear(
            in_features=args.dim, out_features=args.dim, use_bias=False, rngs=rngs, dtype=dtype
        )

        # context projections
        self.context_proj_1 = nnx.Linear(
            in_features=args.context_dim,
            out_features=args.context_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype
        )
        self.context_proj_2 = nnx.Linear(
            in_features=args.context_dim,
            out_features=args.context_dim,
            use_bias=False,
            rngs=rngs,
            dtype=dtype
        )

        self.xf_layers = [TransformerBlock(args=args, rngs=rngs, dtype=dtype) for _ in range(args.depth)]
        self.xf_proj_out = nnx.Linear(
            in_features=args.dim,
            out_features=args.channels * args.patch,
            use_bias=False,
            rngs=rngs, 
            dtype=dtype
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(
        self,
        x: Float[Array, "x_dim x_chan"],
        t: Float[Array, " 1"],
        g: Float[Array, " global_dim"],
        ctx: Float[Array, "ctx_seq ctx_dim"],
    ):
        """
        The Stable Audio 1.0 DiT

        - Doesn't have any prepend cond but DOES apply global cond as a prepend.
        - Uses cross attention for the prompt
        """
        # project x, then transpose
        x = self.proj_x_1(x) + x
        x = self.proj_x_2(x)

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
        g = jnp.expand_dims(g + t, axis=0)

        # combine g and x
        x = jnp.concat([g, x], axis=0)

        # TODO: cached rotary embedding

        # oh yeah, attend it ;)
        for block in self.xf_layers:
            x = block(x=x, ctx=ctx)  # TODO: will also need rotary eventually

        # final projection
        x = self.xf_proj_out(x)

        return x


if __name__ == "__main__":
    print("starting init test")
    DMODEL = 256
    DEPTH = 2
    HEADS = 16
    PATCH = 1
    CHANNELS = 16
    CONTEXT_DIM = 128
    NTOKENS = 77
    args = ModelArgs(
        dim=DMODEL, depth=DEPTH, heads=HEADS, patch=PATCH, channels=CHANNELS, context_dim=CONTEXT_DIM
    )
    rngs = nnx.Rngs(0)
    model = DiT(args=args, rngs=rngs)

    key = jax.random.PRNGKey(0)

    t = jax.random.normal(key=key, shape=(1,))
    x = jax.random.normal(key=key, shape=(NTOKENS, CHANNELS))  # jax does N H C vs torch where its NCH
    ctx = jax.random.normal(key=key, shape=(NTOKENS, CONTEXT_DIM))
    g = jax.random.normal(key=key, shape=(DMODEL,))

    print(model(x=x, t=t, g=g, ctx=ctx).shape)
