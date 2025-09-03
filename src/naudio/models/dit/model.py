from dataclasses import dataclass
from naudio.models.activations import GLU
import jax
from jax import lax
import jax.numpy as jnp
from beartype import beartype
from einops import rearrange
from flax import nnx
from jaxtyping import Array, Float, jaxtyped
from safetensors.torch import save_file, load_file
import numpy as np
from pathlib import Path
from typing import Union

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
    # @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.weight = nnx.Linear(
            in_features=1,
            out_features=args.timestep_dim // 2,
            rngs=rngs,
            use_bias=False,
            dtype=dtype
        )

    # @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x: Float[Array, " 1"]) -> Float[Array, " timestep_dim"]:
        f = 2 * jnp.pi * self.weight(x)
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class FeedForward(nnx.Module):
    # @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.linear_in = GLU(in_features=args.dim, out_features=args.dim * 4, rngs=rngs, dtype=dtype)
        self.linear_out = nnx.Linear(
            in_features=args.dim * 4, out_features=args.dim, use_bias=True, rngs=rngs, dtype=dtype
        )

    # @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x):
        x = self.linear_in(x)
        # silu act inside linear_in
        x = self.linear_out(x)
        return x


# TODO: make generic
class Attention(nnx.Module):
    # @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(
        self, q_in_features, k_in_features, v_in_features, d_model, num_heads, rngs: nnx.Rngs, dtype=jnp.float32
    ) -> None:
        self.num_heads = num_heads
        self.to_q = nnx.Linear(in_features=q_in_features, out_features=d_model, rngs=rngs, dtype=dtype)
        self.to_k = nnx.Linear(in_features=k_in_features, out_features=d_model, rngs=rngs, dtype=dtype)
        self.to_v = nnx.Linear(in_features=v_in_features, out_features=d_model, rngs=rngs, dtype=dtype)

    # @jaxtyped(typechecker=TYPE_CHECKER)
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
    # @jaxtyped(typechecker=TYPE_CHECKER)
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

    # @jaxtyped(typechecker=TYPE_CHECKER)
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
    # @jaxtyped(typechecker=TYPE_CHECKER)
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

    # @jaxtyped(typechecker=TYPE_CHECKER)
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
    
    def save_safetensors(self, path: Union[str, Path]) -> None:
        """Save DiT model parameters to safetensors format."""
        path = Path(path)
        if path.suffix != '.safetensors':
            path = path.with_suffix('.safetensors')
        
        # Get the state dict
        state_dict = nnx.state(self)
        
        # Convert JAX arrays to numpy arrays for safetensors
        numpy_state = {}
        def _flatten_state(state, prefix=""):
            for key, value in state.items():
                full_key = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, jnp.ndarray):
                    numpy_state[full_key] = np.array(value)
                elif hasattr(value, 'items'):
                    _flatten_state(value, full_key)
        
        _flatten_state(state_dict)
        
        # Save using safetensors
        save_file(numpy_state, str(path))
    
    @classmethod
    def load_safetensors(cls, path: Union[str, Path], model_args: ModelArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> 'DiT':
        """Load DiT model parameters from safetensors format."""
        path = Path(path)
        if path.suffix != '.safetensors':
            path = path.with_suffix('.safetensors')
        
        # Load the safetensors file
        loaded_state = load_file(str(path))
        
        # Create a new model instance
        model = cls(model_args, rngs, dtype)
        
        # Get the model's state dict for comparison
        model_state = nnx.state(model)
        
        # Convert numpy arrays back to JAX arrays and reconstruct nested dict
        jax_state = {}
        loaded_keys = set()
        model_keys = set()
        
        for key_str, value in loaded_state.items():
            keys = key_str.split('.')
            current = jax_state
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = jnp.array(value)
            loaded_keys.add(key_str)
        
        # Flatten model state for comparison
        def flatten_state(state, prefix=""):
            keys = set()
            for key, value in state.items():
                
                full_key = f"{prefix}.{key}" if prefix else str(key)
                if type(value) is not nnx.VariableState:
                    # this isnt a key we should load
                    if hasattr(value, 'items'):
                        keys.update(flatten_state(value, full_key))
                    continue
                # check that the value isnt None
                if value.value is not None:
                    keys.add(full_key)
            return keys
        
        model_keys = flatten_state(model_state)
        
        # Log key differences
        missing_in_model = loaded_keys - model_keys
        missing_in_file = model_keys - loaded_keys
        
        if missing_in_model:
            print(f"Warning: {len(missing_in_model)} keys in safetensors file not found in DiT model:")
            for key in sorted(missing_in_model):
                print(f"  - {key}")
        
        if missing_in_file:
            print(f"Warning: {len(missing_in_file)} keys in DiT model not found in safetensors file:")
            for key in sorted(missing_in_file):
                print(f"  - {key}")
        
        if not missing_in_model and not missing_in_file:
            print(f"✓ All {len(loaded_keys)} DiT keys matched successfully")
        
        # Create a State object and update the model
        state = nnx.State(jax_state)
        nnx.update(model, state)
        
        return model


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
