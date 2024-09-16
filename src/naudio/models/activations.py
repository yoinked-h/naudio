from jaxtyping import Array, Float, jaxtyped
from typing_extensions import Literal, Any
from beartype import beartype
import jax.numpy as jnp
import flax.nnx as nnx

TYPE_CHECKER = beartype

class ELU(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self) -> None:
        pass
    def __call__(self, x: Array) -> Array:
        return nnx.elu(x)

class SiLU(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self) -> None:
        pass
    def __call__(self, x: Array) -> Array:
        return nnx.silu(x)

class ReLU(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self) -> None:
        pass
    def __call__(self, x: Array) -> Array:
        return nnx.relu(x)
class LeakyReLU(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, alpha: float=0.005) -> None:
        self.alpha = alpha
    def __call__(self, x: Array) -> Array:
        return nnx.leaky_relu(x, negative_slope=self.alpha)

class Snake(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, alpha: float=1.0, alpha_logscale: bool=False) -> None:
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = nnx.Param(jnp.zeros(in_features) * alpha)
            self.beta = nnx.Param(jnp.zeros(in_features) * alpha)
        else:
            self.alpha = nnx.Param(jnp.ones(in_features) * alpha)
            self.beta = nnx.Param(jnp.ones(in_features) * alpha)
        self.eps = 1e-8
    def __call__(self, x: Array) -> Array:
        a = self.alpha.value
        b = self.beta.value
        if self.alpha_logscale:
            a = jnp.exp(a)
            b = jnp.exp(b)
        b = 1 / (b + self.eps)
        x = x + b * jnp.square(jnp.sin(a * x)) / a
        return x

def get_activation(activation: Literal["elu", "snake", "relu", "leaky_relu"]|str, channels: int, **kwargs) -> Any:
    if activation == "snake":
        return Snake(channels, alpha=1.0, alpha_logscale=True)
    elif activation == "elu":
        return ELU()
    elif activation == "relu":
        return ReLU() 
    elif activation == "leaky_relu":
        return LeakyReLU(**kwargs)
    else:
        raise ValueError(f"Activation {activation} not supported")


class GLU(nnx.Module):
    @jaxtyped(typechecker=TYPE_CHECKER)
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs, use_bias: bool = True, dtype=jnp.float32) -> None:
        self.proj = nnx.Linear(
            in_features=in_features,
            out_features=out_features * 2,
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype
        )

    @jaxtyped(typechecker=TYPE_CHECKER)
    def __call__(self, x: Float[Array, "tokens in_dim"]) -> Float[Array, "tokens out_dim"]:
        x, gate = jnp.split(self.proj(x), indices_or_sections=2, axis=-1)
        return x + nnx.silu(gate)
