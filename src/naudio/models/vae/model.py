from dataclasses import dataclass

import jax, math
from jax import lax
import jax.numpy as jnp
from beartype import beartype
from einops import rearrange
from flax import nnx
from jaxtyping import Array, Float, jaxtyped

TYPE_CHECKER = beartype

@dataclass
class EncoderArgs:
    in_dim: int
    dim: int
    c_mults: tuple[int, ...]
    strides: tuple[int, ...]
    latent_dim: int
    use_snake: bool = True
    
class Snake(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, alpha: float=1.0, alpha_logscale: bool=False, rngs: nnx.Rngs|None=None) -> None:
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = nnx.Param(jnp.zeros(in_features) * alpha)
            self.beta = nnx.Param(jnp.zeros(in_features) * alpha)
        else:
            self.alpha = nnx.Param(jnp.ones(in_features) * alpha)
            self.beta = nnx.Param(jnp.ones(in_features) * alpha)
    def __call__(self, x):
        a = jnp.expand_dims(self.alpha.value, 0)
        a = jnp.expand_dims(a, -1)
        b = jnp.expand_dims(self.beta.value, 0)
        b = jnp.expand_dims(b, -1)
        if self.alpha_logscale:
            a = jnp.exp(a)
            b = jnp.exp(b)
        x = (x + (1.0 / (b + 0.000000001)) * pow(jnp.sin(x * a), 2))
        return x

class ResidualUnit(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, dilation: int, use_snake: bool, rngs: nnx.Rngs) -> None:
        self.dilation = dilation
        if not use_snake:
            raise NotImplementedError("non-snake activation isnt implemented yet")
        padding = (dilation * (7-1)) // 2
        self.layers = [
        Snake(out_features),
        nnx.Conv(
            in_features=in_features, out_features=out_features, kernel_size=7, padding=padding, kernel_dilation=dilation, rngs=rngs
        ),
        Snake(out_features),
        nnx.Conv(
            in_features=out_features, out_features=out_features, kernel_size=1, rngs=rngs
        ),
        ]
    def __call__(self, x):
        res = x
        for n in self.layers:
            res = n(res)
        return x + res

class EncoderBlock(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, stride: int, use_snake: bool, rngs: nnx.Rngs) -> None:
        self.layers = nnx.Sequential(
            ResidualUnit(in_features, in_features, 1, use_snake, rngs),
            ResidualUnit(in_features, in_features, 3, use_snake, rngs),
            ResidualUnit(in_features, in_features, 9, use_snake, rngs),
            Snake(in_features),
            nnx.Conv(
            in_features=in_features, out_features=out_features, kernel_size=2*stride, strides=[stride], padding=math.ceil(stride/2), rngs=rngs
        ))
        self.out_features = out_features
    def __call__(self, x):
        a = self.layers(x)
        return a
@dataclass
class DecoderArgs:
    in_dim: int
    dim: int
    c_mults: tuple[int, ...]
    strides: tuple[int, ...]
    latent_dim: int
    use_snake: bool = True

class NNUpsampler(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, stride:int, rngs:nnx.Rngs):
        self.scale_factor = stride
        self.conv = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=2*stride,strides=[1],use_bias=False, rngs=rngs)
    def __call__(self,x):
        #hijacking the image resize for audio, blegh
        # [batch, channels, length]
        x = jax.image.resize(x, (x.shape[0], x.shape[1], x.shape[2] * self.scale_factor), "nearest").astype(x.dtype) # blegh...
        x = self.conv(x)
        return x
class DecoderBlock(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, stride: int, use_snake: bool, antialiasing: bool, use_nearest_neighbor: bool, rngs: nnx.Rngs) -> None:
        if use_nearest_neighbor:
            upsampler = NNUpsampler(in_features, out_features, stride, rngs)
        else:
            upsampler = nnx.ConvTranspose(in_features=in_features, out_features=out_features, kernel_size=2*stride, strides=[stride], padding=math.ceil(stride/2), rngs=rngs)
        self.layers = nnx.Sequential(
            Snake(in_features),
            upsampler,
            ResidualUnit(out_features, out_features, 1, use_snake, rngs),
            ResidualUnit(out_features, out_features, 3, use_snake, rngs),
            ResidualUnit(out_features, out_features, 9, use_snake, rngs),
        )
    def __call__(self, x):
        return self.layers(x)

class OobleckEncoder(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, channels: int, latent_dim: int, strides: tuple[int, ...], c_mults: tuple[int, ...], use_snake: bool, rngs: nnx.Rngs) -> None:
        c_mults = (1,) + c_mults
        self.depth = len(c_mults)
        layers = [
            nnx.Conv(in_features=in_features, out_features=c_mults[0]*channels, kernel_size=7, padding=3, rngs=rngs)
        ] 
        for i in range(self.depth-1):
            layers += [
                EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels, strides[i], use_snake, rngs)
            ]
        layers += [
            Snake(c_mults[-1] * channels),
            nnx.Conv(in_features=c_mults[-1] * channels, out_features=latent_dim, kernel_size=3, padding=1, rngs=rngs)
        ]
        # self.layers = layers
        self.layers = nnx.Sequential(*layers)
    def __call__(self, x):
        return self.layers(x)



