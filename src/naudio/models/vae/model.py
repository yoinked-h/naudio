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
        self.eps = 1e-8
    def __call__(self, x):
        a = self.alpha.value
        b = self.beta.value
        if self.alpha_logscale:
            a = jnp.exp(a)
            b = jnp.exp(b)
        b = 1 / (b + self.eps)
        x = x + b * jnp.square(jnp.sin(a * x)) / a
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
    def __call__(self,x: jnp.ndarray):
        #hijacking the image resize for audio, blegh
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * self.scale_factor, x.shape[2]), "nearest").astype(x.dtype) # blegh...
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
        self.layers = nnx.Sequential(*layers)
    def __call__(self, x):
        return self.layers(x)

class OobleckDecoder(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, out_features: int, channels: int, latent_dim: int, strides: tuple[int, ...], c_mults: tuple[int, ...], use_snake: bool, antialiasing:bool, use_nearest_upsample: bool, final_tanh: bool, rngs: nnx.Rngs) -> None:
        c_mults = (1,) + c_mults
        self.depth = len(c_mults)
        layers = [
            nnx.Conv(latent_dim, c_mults[-1]*channels, kernel_size=7, padding=3)
        ]
        for i in range(self.depth-1, 0, -1):
            layers += [
                DecoderBlock(c_mults[i]*channels,
                             out_channels=c_mults[i-1]*channels,
                             stride=strides[i-1],
                             use_snake=use_snake,
                             antialiasing=antialiasing,
                             use_nearest_upsample=use_nearest_upsample,
                             rngs=rngs
                             )
            ]
        layers += [
            Snake(c_mults[0] * channels),
            nnx.Conv(c_mults[0] * channels, out_features, kernel_size=7, padding=3, use_bias=False, rngs=rngs),
        ]
        self.layers = nnx.Sequential(*layers)
        self.use_tanh = final_tanh
    def __call__(self,x):
        x = self.layers(x)
        if self.use_tanh:
            x = nnx.tanh(x)
        return x

def vae_sample(mean, scale):
    stdev = nnx.softplus(scale) + 1e-4
    var = stdev * stdev
    lvar = jnp.log(var)
    latents = jax.random.normal(jax.random.PRNGKey(0x8888), mean.shape) * stdev + mean

    kl = (mean*mean + var - lvar - 1).sum(1).mean()

class VaeBottleneck(nnx.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self):
        self.is_discrete = False # does this even do anything?
    def __call__(self, x, return_info=False):
        mean = self.mean(x)
        scale = self.scale(x)
        latents, kl = vae_sample(mean, scale)
        if return_info:
            return latents, kl, mean, scale
        else:
            return latents
