from dataclasses import dataclass
from naudio.models.activations import get_activation
import jax
import math
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, jaxtyped

TYPE_CHECKER = beartype

def torchfix(n, k, d=1):
    # return n # a
    return d*(k-1)-n

class Loss(nnx.Variable): pass

class ResidualUnit(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, dilation: int, use_snake: bool, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.dilation = dilation
        padding = (dilation * (7-1)) // 2
        self.layers = [
        get_activation('snake' if use_snake else 'elu', out_features, dtype=dtype),
        nnx.Conv(
            in_features=in_features, out_features=out_features, kernel_size=7, padding=padding, kernel_dilation=dilation, rngs=rngs, dtype=dtype
        ),
        get_activation('snake' if use_snake else 'elu', out_features, dtype=dtype),
        nnx.Conv(
            in_features=out_features, out_features=out_features, kernel_size=1, rngs=rngs, dtype=dtype
        ),
        ]
    # @jaxtyped(typechecker=beartype)
    def __call__(self, x):
        res = x
        for n in self.layers:
            res = n(res)
        return x + res

class EncoderBlock(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, stride: int, use_snake: bool, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        self.layers = nnx.Sequential(
            ResidualUnit(in_features, in_features, 1, use_snake, rngs, dtype=dtype),
            ResidualUnit(in_features, in_features, 3, use_snake, rngs, dtype=dtype),
            ResidualUnit(in_features, in_features, 9, use_snake, rngs, dtype=dtype),
            get_activation('snake' if use_snake else 'elu', in_features, dtype=dtype),
            nnx.Conv(
            in_features=in_features, out_features=out_features, kernel_size=2*stride, strides=[stride], padding=math.ceil(stride/2), rngs=rngs, dtype=dtype
        ))
        self.out_features = out_features
    # @jaxtyped(typechecker=beartype)
    def __call__(self, x):
        a = self.layers(x)
        return a

class NNUpsampler(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, stride:int, rngs:nnx.Rngs, dtype=jnp.float32) -> None:
        self.scale_factor = stride
        self.conv = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=2*stride,strides=[1],use_bias=False, rngs=rngs, dtype=dtype)
    # @jaxtyped(typechecker=beartype)
    def __call__(self,x: jnp.ndarray):
        #hijacking the image resize for audio, blegh
        x = jax.image.resize(x, (x.shape[0], x.shape[1] * self.scale_factor, x.shape[2]), "nearest").astype(x.dtype) # blegh...
        x = self.conv(x)
        return x
class DecoderBlock(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, in_features: int, out_features: int, stride: int, use_snake: bool, use_nearest_neighbor: bool, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        if use_nearest_neighbor:
            upsampler = NNUpsampler(in_features, out_features, stride, rngs, dtype=dtype)
        else:
            upsampler = nnx.ConvTranspose(in_features=in_features, out_features=out_features, kernel_size=2*stride, strides=stride, padding=torchfix(math.ceil(stride/2),2*stride), rngs=rngs, dtype=dtype)
        self.layers = nnx.Sequential(
            get_activation('snake' if use_snake else 'elu', in_features, dtype=dtype),
            upsampler,
            ResidualUnit(out_features, out_features, 1, use_snake, rngs, dtype=dtype),
            ResidualUnit(out_features, out_features, 3, use_snake, rngs, dtype=dtype),
            ResidualUnit(out_features, out_features, 9, use_snake, rngs, dtype=dtype),
        )
    # @jaxtyped(typechecker=beartype)
    def __call__(self, x):
        return self.layers(x)

@dataclass
class EncoderArgs:
    in_features: int
    channels: int
    latent_dim: int
    c_mults: tuple[int, ...]
    strides: tuple[int, ...]
    use_snake: bool = True

class OobleckEncoder(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, EncoderArgs: EncoderArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        c_mults = (1,) + EncoderArgs.c_mults
        self.depth = len(c_mults)
        layers = [
            nnx.Conv(in_features=EncoderArgs.in_features, out_features=c_mults[0]*EncoderArgs.channels, kernel_size=7, padding=3, rngs=rngs, dtype=dtype)
        ] 
        for i in range(self.depth-1):
            layers += [
                EncoderBlock(c_mults[i] * EncoderArgs.channels, c_mults[i + 1] * EncoderArgs.channels, EncoderArgs.strides[i], EncoderArgs.use_snake, rngs, dtype=dtype)
            ]
        layers += [
            get_activation('snake' if EncoderArgs.use_snake else 'elu', c_mults[-1] * EncoderArgs.channels, dtype=dtype),
            nnx.Conv(in_features=c_mults[-1] * EncoderArgs.channels, out_features=EncoderArgs.latent_dim, kernel_size=3, padding=1, rngs=rngs, dtype=dtype)
        ]
        self.layers = nnx.Sequential(*layers)
    # @jaxtyped(typechecker=beartype)
    def __call__(self, x):
        return self.layers(x)

@dataclass
class DecoderArgs:
    out_features: int
    channels: int
    latent_dim: int
    strides: tuple[int, ...]
    c_mults: tuple[int, ...]
    use_snake: bool = True
    use_nearest_neighbor: bool = False
    final_tanh: bool = False

class OobleckDecoder(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, DecoderArgs: DecoderArgs, rngs: nnx.Rngs, dtype=jnp.float32) -> None:
        c_mults = (1,) + DecoderArgs.c_mults
        self.depth = len(c_mults)
        layers = [
            nnx.Conv(DecoderArgs.latent_dim, c_mults[-1]*DecoderArgs.channels, kernel_size=7, padding=3, rngs=rngs, dtype=dtype)
        ]
        for i in range(self.depth-1, 0, -1):
            layers += [
                DecoderBlock(c_mults[i]*DecoderArgs.channels,
                    out_features=c_mults[i-1]*DecoderArgs.channels,
                    stride=DecoderArgs.strides[i-1],
                    use_snake=DecoderArgs.use_snake,
                    use_nearest_neighbor=DecoderArgs.use_nearest_neighbor,
                    rngs=rngs,
                    dtype=dtype
                )
            ]
        layers += [
            get_activation('snake' if DecoderArgs.use_snake else 'elu', c_mults[0] * DecoderArgs.channels, dtype=dtype),
            nnx.Conv(c_mults[0] * DecoderArgs.channels, DecoderArgs.out_features, kernel_size=7, padding=3, use_bias=False, rngs=rngs, dtype=dtype),
        ]
        self.layers = nnx.Sequential(*layers)
        self.use_tanh = DecoderArgs.final_tanh
    # @jaxtyped(typechecker=beartype)
    def __call__(self,x):
        x = self.layers(x)
        if self.use_tanh:
            x = nnx.tanh(x)
        return x

def vae_sample(mean, scale):
    stdev = nnx.softplus(scale) + 1e-4
    var = stdev * stdev
    lvar = jnp.log(var)
    latents = jax.random.normal(jax.random.PRNGKey(0x4361), mean.shape, dtype=mean.dtype) * stdev + mean # TODO: fix random seed thing eventually
    kl = (mean*mean + var - lvar - 1).sum(1).mean()
    return latents, kl

class VaeBottleneck(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self):
        self.is_discrete = False # does this even do anything?
    def encode(self, x: Array,return_info: bool = False) -> Array:
        info = {}
        mean, scale = jnp.split(x, 2, axis=-1)
        x, kl = vae_sample(mean, scale)
        info['kl'] = kl
        self.kl_loss = Loss(kl)
        if return_info: 
            return x, info # type: ignore
        else:
            # x = rearrange(x, 'b c t -> b t c')
            return x

@dataclass
class VaeArgs:
    features: int
    channels: int
    latent_dim: int
    decoder_latent_dim: int
    c_mults: tuple[int, ...]
    strides: tuple[int, ...]
    use_snake: bool = True
    use_nearest_neighbor: bool = False
    final_tanh: bool = False
    
class AudioOobleckVae(nnx.Module):
    # @jaxtyped(typechecker=beartype)
    def __init__(self, VaeArgs: VaeArgs, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32) -> None:
        encargs = EncoderArgs(
            in_features=VaeArgs.features,
            channels=VaeArgs.channels,
            latent_dim=VaeArgs.latent_dim,
            c_mults=VaeArgs.c_mults,
            strides=VaeArgs.strides,
            use_snake=VaeArgs.use_snake
        )
        decargs = DecoderArgs(
            out_features=VaeArgs.features,
            channels=VaeArgs.channels,
            latent_dim=VaeArgs.decoder_latent_dim,
            c_mults=VaeArgs.c_mults,
            strides=VaeArgs.strides,
            use_snake=VaeArgs.use_snake,
            use_nearest_neighbor=VaeArgs.use_nearest_neighbor,
            final_tanh=VaeArgs.final_tanh
        )
        self.encoder = OobleckEncoder(encargs, rngs=rngs, dtype=dtype)
        self.decoder = OobleckDecoder(decargs, rngs=rngs, dtype=dtype)
        self.bottleneck = VaeBottleneck()
        self.audio_channels = VaeArgs.features
    # @jaxtyped(typechecker=beartype)
    def encode(self, x, return_info: bool = False) -> Array:
        x = self.bottleneck.encode(self.encoder(x), return_info=return_info) # this is Bad:tm:
        # x = self.encoder(x)
        return x
    def __call__(self, x):
        z = self.encode(x)
        logit = self.decoder(z)
        logit = jnp.reshape(logit, (-1, self.audio_channels))
        return logit
    # @jaxtyped(typechecker=beartype)
    def decode(self, x) -> Array:
        return self.decoder(x)

if __name__ == '__main__':
    with jax.default_device(jax.devices('cpu')[0]):
        print('testing audio oobleck vae')
        args = VaeArgs(
            features=2,
            channels=128,
            latent_dim=128,
            decoder_latent_dim=64,
            c_mults = (1, 2, 4, 8, 16),
            strides = (2, 4, 4, 8, 8),
            use_snake=True
        )
        rngs = nnx.Rngs(0x7e57)
        model = AudioOobleckVae(args, rngs, dtype=jnp.bfloat16)

        x = jnp.ones((1318912, 2)).astype('bfloat16') # Batch, Samples/Length, Channels
        print(x.shape)
        enc = model.encode(x)
        print(enc.shape)
        dec = model.decode(enc)
        print(dec.shape)