import jax.numpy as jnp
from flax import nnx
from einops import rearrange
from vae.model import AudioOobleckVae, VaeArgs
from dit.model import DiT, ModelArgs

vaeargs = VaeArgs(
    features=2,
    channels=128,
    latent_dim=128,
    decoder_latent_dim=64,
    c_mults = (1, 2, 4, 8, 16),
    strides = (2, 4, 4, 8, 8),
    use_snake=True
)

modelargs = ModelArgs(
    dim=1536,
    depth=24,
    heads=24,
    patch=1,
    channels=64,
    context_dim=768,
)

rngs = nnx.Rngs(0x7e57)

vae = AudioOobleckVae(vaeargs, rngs)
dit = DiT(modelargs, rngs)


raw = jnp.ones((2**16, 2)) # Samples/Length, Channels
print("Input shape: ", raw.shape)
vaeenc = vae.encode(raw)
print("Internal Latent Shape: ", vaeenc.shape)
# exit()

timestep_embed = jnp.ones((1))
context = jnp.ones((vaeenc.shape[0], modelargs.context_dim))
global_conditioning = jnp.ones((modelargs.dim,))

x = dit(x=vaeenc, g=global_conditioning, t=timestep_embed, ctx=context)

vaedec = vae.decode(x)
print("Ouput shape: ", vaedec.shape)
vaedec = vae.decode(vaeenc)
print("Vaeloop shape: ", vaedec.shape)