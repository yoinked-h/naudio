import src.naudio.models.vae.model as vae
import jax.numpy as jnp 
import flax.nnx as nnx
rngs = nnx.Rngs(0x8888)
model = vae.OobleckEncoder(
            in_features=2, 
            channels=128, 
            latent_dim=128, 
            c_mults = (1, 2, 4, 8, 16), 
            strides = (2, 4, 4, 8, 8),
            use_snake=True,
            rngs=rngs
)

#test forward pass

x = jnp.ones((1, 2**16, 2))
print(x)
x = model(x)
print(x.shape)