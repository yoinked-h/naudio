import jax 
import jax.numpy as jnp
from functools import partial
from flax import nnx
from naudio.models.vae import AudioOobleckVae, VaeArgs
from naudio.models.discrims import EncodecDiscriminator
import optax
class Loss(nnx.Variable):pass

@jax.jit
def trainstep(vae: AudioOobleckVae, opt: nnx.Optimizer, x: jnp.ndarray, disc: EncodecDiscriminator, **kwargs):
    def gen_loss_fn(vae: AudioOobleckVae):
        logits = vae(x)
        losses = nnx.pop(vae, Loss)
        kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)
        reconstruction_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits, x)
        )
        loss = reconstruction_loss + 0.1 * kl_loss
        return loss
    def disc_fn(vae: AudioOobleckVae, disc: EncodecDiscriminator):
        