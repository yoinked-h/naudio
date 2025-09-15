import jax 
import jax.numpy as jnp
from functools import partial
from flax import nnx
from naudio.models.vae import AudioOobleckVae, VaeArgs
from naudio.models.discrims import EncodecDiscriminator
import optax
class Loss(nnx.Variable):pass
@nnx.jit
def trainstep(vae: AudioOobleckVae, opt: nnx.Optimizer, x: jnp.ndarray):
    def gen_loss_fn(vae: AudioOobleckVae):
        logits = vae(x)
        losses = nnx.pop(vae, Loss)
        kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)
        reconstruction_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits, x)
        )
        loss = reconstruction_loss + 0.1 * kl_loss
        return loss
    
    loss, grads = nnx.value_and_grad(gen_loss_fn)(vae)
    opt.update(grads)
    return loss

if __name__ == "__main__":
    args = VaeArgs(
        features=2,
        channels=128,
        latent_dim=128,
        decoder_latent_dim=64,
        c_mults = (1, 2, 4, 8, 16),
        strides = (2, 4, 4, 8, 8),
        use_snake=True
    )
    with jax.default_device(jax.devices('cpu')[0]):
        dtype = jnp.float16
        vae = AudioOobleckVae(args, nnx.Rngs(0x7e57), dtype=dtype)
        opt = nnx.Optimizer(vae, optax.adamw(learning_rate=5e-4, b1=0.9, b2=0.999))
        epochs = 1000
        from naudio.dataset.dataset import PureAudioDataset
        import json, tqdm
        datasetpath = "./src/naudio/configs/datasets/test.json"
        dataset = PureAudioDataset(json.load(open(datasetpath)), snapto=2048, dtype=dtype) #48khz since that is div by 2048
        for epoch in tqdm.tqdm(range(epochs)):
            for x in dataset:
                loss = trainstep(vae, opt, x)
                print(f"epoch: {epoch}, loss: {loss}")
                print(f"epoch: {epoch}, loss: {loss}")
