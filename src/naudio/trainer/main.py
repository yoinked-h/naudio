import jax
import jax.numpy as jnp
from flax import nnx
from pathlib import Path
import json
import optax
from naudio.models import StableAudioOpen
from naudio.dataset import AudioDataset

def main(model_config_path: Path, dataset_config_path: Path, train_config_path: Path):
    tcfg = json.loads(train_config_path.read_text())
    if tcfg["model"] != "stable-audio-open":
        raise NotImplementedError('naudio trainer only supports stable-audio-open model')

    dataset_config = json.loads(dataset_config_path.read_text())
    dataset = AudioDataset(dataset_config)

    model = StableAudioOpen.from_config_file(model_config_path, rngseed=tcfg["seed"].encode('utf-8').hex())
    model.train() # set train mode
    optimizers = {}
    for mod in ['vae', 'text_encoder', 'dit']:
        modcfg = tcfg['training_hparams'][mod]
        match tcfg['training_hparams']["lr_scheduler"]["type"]:
            case "cosine":
                lr = optax.schedules.cosine_decay_schedule(modcfg["lr_scheduler"]["value"], modcfg["lr_scheduler"]["decay_steps"])
            case "constant":
                lr = optax.constant_schedule(modcfg["lr_scheduler"]["value"])
            case _:
                lr = optax.constant_schedule(0.0005)
        opt = modcfg["optimizer"]["type"]
        match opt:
            case "adam":
                optimizer = optax.adam(lr, b1=modcfg["optimizer"]['b1'], b2=modcfg["optimizer"]['b2'])
            case "adamw":
                optimizer = optax.adamw(lr, b1=modcfg["optimizer"]['b1'], b2=modcfg["optimizer"]['b2'])
            case "sgd":
                optimizer = optax.sgd(lr)
            case _:
                print(f"Invalid optimizer: {opt}, using adam")
                optimizer = optax.adam(lr)
        if mod == 'vae':
            optimizer = nnx.Optimizer(model.vae, optimizer)
        elif mod == 'text_encoder':
            optimizer = nnx.Optimizer(model.tenc, optimizer)
        elif mod == 'dit':
            optimizer = nnx.Optimizer(model.dit, optimizer)
        optimizers[mod] = optimizer
    def mse_loss(y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)
    
    @jax.jit
    def t5_train_step(state, batch):
        def loss_fn(model, inputs, target):
            output = model.tenc_encode(inputs)
            loss = mse_loss(target, output)
            return loss
        inp, tar = batch
        grads = nnx.grad(loss_fn)(state, inp, tar)
        state = optimizers['tenc'].update(grads)
        return state

    @jax.jit
    def dit_train_step(state, batch):
        def loss_fn(model, inputs, target):
            output = model.dit_encode(inputs)
            loss = mse_loss(target, output)
            return loss
        inp, tar = batch
        grads = nnx.grad(loss_fn)(state, inp, tar)
        state = optimizers['dit'].update(grads)
        return state
    
    @jax.jit
    def vae_train_step(state, batch):