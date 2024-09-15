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
    match tcfg['training_hparams']["lr_scheduler"]["type"]:
        case "cosine":
            lr = optax.schedules.cosine_decay_schedule(tcfg['training_hparams']["lr_scheduler"]["type"], tcfg['training_hparams']["lr_scheduler"]["decay_steps"])
        case "constant":
            lr = optax.constant_schedule(tcfg['training_hparams']["lr_scheduler"]["value"])
        case _:
            lr = optax.constant_schedule(0.0005)
    opt = tcfg['training_hparams']["optimizer"]["type"]
    match opt:
        case "adam":
            optimizer = optax.adam(lr, b1=tcfg['training_hparams']["optimizer"]['b1'], b2=tcfg['training_hparams']["optimizer"]['b2'])
        case "adamw":
            optimizer = optax.adamw(lr, b1=tcfg['training_hparams']["optimizer"]['b1'], b2=tcfg['training_hparams']["optimizer"]['b2'])
        case "sgd":
            optimizer = optax.sgd(lr)
        case _:
            print(f"Invalid optimizer: {opt}, using adam")
            optimizer = optax.adam(lr)
    optimizer = nnx.Optimizer(model, optimizer)
    # 2. Define a training step function
    def mse_loss(y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)
    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(params):
            aud, txt = batch
            logits = model(params, aud)
            loss = mse_loss(logits, txt)
            return loss

    # 3. Create a loop to iterate over epochs and batches
    num_epochs = tcfg["num_epochs"]
    batch_size = tcfg["batch_size"]

    for epoch in range(num_epochs):
        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset.get(batch_idx, batch_size)
            model.params, opt_state, loss = train_step(model.params, opt_state, batch)

            # 4. Add logging
            if batch_idx % tcfg["log_every"] == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss}")

        # Optionally, add validation step here

        # 5. Checkpointing
        if (epoch + 1) % tcfg["save_every"] == 0:
            
