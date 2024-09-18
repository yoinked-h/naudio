import jax
import jax.numpy as jnp
import optax
from flax import nnx
from naudio.models.vae import AudioOobleckVae, VaeArgs
from naudio.trainer.loss import SumAndDifferenceSTFTLoss, AltL1Loss, MultiResolutionSTFTLoss, MultiLoss, AuralossLoss, ValueLoss
from naudio.models.discrims import OobleckDiscriminator, EncodecDiscriminator
import orbax.checkpoint
import neptune
class TrainState(nnx.Optimizer):
    def __init__(self, model, tx, metrics):
        self.metrics = metrics
        super().__init__(model, tx)
    def update(self, *, grads, **updates): #type: ignore
        self.metrics.update(**updates)
        super().update(grads)


def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []
    try:
        kl_weight = loss_config['bottleneck']['weights']['kl']
    except:
        kl_weight = 1e-6
    kl_loss = ValueLoss(key='kl', weight=kl_weight)
    losses.append(kl_loss)
    return losses

class InverseLR:
    def __init__(self, lr, inv_gamma, power, final_lr=0.0):
        self.lr = lr
        self.inv_gamma = inv_gamma
        self.power = power 
        self.final_lr = final_lr
        self.last_epoch = 0

    def __call__(self, step):
        lr_mult = (1 + step / self.inv_gamma) ** -self.power
        return max(self.final_lr, self.lr * lr_mult)

class AudioVaeTrainer(nnx.Module):
    def __init__(
        self,
        ae: AudioOobleckVae,
        warmup_steps: int,
        sr: int = 44100,
        loss_cfg: dict|None = None,
        optimizer_cfg: dict|None = None,
        ema: bool = False,
        force_mono: bool = False,
        latent_mask_ratio: float = 0.0,
        teacher_model: AudioOobleckVae|None = None,
        neptune_project: str|None = None,
        rngs: nnx.Rngs|None = None
    ):
        assert rngs
        if neptune_project:
            self.neptunerun = neptune.init_run(project=neptune_project)
        self.automatic_optimization = False

        self.autoencoder = ae
        self.autoencoder.train()

        self.warmed_up = False
        self.warmup_steps = warmup_steps

        self.force_input_mono = force_mono

        self.teacher_model = teacher_model

        if optimizer_cfg is None:
            optimizer_cfg ={"autoencoder": {
                "optimizer": {
                    "type": "AdamW",
                    "config": {
                        "betas": [0.8, 0.99],
                        "lr": 1.5e-4,
                        "weight_decay": 1e-3
                    }
                },
                "scheduler": {
                    "type": "InverseLR",
                    "config": {
                        "inv_gamma": 200000,
                        "power": 0.5,
                        "warmup": 0.999
                    }
                }
            },
            "discriminator": {
                "optimizer": {
                    "type": "AdamW",
                    "config": {
                        "betas": [0.8, 0.99],
                        "lr": 3e-4,
                        "weight_decay": 1e-3
                    }
                },
                "scheduler": {
                    "type": "InverseLR",
                    "config": {
                        "inv_gamma": 200000,
                        "power": 0.5,
                        "warmup": 0.999
                    }
                }
            }
            } 
            
        self.optimizer_cfg = optimizer_cfg
        if loss_cfg is None:        
            loss_cfg = {
            "discriminator": {
                "type": "encodec",
                "config": {
                    "filters": 64,
                    "n_ffts": [2048, 1024, 512, 256, 128],
                    "hop_lengths": [512, 256, 128, 64, 32],
                    "win_lengths": [2048, 1024, 512, 256, 128]
                },
                "weights": {
                    "adversarial": 0.1,
                    "feature_matching": 5.0
                }
            },
            "spectral": {
                "type": "mrstft",
                "config": {
                    "fft_sizes": [2048, 1024, 512, 256, 128, 64, 32],
                    "hop_sizes": [512, 256, 128, 64, 32, 16, 8],
                    "win_lengths": [2048, 1024, 512, 256, 128, 64, 32],
                    "perceptual_weighting": True
                },
                "weights": {
                    "mrstft": 1.0
                }
            },
            "time": {
                "type": "l1",
                "config": {},
                "weights": {
                    "l1": 0.0,
                }
            }
            }
            
            

        self.loss_cfg = loss_cfg
        
        
        stft_loss_args = loss_cfg['spectral']['config']

        if self.autoencoder.audio_channels == 2:
            self.sdstft = SumAndDifferenceSTFTLoss(**stft_loss_args, sample_rate=sr, name="stereo_sum_and_difference")
            self.lrstft = MultiResolutionSTFTLoss(**stft_loss_args, sample_rate=sr, name="stereo_stft")
        else:
            self.sdstft = MultiResolutionSTFTLoss(**stft_loss_args, sample_rate=sr, name="mono_stft")

        # Discriminator

        if loss_cfg['discriminator']['type'] == 'oobleck':
            self.discriminator = OobleckDiscriminator(**loss_cfg['discriminator']['config'])#, rngs=rngs)
        elif loss_cfg['discriminator']['type'] == 'encodec':
            self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.audio_channels, **loss_cfg['discriminator']['config'], rngs=rngs)
        self.gen_loss_modules = []

        # Adversarial and feature matching losses
        self.gen_loss_modules += [
            ValueLoss(key='loss_adv', weight=self.loss_cfg['discriminator']['weights']['adversarial'], name="adv_loss"),
            ValueLoss(key='feature_matching_distance', weight=self.loss_cfg['discriminator']['weights']['feature_matching'], name="feature_matching_loss"),
        ]

        if self.teacher_model is not None:
            # Distillation losses

            stft_loss_weight = self.loss_cfg['spectral']['weights']['mrstft'] * 0.25
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', weight=stft_loss_weight, name="aura_gen_loss_own_to_own"), # Reconstruction loss
                AuralossLoss(self.sdstft, 'decoded', 'teacher_decoded', weight=stft_loss_weight, name="aura_gen_loss_teacher_to_teach"), # Distilled model's decoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'own_latents_teacher_decoded', weight=stft_loss_weight, name="aura_gen_loss_teacher_own_to_teach"), # Distilled model's encoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'teacher_latents_own_decoded', weight=stft_loss_weight, name="aura_gen_loss_teacher_teach_to_own") # Teacher's encoder is compatible with distilled model's decoder
            ]

        else:

            # Reconstruction loss
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', weight=self.loss_cfg['spectral']['weights']['mrstft']),
            ]

            if self.autoencoder.audio_channels == 2:

                # Add left and right channel reconstruction losses in addition to the sum and difference
                self.gen_loss_modules += [
                    AuralossLoss(self.lrstft, 'reals_left', 'decoded_left', weight=self.loss_cfg['spectral']['weights']['mrstft']/2, name="aura_gen_loss_left"),
                    AuralossLoss(self.lrstft, 'reals_right', 'decoded_right', weight=self.loss_cfg['spectral']['weights']['mrstft']/2, name="aura_gen_loss_right"),
                ]

            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', weight=self.loss_cfg['spectral']['weights']['mrstft'], name="aura_gen_mono_loss"),
            ]

        if self.loss_cfg['time']['weights']['l1'] > 0.0:
            self.gen_loss_modules.append(AltL1Loss(key_a='reals', key_b='decoded', weight=self.loss_cfg['time']['weights']['l1'], name="altl1_loss"))

        if self.autoencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_cfg)

        self.losses_gen = MultiLoss(self.gen_loss_modules, name="gen_loss")

        self.disc_loss_modules = [
            ValueLoss(key='loss_dis', weight=1.0, name="disc_loss"),
        ]

        self.losses_disc = MultiLoss(self.disc_loss_modules, name="m_disc_loss")

        # Set up EMA for model weights
        self.autoencoder_ema = None
        
        self.use_ema = ema

        if self.use_ema:
            raise NotImplementedError

        self.latent_mask_ratio = latent_mask_ratio
        self.global_step = 0
    def set_b(self, b):
        self.b = b
    
    def configure_optimizers(self):

        if "scheduler" in self.optimizer_cfg['autoencoder'] and "scheduler" in self.optimizer_cfg['discriminator']:
            match self.optimizer_cfg['autoencoder']['scheduler']['type'].lower():
                case 'cosine':
                    sched_gen = optax.cosine_decay_schedule(
                        init_value=self.optimizer_cfg['autoencoder']['optimizer']['config']['init_value'],
                        decay_steps=self.optimizer_cfg['autoencoder']['scheduler']['config']['decay_steps'])
                case 'polynomial':
                    sched_gen = optax.polynomial_schedule(
                        init_value=self.optimizer_cfg['autoencoder']['optimizer']['config']['init_value'],
                        end_value=self.optimizer_cfg['autoencoder']['scheduler']['config']['end_value'],
                        power=self.optimizer_cfg['autoencoder']['scheduler']['config']['power'],
                        transition_steps=self.optimizer_cfg['autoencoder']['scheduler']['config']['transition_steps'])
                case 'inverselr':
                    sched_gen = InverseLR(lr=self.optimizer_cfg['autoencoder']['optimizer']['config']['lr'], inv_gamma=self.optimizer_cfg['autoencoder']['scheduler']['config']['inv_gamma'],
                                        power=self.optimizer_cfg['autoencoder']['scheduler']['config']['power'])
                case _:
                    sched_gen = optax.constant_schedule(0.0005)
            match self.optimizer_cfg['discriminator']['scheduler']['type'].lower():
                case 'cosine':
                    sched_disc = optax.cosine_decay_schedule(
                        init_value=self.optimizer_cfg['discriminator']['optimizer']['config']['init_value'],
                        decay_steps=self.optimizer_cfg['discriminator']['scheduler']['config']['decay_steps'])
                case 'polynomial':
                    sched_disc = optax.polynomial_schedule(
                        init_value=self.optimizer_cfg['discriminator']['optimizer']['config']['init_value'],
                        end_value=self.optimizer_cfg['discriminator']['scheduler']['config']['end_value'],
                        power=self.optimizer_cfg['discriminator']['scheduler']['config']['power'],
                        transition_steps=self.optimizer_cfg['discriminator']['scheduler']['config']['transition_steps'])
                case 'inverselr':
                    sched_disc = InverseLR(lr=self.optimizer_cfg['discriminator']['optimizer']['config']['lr'], inv_gamma=self.optimizer_cfg['discriminator']['scheduler']['config']['inv_gamma'],
                                        power=self.optimizer_cfg['discriminator']['scheduler']['config']['power'])
                case _:
                    sched_disc = optax.constant_schedule(0.0005)
                    
        match self.optimizer_cfg['autoencoder']['optimizer']['type'].lower():
            case 'adam':
                opt_gen = optax.adam(sched_gen if 'scheduler' in self.optimizer_cfg['autoencoder'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['autoencoder']['optimizer']['config']['betas'][0],
                                    b2=self.optimizer_cfg['autoencoder']['optimizer']['config']['betas'][1])
            case 'adamw':
                opt_gen = optax.adamw(sched_gen if 'scheduler' in self.optimizer_cfg['autoencoder'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['autoencoder']['optimizer']['config']['betas'][0],
                                    b2=self.optimizer_cfg['autoencoder']['optimizer']['config']['betas'][1],
                                    weight_decay=self.optimizer_cfg['autoencoder']['optimizer']['config']['weight_decay'])
            case _:
                raise NotImplementedError

        match self.optimizer_cfg['discriminator']['optimizer']['type'].lower():
            case 'adam':
                opt_disc = optax.adam(sched_disc if 'scheduler' in self.optimizer_cfg['discriminator'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['discriminator']['optimizer']['config']['b1'],
                                    b2=self.optimizer_cfg['discriminator']['optimizer']['config']['b2'])
            case 'adamw':
                opt_disc = optax.adamw(sched_disc if 'scheduler' in self.optimizer_cfg['discriminator'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['discriminator']['optimizer']['config']['betas'][0],
                                    b2=self.optimizer_cfg['discriminator']['optimizer']['config']['betas'][1],
                                    weight_decay=self.optimizer_cfg['discriminator']['optimizer']['config']['weight_decay'])
            case _:
                raise NotImplementedError
        if 'scheduler' in self.optimizer_cfg['discriminator']:
            return [opt_gen, opt_disc], [sched_gen, sched_disc] # type: ignore
        return [opt_gen, opt_disc], None
    @nnx.jit
    def training_step(self, batch):
        self.global_step += 1
        reals = batch
        reals = jnp.array(reals, dtype=jnp.float32)
        reals = jnp.expand_dims(reals, axis=0) # (b, l, c)

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[2] > 1:
            encoder_input = encoder_input.mean(axis=2, keepdims=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        loss_info["latents"] = latents

        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if self.teacher_model is not None:
            #nograd
            teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
            loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = jax.random.uniform(jax.random.PRNGKey(self.global_step), latents.shape)
            mask = mask < self.latent_mask_ratio
            latents = jnp.where(mask, jnp.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)

        loss_info["decoded"] = decoded

        if self.autoencoder.audio_channels == 2:
            loss_info["decoded_left"] = decoded[:, :, 0:1]
            loss_info["decoded_right"] = decoded[:, :, 1:2]
            loss_info["reals_left"] = reals[:, :, 0:1]
            loss_info["reals_right"] = reals[:, :, 1:2]

        # Distillation
        if self.teacher_model is not None:
            teacher_decoded = self.teacher_model.decode(teacher_latents) # type: ignore
            own_latents_teacher_decoded = self.teacher_model.decode(latents) #Distilled model's latents decoded by teacher
            teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents) #Teacher's latents decoded by distilled model  # type: ignore
            loss_info['teacher_decoded'] = teacher_decoded
            loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
            loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if self.warmed_up:
            loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = jnp.array(0.)
            loss_adv = jnp.array(0.)
            feature_matching_distance = jnp.array(0.)

        loss_info["loss_dis"] = loss_dis
        loss_info["loss_adv"] = loss_adv
        loss_info["feature_matching_distance"] = feature_matching_distance


        sched_gen = None
        sched_disc = None

        if self.b is not None:
            sched_gen, sched_disc = self.b
        # Train the discriminator
        if self.global_step % 2 and self.warmed_up:
            loss, losses = self.losses_disc(loss_info)
            log_dict = {
                'train/loss_dis': loss_dis,
            }
            if sched_disc is not None:
                # sched step every step 
                sched_disc(self.global_step)

        # Train the generator 
        else:

            loss, losses = self.losses_gen(loss_info)



            if sched_gen is not None:
                # scheduler step every step
                sched_gen(self.global_step)
            if isinstance(latents, tuple):
                latents = latents[0]
            log_dict = {
                'train/loss': loss,
                'train/latent_std': latents.std().item(),
                'train/data_std': data_std
            }

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss, loss_info, self.global_step % 2 and self.warmed_up
    def bettertrain(self, batch, disc_state, gen_state):
        l, linfo, discturn = self.training_step(batch)
        if discturn:
            log_dict = {
                'train/disc_lr': disc_state.metrics.compute()
            }

            disc_grads = jax.grad(self.losses_disc)(disc_state, linfo)
            disc_state.update(grads=disc_grads)
        else:
            
            log_dict = {
                'train/gen_lr': disc_state.metrics.compute()
            }
            gen_grads = jax.grad(self.losses_gen)(gen_state, linfo)
            gen_state.update(grads=gen_grads)
        self.log_dict(log_dict, prog_bar=False, on_step=False)
    def log_dict(self, log_dict, prog_bar=False, on_step=False):
        if self.neptunerun is None:
            return
        for k, v in log_dict.items():
            self.neptunerun[k].append(v, step=self.global_step)
    def export_model(self, path, gen_state):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder
        opts = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3)
        serializer = orbax.checkpoint.CheckpointManager(
            path, options=opts
        )
        ckpt = {'model': model, 'data': [gen_state.opt_state]}
        serializer.save(self.global_step, ckpt)


if __name__ == "__main__":
    vargs = VaeArgs(
        features=2,
        channels=128,
        latent_dim=128,
        decoder_latent_dim=64,
        c_mults = (1, 2, 4, 8, 16),
        strides = (2, 4, 4, 8, 8),
        use_snake=True
    )
    rngs = nnx.Rngs(0x6589ffea)
    model = AudioOobleckVae(vargs, rngs)
    trainer = AudioVaeTrainer(
        model,
        125,
        neptune_project="yoinked/zunda0001",
        rngs=rngs
    )
    a, b = trainer.configure_optimizers()
    genst, discst = a
    trainer.set_b(b)
    epochs = 10
    from naudio.dataset.dataset import PureAudioDataset

    dataset = PureAudioDataset(
        {"audio_dir": "D:\\code\\zunset\\tod\\", "audio_ext": ".wav"}
    )

    for epoch in range(epochs):
        for data in dataset:
            print(data)
            trainer.bettertrain(data, genst, discst)
        print(f"Epoch {epoch} completed")