import jax
import jax.numpy as jnp
import optax
from flax import nnx
from models.vae import AudioOobleckVae, VaeBottleneck
from .loss import SumAndDifferenceSTFTLoss, AltL1Loss, MultiResolutionSTFTLoss, MultiLoss, AuralossLoss, ValueLoss
from models.discrims import OobleckDiscriminator, EncodecDiscriminator

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

class AudioVaeTrainer():
    def __init__(
        self,
        ae: AudioOobleckVae,
        lr: float,
        warmup_steps: int,
        enc_freeze_on_warmup: bool,
        sr: int = 44100,
        loss_cfg: dict|None = None,
        optimizer_cfg: dict|None = None,
        ema: bool = False,
        ema_copy: bool = False,
        force_mono: bool = False,
        latent_mask_ratio: float = 0.0,
        teacher_model: AudioOobleckVae|None = None
    ):
        
        self.automatic_optimization = False

        self.autoencoder = ae

        self.warmed_up = False
        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = enc_freeze_on_warmup
        self.lr = lr

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
            loss_cfg = {"discriminator": {
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
            }}

        self.loss_cfg = loss_cfg
        
        
        stft_loss_args = loss_cfg['spectral']['config']

        if self.autoencoder.audio_channels == 2:
            self.sdstft = SumAndDifferenceSTFTLoss(**stft_loss_args, sample_rate=sr, name="stereo_sum_and_difference")
            self.lrstft = MultiResolutionSTFTLoss(**stft_loss_args, sample_rate=sr, name="stereo_stft")
        else:
            self.sdstft = MultiResolutionSTFTLoss(**stft_loss_args, sample_rate=sr, name="mono_stft")

        # Discriminator

        if loss_cfg['discriminator']['type'] == 'oobleck':
            self.discriminator = OobleckDiscriminator(**loss_cfg['discriminator']['config'])
        elif loss_cfg['discriminator']['type'] == 'encodec':
            self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.audio_channels, **loss_cfg['discriminator']['config'])
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
        self.optimizers = self.configure_optimizers()
    def configure_optimizers(self):

        if "scheduler" in self.optimizer_cfg['autoencoder'] and "scheduler" in self.optimizer_cfg['discriminator']:
            match self.optimizer_cfg['autoencoder']['scheduler']['type'].lower():
                case 'cosine':
                    sched_gen = optax.cosine_decay_schedule(
                        init_value=self.optimizer_cfg['autoencoder']['optimizer']['init_value'],
                        decay_steps=self.optimizer_cfg['autoencoder']['scheduler']['decay_steps'])
                case 'polynomial':
                    sched_gen = optax.polynomial_schedule(
                        init_value=self.optimizer_cfg['autoencoder']['optimizer']['init_value'],
                        end_value=self.optimizer_cfg['autoencoder']['scheduler']['end_value'],
                        power=self.optimizer_cfg['autoencoder']['scheduler']['power'],
                        transition_steps=self.optimizer_cfg['autoencoder']['scheduler']['transition_steps'])
                case 'inverselr':
                    sched_gen = InverseLR(lr=self.optimizer_cfg['autoencoder']['optimizer']['lr'], inv_gamma=self.optimizer_cfg['autoencoder']['scheduler']['inv_gamma'],
                                        power=self.optimizer_cfg['autoencoder']['scheduler']['power'])
                case _:
                    sched_gen = optax.constant_schedule(0.0005)
            match self.optimizer_cfg['discriminator']['scheduler']['type'].lower():
                case 'cosine':
                    sched_disc = optax.cosine_decay_schedule(
                        init_value=self.optimizer_cfg['discriminator']['optimizer']['init_value'],
                        decay_steps=self.optimizer_cfg['discriminator']['scheduler']['decay_steps'])
                case 'polynomial':
                    sched_disc = optax.polynomial_schedule(
                        init_value=self.optimizer_cfg['discriminator']['optimizer']['init_value'],
                        end_value=self.optimizer_cfg['discriminator']['scheduler']['end_value'],
                        power=self.optimizer_cfg['discriminator']['scheduler']['power'],
                        transition_steps=self.optimizer_cfg['discriminator']['scheduler']['transition_steps'])
                case 'inverselr':
                    sched_disc = InverseLR(lr=self.optimizer_cfg['discriminator']['optimizer']['lr'], inv_gamma=self.optimizer_cfg['discriminator']['scheduler']['inv_gamma'],
                                        power=self.optimizer_cfg['discriminator']['scheduler']['power'])
                case _:
                    sched_disc = optax.constant_schedule(0.0005)
                    
        match self.optimizer_cfg['autoencoder']['optimizer']['type'].lower():
            case 'adam':
                opt_gen = optax.adam(sched_gen if 'scheduler' in self.optimizer_cfg['autoencoder'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['autoencoder']['optimizer']['b1'],
                                    b2=self.optimizer_cfg['autoencoder']['optimizer']['b2'])
            case 'adamw':
                opt_gen = optax.adamw(sched_gen if 'scheduler' in self.optimizer_cfg['autoencoder'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['autoencoder']['optimizer']['b1'],
                                    b2=self.optimizer_cfg['autoencoder']['optimizer']['b2'],
                                    weight_decay=self.optimizer_cfg['autoencoder']['optimizer']['weight_decay'])
            case _:
                raise NotImplementedError

        match self.optimizer_cfg['discriminator']['optimizer']['type'].lower():
            case 'adam':
                opt_disc = optax.adam(sched_disc if 'scheduler' in self.optimizer_cfg['discriminator'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['discriminator']['optimizer']['b1'],
                                    b2=self.optimizer_cfg['discriminator']['optimizer']['b2'])
            case 'adamw':
                opt_disc = optax.adamw(sched_disc if 'scheduler' in self.optimizer_cfg['discriminator'] else 0.0005, # type: ignore
                                    b1=self.optimizer_cfg['discriminator']['optimizer']['b1'],
                                    b2=self.optimizer_cfg['discriminator']['optimizer']['b2'],
                                    weight_decay=self.optimizer_cfg['discriminator']['optimizer']['weight_decay'])
            case _:
                raise NotImplementedError
        if 'scheduler' in self.optimizer_cfg['discriminator']:
            return [opt_gen, opt_disc], [sched_gen, sched_disc] # type: ignore
        return [opt_gen, opt_disc] 
    def training_step(self, batch, batch_idx):
        reals, _ = batch

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        if self.global_step >= self.warmup_steps:
            self.warmed_up = True

        loss_info = {}

        loss_info["reals"] = reals

        encoder_input = reals

        if self.force_input_mono and encoder_input.shape[1] > 1:
            encoder_input = encoder_input.mean(dim=1, keepdim=True)

        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        if self.warmed_up and self.encoder_freeze_on_warmup:
            #nograd
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)
        else:
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
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if self.teacher_model is not None:
            #noqa
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

        opt_gen, opt_disc = self.optimizers[0]

        lr_schedulers = self.optimizers[1]

        sched_gen = None
        sched_disc = None

        if lr_schedulers is not None:
            sched_gen, sched_disc = lr_schedulers

        # Train the discriminator
        if self.global_step % 2 and self.warmed_up:
            loss, losses = self.losses_disc(loss_info)

            log_dict = {
                'train/disc_lr': 
            }

            opt_disc.zero_grad()
            self.manual_backward(loss)
            opt_disc.step()

            if sched_disc is not None:
                # sched step every step
                sched_disc.step()

        # Train the generator 
        else:

            loss, losses = self.losses_gen(loss_info)

            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()

            if sched_gen is not None:
                # scheduler step every step
                sched_gen.step()

            log_dict = {
                'train/loss': loss.detach(),
                'train/latent_std': latents.std().detach(),
                'train/data_std': data_std.detach(),
                'train/gen_lr': opt_gen.param_groups[0]['lr']
            }

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss
    
    def export_model(self, path, use_safetensors=False):
        if self.autoencoder_ema is not None:
            model = self.autoencoder_ema.ema_model
        else:
            model = self.autoencoder
            
        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)
        
