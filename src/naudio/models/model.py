from vae.model import AudioOobleckVae, VaeArgs
from dit.model import DiT, ModelArgs
from t5.model import T5, T5Config
from t5.tokenizer import T5Tokenizer

import jax.numpy as jnp
from pathlib import Path
from json import loads
from flax import nnx
from dataclasses import dataclass
DTYPE_DEALIAS = {'fp32': jnp.float32, 'fp16': jnp.float16, 'bf16': jnp.bfloat16,
                'fp8_e4m3': jnp.float8_e5m2, 'fp8_e5m2': jnp.float8_e5m2} #i have not tested fp8 yet

@dataclass
class StableAudioOpenConfig:
    latent_dim: int
    
    vae_dtype: str
    audio_channels: int
    vae_channels: int
    encoder_latent_dim: int
    vae_c_mults: tuple[int, ...]
    vae_strides: tuple[int, ...]
    
    dit_dtype: str
    dit_dim: int
    dit_depth: int
    dit_heads: int
    dit_patch: int
    dit_channels: int

    tenc_dtype: str
    tenc_model_dim: int
    tenc_ff_dim: int
    tenc_attn_heads: int



    use_snake: bool = True
    use_nearest_neighbor: bool = False
    final_tanh: bool = False
    
    dit_timestep_dim: int = 256
    dit_timestep_std: float = 1.0

    tenc_layer_norm_epsilon: float = 1e-6
    tenc_num_layers: int = 6
    tenc_vocab_size: int = 32128
    tenc_relative_attention_num_buckets: int = 32
    tenc_relative_attention_max_distance: int = 128
class StableAudioOpen(nnx.Module):
    def __init__(self, config: StableAudioOpenConfig, rngs:nnx.Rngs):
        self.vae = AudioOobleckVae(VaeArgs(
            features=config.audio_channels,
            channels=config.encoder_latent_dim,
            latent_dim=config.latent_dim,
            decoder_latent_dim=config.latent_dim,
            c_mults=tuple(config.vae_c_mults),
            strides=tuple(config.vae_strides),
            use_snake=config.use_snake,
            use_nearest_neighbor=config.use_nearest_neighbor,
            final_tanh=config.final_tanh
        ), rngs=rngs)
        self.dit = DiT(ModelArgs(
            dim=config.dit_dim,
            depth=config.dit_depth,
            heads=config.dit_heads,
            patch=config.dit_patch,
            channels=config.latent_dim,
            context_dim=config.tenc_model_dim
        ), rngs=rngs)
        self.tenc = T5(T5Config(
            d_model=config.tenc_model_dim,
            d_ff=config.tenc_ff_dim,
            num_heads=config.tenc_attn_heads,
            layer_norm_epsilon=config.tenc_layer_norm_epsilon,
            num_layers=config.tenc_num_layers,
            vocab_size=config.tenc_vocab_size,
            relative_attention_num_buckets=config.tenc_relative_attention_num_buckets,
            relative_attention_max_distance=config.tenc_relative_attention_max_distance
        ), rngs=rngs)
        self.tokenizer = T5Tokenizer()
    
    def vae_encode(self, x):
        return self.vae.encode(x)
    def vae_decode(self, x):
        return self.vae.decode(x)
    def _tenc_encode(self, x):
        return self.tenc(x)
    def tenc_encode(self, x):
        return self._tenc_encode(self.tokenizer(x))
    def dit_call(self, latent, timestep, context, global_cond):
        return self.dit(latent, timestep, context, global_cond)
    @classmethod
    def from_config_file(cls, pathtoconfig:Path, rngseed:int|None=None):
        if rngseed is not None:
            rngs = nnx.Rngs(rngseed)
        else:
            rngs = nnx.Rngs(0x00000000)
        d = loads(pathtoconfig.read_text())
        return cls(StableAudioOpenConfig(
            latent_dim=d['latent_dim'],
            vae_dtype=d['vae']['dtype'],
            audio_channels=d['audio_channels'],
            vae_channels=d['vae']['channels'],
            encoder_latent_dim=d['vae']['encoder_latent_dim'],
            vae_c_mults=tuple(d['vae']['c_mults']),
            vae_strides=tuple(d['vae']['strides']),
            dit_dtype=d['dit']['dtype'],
            dit_dim=d['dit']['model_dim'],
            dit_depth=d['dit']['depth'],
            dit_heads=d['dit']['heads'],
            dit_patch=d['dit']['patch'],
            dit_channels=d['vae']['channels'],
            tenc_dtype=d['text_encoder']['dtype'],
            tenc_model_dim=d['text_encoder']['model_dim'],
            tenc_ff_dim=d['text_encoder']['ff_dim'],
            tenc_attn_heads=d['text_encoder']['num_heads']), rngs=rngs)