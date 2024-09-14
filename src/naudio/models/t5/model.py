import jax
import jax.numpy as jnp
from beartype import beartype
from dataclasses import dataclass
from flax import nnx
from jaxtyping import Array, Float, jaxtyped
from naudio.models.activations import ReLU
@dataclass
class T5Config:
    d_model: int
    d_ff: int
    num_heads: int
    layer_norm_epsilon: float = 1e-6
    num_layers: int = 6
    vocab_size: int = 32128
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
class T5LayerNorm(nnx.Module):
    epsilon: float = 1e-6
    def __init__(self, config, rngs:nnx.Rngs):
        self.epsilon = config.layer_norm_epsilon
        self.weight = nnx.Param(
            jnp.ones(config.d_model), name="weight", rngs=rngs
        )
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        return x * self.weight.value

class T5DenseActDense(nnx.Module):
    def __init__(self, config, rngs:nnx.Rngs):
        self.config = config
        self.wi = nnx.Linear(config.d_model, config.d_ff, use_bias=False, rngs=rngs)
        self.wo = nnx.Linear(config.d_ff, config.d_model, use_bias=False, rngs=rngs)
        self.act = ReLU()
    def __call__(self, x):

        h = self.wi(x)
        h = self.act(h)
        h = self.wo(h)
        return h

class T5LayerFF(nnx.Module):
    def __init__(self, config, rngs:nnx.Rngs):
        self.config = config
        self.t5ln = T5LayerNorm(config, rngs)
        self.dense = T5DenseActDense(config, rngs)
    def __call__(self, x):

        y = self.t5ln(x)
        y = self.dense(y)
        return x + y

class T5Attention(nnx.Module):
    
    def __init__(self, config, rngs:nnx.Rngs):
        
        self.config = config
        # todo        
        self.toq = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.tok = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.tov = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        self.too = nnx.Linear(config.d_model, config.d_model, use_bias=False, rngs=rngs)
        
        self.relattnbias = nnx.Embed(config.relative_attention_num_buckets, config.num_heads, rngs=rngs)
    def __call__(self, hidden_states, mask=None):
        num_heads = self.config.num_heads
        d_kv = self.config.d_model // num_heads

        # Linear layers for query, key, and value
        q = self.toq(hidden_states)
        k = self.tok(hidden_states)
        v = self.tov(hidden_states)
        # Reshape heads
        batch_size, seq_length = hidden_states.shape[:2]
        q = q.reshape(batch_size, seq_length, num_heads, d_kv).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_length, num_heads, d_kv).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length, num_heads, d_kv).transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_kv)
        scores += self.compute_bias(seq_length, seq_length)
        # Apply attention mask if provided
        if mask is not None:
            scores = jnp.where(mask[:, None, None, :] == 0, -1e9, scores)

        # Compute attention probabilities
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        context = jnp.matmul(attn_weights, v)

        # Reshape output
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.config.d_model)

        # Final linear layer
        output = self.too(context)

        return output, attn_weights
    def compute_bias(self, query_length, key_length):
        num_buckets = self.config.relative_attention_num_buckets
        max_distance = self.config.relative_attention_max_distance
        context_position = jnp.arange(query_length)[:, None]
        memory_position = jnp.arange(key_length)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=num_buckets,
            max_distance=max_distance
        ).astype(jnp.int32)
        
        values = self.relattnbias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]  # Shape: [1, num_heads, query_length, key_length]
        return values

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(jnp.int32) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.maximum(n, 0)
        
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            jnp.log(n.astype(jnp.float32) / max_exact) / jnp.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).astype(jnp.int32)
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
        
        ret += jnp.where(is_small, n, val_if_large)
        return ret


class T5LayerSelfAttention(nnx.Module):
    def __init__(self, config, rngs:nnx.Rngs):
        self.config = config
        self.t5ln = T5LayerNorm(config, rngs)
        self.attention = T5Attention(config,  rngs)
    def __call__(self, x, mask=None):

        y = self.t5ln(x)
        y, _ = self.attention(y, mask)
        
        return x + y

class T5Block(nnx.Module):
    def __init__(self, config, rngs:nnx.Rngs):
        self.config = config
        self.attention = T5LayerSelfAttention(config, rngs)
        self.ff = T5LayerFF(config, rngs)
    def __call__(self, x, mask=None):
        
        x = self.attention(x, mask)
        x = self.ff(x)
        return x

class T5Stack(nnx.Module):
    def __init__(self, config, rngs:nnx.Rngs):
        self.config = config
        self.t5ln = T5LayerNorm(config, rngs)
        self.blocks = [T5Block(config, rngs) for _ in range(config.num_layers)]
    def __call__(self, x, mask=None):
        
        for block in self.blocks:
            x = block(x, mask)
        return self.t5ln(x)

class T5(nnx.Module):
    def __init__(self, config, rngs:nnx.Rngs):
        self.config = config
        self.embed = nnx.Embed(self.config.vocab_size, self.config.d_model, rngs=rngs)
        self.encoder = T5Stack(self.config, rngs)
    def __call__(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        return self.encoder(x, attention_mask)


if __name__ == "__main__":
    config = T5Config(
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        vocab_size=32128
    )
    rngs = nnx.Rngs(0x55b1)
    model = T5(config, rngs)
    x = jnp.ones((1, 128), dtype=jnp.int32)
    print(x.shape)
    x = model(x)
    print(x.shape, x.dtype)