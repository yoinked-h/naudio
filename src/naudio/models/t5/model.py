import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Float, jaxtyped
from naudio.models.activations import ReLU


class T5SelfAttention(nnx.Module):
    def __init__(self, d_model, num_heads, rel_attn_bias:bool=True, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.n_heads = num_heads
        self.d_model = d_model
        self.self_attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=d_model, out_features=d_model, qkv_features=d_model, use_bias=rel_attn_bias, decode=False, rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
        val = jnp.ones((num_heads, 12)) #todo, make the init random
        self.relative_attn_bias = nnx.Param(value=val)
            
    @staticmethod
    def _rel_pos_bucket(relative_position, bidirectional=True, num_buckets=12, max_distance=128):
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
            jnp.log(
                n.astype(jnp.float32) / max_exact +
                jnp.finfo(jnp.float32).eps) /
            jnp.log(max_distance / max_exact) *
            (num_buckets - max_exact)).astype(jnp.int32)
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
        ret += jnp.where(is_small, n, val_if_large)
        return ret
    def calc_relative_position(self, query_length, key_length):
        
        # archaic code from linen
        
        context_position = jnp.arange(query_length, dtype=jnp.int32)[:, None]
        memory_position = jnp.arange(key_length, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  
        rp_bucket = self._rel_pos_bucket(relative_position)
        ohot = jax.nn.one_hot(rp_bucket, self.relative_attn_bias.shape[-1])
        out = jnp.matmul(ohot, self.relative_attn_bias.value)  

        return out
    def __call__(self, x, mask=None):
        normed = self.ln(x)
        # bias = self.calc_relative_position(x.shape[1], x.shape[1])
        attn_output = self.self_attn(normed, mask=mask)
        # attn_output = attn_output + bias
        return x + attn_output
class T5EncoderLayer(nnx.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_rate, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.self_attn = T5SelfAttention(d_model, num_heads, rngs=rngs)
        self.ff = nnx.Sequential(
            nnx.Linear(d_model, d_ff, rngs=rngs),
            ReLU(),
            nnx.Linear(d_ff, d_model, rngs=rngs)
        )
        self.layer_norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        return self.layer_norm2(x)

class T5Encoder(nnx.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, dropout_rate, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.layers = [T5EncoderLayer(d_model, d_ff, num_heads, dropout_rate, rngs=rngs) 
                    for _ in range(num_layers)]
        self.final_layer_norm = nnx.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer_norm(x)

if __name__ == "__main__":
    print("testing t5")
    rngs = nnx.Rngs(0x5f16)
    MODEL_DIM = 768
    LAYERS = 12
    HEADS = 12
    DROPOUT = 0.1
    enc = T5Encoder(num_layers=LAYERS, d_model=MODEL_DIM, d_ff=MODEL_DIM*3, num_heads=HEADS, dropout_rate=DROPOUT, rngs=rngs)
    x = jnp.ones((1, 3, MODEL_DIM))
    print(x.shape)
    enc = enc(x)
    print(enc.shape)