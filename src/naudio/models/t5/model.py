import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Float, jaxtyped
from naudio.models.activations import ReLU

class T5Attention(nnx.Module):
    def __init__(self, d_model, num_heads, max_distance=128, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv_proj = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        
        # Relative position embeddings
        self.relative_attention_bias = nnx.Embed(max_distance, num_heads, rngs=rngs)

    def __call__(self, x, mask=None):
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
        else:
            batch_size = 1 # fallback
            seq_len = x.shape[0]
        # Single projection for Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        nnx.dot_product_attention(q, k, v, )
        # Reshape to (batch, seq_len, num_heads, d_head)
        q = q.reshape(seq_len, self.num_heads, self.d_head)
        k = k.reshape(seq_len, self.num_heads, self.d_head)
        v = v.reshape(seq_len, self.num_heads, self.d_head)

        # Add relative position bias
        position_bias = self._compute_bias(seq_len)
        attn = nnx.dot_product_attention(q, k, v, bias=position_bias, mask=mask)

        
        attn = jax.nn.softmax(attn, axis=-1)

        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn, v)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

    def _compute_bias(self, seq_len):
        context_position = jnp.arange(seq_len)[:, None]
        memory_position = jnp.arange(seq_len)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        return self.relative_attention_bias(relative_position_bucket)

    def _relative_position_bucket(self, relative_position, max_distance=128):
        # Implement bucketing logic here
        # This is a simplified version and might need to be adjusted
        return jnp.clip(jnp.abs(relative_position), 0, max_distance - 1)

class T5EncoderLayer(nnx.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout_rate, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.self_attn = T5Attention(d_model, num_heads, rngs=rngs)
        self.ff = nnx.Sequential(
            nnx.Linear(d_model, d_ff, rngs=rngs),
            ReLU(),
            nnx.Linear(d_ff, d_model, rngs=rngs)
        )
        self.layer_norm1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate)

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
    enc = T5Encoder(12, 768, 3072, 12, 0.1, rngs=rngs)
    x = jnp.ones((2**16, 768))
    print(x.shape)
    enc = enc(x)
    print(enc.shape)