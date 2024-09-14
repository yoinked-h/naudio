import jax
import jax.numpy as jnp
from beartype import beartype
from flax import nnx
from jaxtyping import Array, Float, jaxtyped
from naudio.models.activations import ReLU
# most of the t5 attn code is borrowed from comfyui's implementation
class T5Attention(nnx.Module):
    def __init__(self, d_model, num_heads, rel_attn_bias:bool=True, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.toq = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.tok = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.tov = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.too = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.n_heads = num_heads
        if rel_attn_bias:
            self.relattnbias_n_buckets = 32
            self.relattnbias_max_distance = 128
            self.relattnbias = nnx.Embed(self.relattnbias_n_buckets, self.n_heads, rngs=rngs)
        else:
            self.relattnbias = None
    @staticmethod
    def _rel_pos_buckets(rel_pos, n_buckets=32, max_distance=128):
        rel_buckets = 0
        n_buckets //= 2
        rel_buckets += (rel_pos > 0) * n_buckets
        rel_pos = jnp.abs(rel_pos)
        max_exact = n_buckets // 2
        is_small = rel_pos < max_exact
        rel_pos_if_large = max_exact + (
            jnp.log(rel_pos / max_exact)
            / jnp.log(max_distance / max_exact)
            * (n_buckets - max_exact)
        )
        rel_pos_if_large = jnp.minimum(rel_pos_if_large, jnp.full_like(rel_pos_if_large, n_buckets - 1))
        rel_buckets += jnp.where(is_small, rel_pos, rel_pos_if_large)
        return rel_buckets
    
    def compute_bias(self, q_len, k_len):
        assert self.relattnbias
        ctx_pos = jnp.arange(q_len)[:, None]
        mem_pos = jnp.arange(k_len)[None, :]
        rel_pos = mem_pos - ctx_pos 
        rel_pos_bucket = self._rel_pos_buckets(rel_pos, self.relattnbias_n_buckets, self.relattnbias_max_distance)
        vals = self.relattnbias(rel_pos_bucket)
        return vals
    def __call__(self, x, mask=None):
        q = self.toq(x)
        k = self.tok(x)
        v = self.tov(x)
        bias = None
        if self.relattnbias is not None:
            bias = self.compute_bias(x.shape[1], x.shape[1])
        if bias is not None:
            if mask is not None:
                mask = mask + bias
            else:
                mask = bias
        out = nnx.dot_product_attention(q, k * ((k.shape[-1] / self.n_heads) ** 0.5), v, mask=mask)
        out = (
            out.transpose(1,2).reshape(q.shape[0], -1, self.n_heads * q.shape[-1])
        )
        return self.too(out), bias

class T5SelfAttention(nnx.Module):
    def __init__(self, d_model, num_heads, rel_attn_bias:bool=True, rngs: nnx.Rngs|None = None) -> None:
        if rngs is None:
            raise ValueError("rngs must not be None")
        self.self_attn = T5Attention(d_model, num_heads, rel_attn_bias, rngs=rngs)
        self.ln = nnx.LayerNorm(d_model, rngs=rngs)
    
    def __call__(self, x, mask=None):
        normed = self.ln(x)
        attn_output, bias = self.self_attn(normed, mask=mask)
        return x + attn_output, bias
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