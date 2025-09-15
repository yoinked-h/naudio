from transformers.models.t5 import T5Tokenizer as tk
import jax.numpy as jnp

class T5Tokenizer():
    """wrap the transformers's T5Tokenizer in a easy to use class since i dont want to ['input_ids'] every time"""
    def __init__(self, tokenizer_path="google-t5/t5-base"):
        self.tokenizer: tk = tk.from_pretrained(tokenizer_path, legacy=False, clean_up_tokenization_spaces=False)
    def __call__(self, text, return_attnmask=False) -> jnp.ndarray:
        p = self.tokenizer(text, return_tensors="jax")
        if return_attnmask:
            return p # type: ignore
        return p["input_ids"] # type: ignore
    def tokenize_padded(self, text: str, max_length: int, ) -> jnp.ndarray:
        p = self.tokenizer(text, return_tensors="jax", padding="max_length", max_length=max_length, truncation=True, pad_to_multiple_of=None)
        encoded_ids: jnp.ndarray = p["input_ids"] # type: ignore
        return encoded_ids