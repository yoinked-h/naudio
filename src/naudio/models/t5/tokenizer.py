from transformers import T5Tokenizer as tk

class T5Tokenizer():
    """wrap the transformers's T5Tokenizer in a easy to use class since i dont want to ['input_ids'] every time"""
    def __init__(self, tokenizer_path="google-t5/t5-base"):
        self.tokenizer = tk.from_pretrained(tokenizer_path, legacy=False, clean_up_tokenization_spaces=False)

    def __call__(self, text, return_attnmask=False):
        p = self.tokenizer(text, return_tensors="jax")
        if return_attnmask:
            return p
        return p["input_ids"]