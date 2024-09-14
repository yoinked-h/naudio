from transformers import T5Tokenizer as tk

class T5Tokenizer():
    def __init__(self, tokenizer_path="google-t5/t5-base"):
        self.tokenizer = tk.from_pretrained(tokenizer_path)

    def __call__(self, text):
        return self.tokenizer(text, return_tensors="jax")