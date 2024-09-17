import jax
import jax.numpy as jnp
from pathlib import Path
import soundfile as sf

def sfparse(path):
    x = sf.SoundFile(path)
    npied = x.read(dtype='float32')
    jnpd = jax.device_put(npied)
    return jnpd

class AudioDataset:
    def __init__(self, datasetconfig):
        self.audio_dir = datasetconfig["audio_dir"]
        self.text_dir = datasetconfig["text_dir"]
        self.audio_ext = datasetconfig["audio_ext"]
        self.text_ext = datasetconfig["text_ext"]
        self.text_ext_fallback = datasetconfig["text_ext_fallback"]
        self.dataset_enhancements = datasetconfig["dataset_enhancements"]
        for possible_enh in ["random_crop", "caption_dropout", "normalize_audio"]:
            if possible_enh not in self.dataset_enhancements:
                self.dataset_enhancements[possible_enh] = None
        self.setup()
    def setup(self):
        self.data = []
        for path in Path(self.audio_dir).rglob(f"*{self.audio_ext}"):
            #check for the textfile
            if Path(path).with_suffix(self.text_ext) in Path(self.text_dir).rglob(f"*{self.text_ext}"):
                self.data.append((sfparse(path), Path(path).with_suffix(self.text_ext).read_text()))
            elif Path(path).with_suffix(self.text_ext_fallback) in Path(self.text_dir).rglob(f"*{self.text_ext_fallback}"):
                self.data.append((sfparse(path), Path(path).with_suffix(self.text_ext_fallback).read_text()))
            else:
                print(f"Could not find a textfile for {path}")
    def get(self, idx, stepnum):
        aud, txt = self.data[idx]
        if self.dataset_enhancements == {}:
            return aud, txt
        else:
            # hacky way to do randomness, TODO: make this use the train seed
            if self.dataset_enhancements["random_crop"] is not None:
                if jax.random.uniform(jax.random.PRNGKey(stepnum), shape=()).item() < self.dataset_enhancements["random_crop"]["chance"]:
                    # random crop
                    lenofsong = aud.shape[0] 
                    scale = jax.random.uniform(jax.random.PRNGKey(stepnum), shape=())
                    scale = scale * (self.dataset_enhancements["random_crop"]["max_scale"] - self.dataset_enhancements["random_crop"]["min_scale"]) + self.dataset_enhancements["random_crop"]["min_scale"] 
                    lenofcrop = lenofsong * scale
                    softend = lenofsong - lenofcrop
                    start = jax.random.uniform(jax.random.PRNGKey(stepnum), shape=()) * softend
                    start = int(start)
                    aud = aud[start:start+lenofcrop]
            if self.dataset_enhancements["caption_dropout"] is not None:
                if jax.random.uniform(jax.random.PRNGKey(stepnum), shape=()).item() < self.dataset_enhancements["caption_dropout"]["chance"]:
                    # caption dropout
                    if len(txt) > self.dataset_enhancements["caption_dropout"]["min_to_trigger"]:
                        txt = "" #improves unconditional sampling
            if self.dataset_enhancements["normalize_audio"] is not None:
                if jax.random.uniform(jax.random.PRNGKey(stepnum), shape=()).item() < self.dataset_enhancements["normalize_audio"]["chance"]:
                    # normalize audio
                    peak = jnp.abs(aud).max()
                    aud = aud / peak
        return aud, txt
        
    def __len__(self):
        return len(self.data)

class PureAudioDataset():
    def __init__(self, datasetconfig):
        self.audio_dir = datasetconfig["audio_dir"]
        self.audio_ext = datasetconfig["audio_ext"]
        self.setup()
    def setup(self):
        self.data = []
        for path in Path(self.audio_dir).rglob(f"*{self.audio_ext}"):
            self.data.append(sfparse(path))
    def get(self, idx, stepnum):
        aud = self.data[idx]
        return aud
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)