import jax
import jax.numpy as jnp
from flax import nnx
from ..activations import SiLU, get_activation
from functools import reduce
import scipy.signal

def _dft_matrix_np(
    n_points: int,
    inverse: bool = False,
    dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:
    x, y = jnp.meshgrid(jnp.arange(n_points), jnp.arange(n_points))
    if inverse:
        omega = jnp.exp(2.0 * jnp.pi * 1j / n_points)
    else:
        omega = jnp.exp(-2.0 * jnp.pi * 1j / n_points)
    return jnp.power(omega, x * y).astype(dtype)

def stft(signal,
                n_fft: int = 2048,
                frame_length = None,
                frame_step = None,
                window_fn: str = 'hann',
                pad:str = "end",
                pad_mode: str = 'constant',
                ):
    signal_length = signal.shape[1]
    if len(signal.shape) != 2:
        raise ValueError('Input signal should be 2-dimensional.')

    if frame_length is None:
        frame_length = n_fft
    if frame_step is None:
        frame_step = int(frame_length // 2)

    # Add the input channel dimension.
    signal = signal[:, :, jnp.newaxis]

    # Get the window function.
    fft_window = scipy.signal.get_window(window_fn, frame_length, fftbins=True)
    # Pad the window to length n_fft with zeros.
    if frame_length < n_fft:
        left_pad = int((n_fft - frame_length) // 2)
        right_pad = n_fft - frame_length - left_pad
        fft_window = jnp.pad(fft_window, (left_pad, right_pad), mode='constant')
    # Make it broadcastable.
    fft_window = fft_window[:, jnp.newaxis]

    # Pad the signal if needed.
    pad = pad.upper()
    if pad != "NONE":
        if pad == "START":
            pad_shape = (n_fft // 2, 0)    # for istft reconstruction
        elif pad == "END":
            pad_shape = (0, n_fft - 1)    # to mimic pad_end mode of tf.signal.stft
        elif pad == "BOTH":
            pad_shape = (n_fft // 2, n_fft // 2)     # for istft reconstruction
        elif pad == "ALIGNED":
            # Pad signal symmetrically so we obtain aligned frames.
            assert signal_length % frame_step == 0
            assert frame_length % frame_step == 0
            padding = (frame_length - frame_step) // 2
            pad_shape = (padding, padding)
        else:
            raise ValueError(
                    f'Padding should be NONE, START, END, BOTH, or ALIGHED, get {pad}.')

        signal = jnp.pad(signal, pad_width=((0, 0), pad_shape, (0, 0)),
                                        mode=pad_mode)
    elif signal_length < n_fft:
        raise ValueError(
                f'n_fft of {n_fft} is bigger than signal of length {signal_length}')

    # Extract frames and compute FFTs using convlution.
    ch_out = n_fft // 2 + 1
    # w_shape: (kernel_shape, ch_in, ch_out)
    w = (_dft_matrix_np(n_fft)[:, :ch_out] * fft_window)[:, jnp.newaxis, :]
    real = jax.lax.conv_general_dilated(
            signal, jnp.real(w), window_strides=[frame_step], padding='VALID', dimension_numbers=('NHC', 'HIO', 'NHC'))
    imag = jax.lax.conv_general_dilated(
            signal, jnp.imag(w), window_strides=[frame_step], padding='VALID', dimension_numbers=('NHC', 'HIO', 'NHC'))
    return real + 1j * imag


def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = nnx.relu(1 - score_real).mean() + nnx.relu(1 + score_fake).mean()
    return dis_loss, gen_loss
def weightnorm(module):
    return module #todo: implement
class SharedDiscriminatorConvnet(nnx.Module):
    def __init__(self, in_size, conv, out, cap, n_layers, kern, stride, act=lambda: SiLU(), norm=weightnorm, rngs=nnx.Rngs):
        channels = [in_size] 
        channels += list(cap * 2**jnp.arange(n_layers))
        if isinstance(stride, int):
            stride = [stride] * n_layers
        
        net = []
        for n in range(n_layers):
            if isinstance(kern, int):
                pad = kern // 2
                s = stride[n]
            else:
                pad = kern[0] // 2
                s = (stride[n], 1)

            net.append(
                norm(
                    conv(
                        channels[n],
                        channels[n + 1],
                        kernel_size=kern,
                        stride=s,
                        padding=pad,
                        activation=act,
                        rngs=rngs
                    )
                )
            )
            net.append(act())
        net.append(conv(channels[-1], out, 1))

        self.net = net
        
    def __call__(self, x):
        feats = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nnx.Conv):
                feats.append(x)
        score = x.reshape(x.shape[0], -1).mean(1)
        return score, feats


class MultiScaleDiscriminator(nnx.Module):
    def __init__(self, inchannels, n_scales, **kwargs):
        layers = []
        for n in range(n_scales):
            layers.append(
                SharedDiscriminatorConvnet(inchannels, nnx.Conv, **kwargs)
            )
        self.layers = layers

    def __call__(self, x):
        score = 0
        features = []
        for layer in self.layers:
            s, f = layer(x)
            score = score + s
            features.extend(f)
            x = nnx.avg_pool(x, 2)
        return score, features


class MultiPeriodDiscriminator(nnx.Module):

    def __init__(self,
                in_channels,
                periods,
                **conv_kwargs) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(SharedDiscriminatorConvnet(in_channels, nnx.Conv, **conv_kwargs))

        self.layers = layers

    def __call__(self, x):
        score = 0
        features = []
        for layer, n in zip(self.layers, self.periods):
            s, f = layer(self.fold(x, n))
            score = score + s
            features.extend(f)
        return score, features

    def fold(self, x, n):
        pad = (n - (x.shape[-1] % n)) % n
        x = jnp.pad(x, (0, pad))
        return x.reshape(*x.shape[:2], -1, n)

class MultiDiscriminator(nnx.Module):
    """
    Individual discriminators should take a single tensor as input (NxB C T) and
    return a tuple composed of a score tensor (NxB) and a Sequence of Features
    Sequence[NxB C' T'].
    """

    def __init__(self, discriminator_list,
                keys) -> None:
        super().__init__()
        self.discriminators = discriminator_list
        self.keys = keys

    def unpack_tensor_to_dict(self, features):
        features = features.chunk(len(self.keys), 0)
        return {k: features[i] for i, k in enumerate(self.keys)}

    @staticmethod
    def concat_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = []
            if k in dict_a:
                if isinstance(dict_a[k], list):
                    out_dict[k].extend(dict_a[k])
                else:
                    out_dict[k].append(dict_a[k])
            if k in dict_b:
                if isinstance(dict_b[k], list):
                    out_dict[k].extend(dict_b[k])
                else:
                    out_dict[k].append(dict_b[k])
        return out_dict

    @staticmethod
    def sum_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = 0.
            if k in dict_a:
                out_dict[k] = out_dict[k] + dict_a[k]
            if k in dict_b:
                out_dict[k] = out_dict[k] + dict_b[k]
        return out_dict

    def __call__(self, inputs):
        discriminator_input = jnp.concatenate([inputs[k] for k in self.keys], 0)
        all_scores = []
        all_features = []

        for discriminator in self.discriminators:
            score, features = discriminator(discriminator_input)
            scores = self.unpack_tensor_to_dict(score)
            scores = {f"score_{k}": scores[k] for k in scores.keys()}
            all_scores.append(scores)

            features = map(self.unpack_tensor_to_dict, features)
            features = reduce(self.concat_dicts, features)
            features = {f"features_{k}": features[k] for k in features.keys()}
            all_features.append(features)

        all_scores = reduce(self.sum_dicts, all_scores)
        all_features = reduce(self.concat_dicts, all_features)

        inputs.update(all_scores)
        inputs.update(all_features)

        return inputs


def spectrogram(
        waveform,
        power: float = 1.0,
        frame_length = 2048,
        frame_step = None,
        num_features = None,
        window_fn = 'hann',
        pad: str = 'end',
        pad_mode: str = 'constant',
        normalized: bool = False
        ):
    stfts = stft(
            signal=waveform,
            n_fft=frame_length,
            frame_length=frame_length,
            frame_step=frame_step,
            window_fn=window_fn,
            pad=pad,
            pad_mode=pad_mode)

    spectrograms = jnp.power(jnp.abs(stfts), power)
    if normalized:
        spectrograms = spectrograms / jnp.max(spectrograms, axis=-1, keepdims=True)
    return spectrograms[..., :num_features]



def get_2d_padding(kernel_size, dilation = (1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)


class DiscriminatorSTFT(nnx.Module):
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                 n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, max_filters: int = 1024,
                 filters_scale: int = 1, kernel_size = (3, 9), dilations = [1, 2, 4],
                 stride = (1, 2), normalized: bool = True,
                 activation: str = 'LeakyReLU', activation_params: dict = {'negative_slope': 0.2}, rngs: nnx.Rngs|None = None):
        super().__init__()
        assert rngs
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = get_activation(activation, **activation_params)
        spec_channels = 2 * self.in_channels
        self.convs = []
        self.convs.append(
            nnx.Conv(spec_channels, self.filters, kernel_size=kernel_size, padding=get_2d_padding(kernel_size), rngs=rngs)
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(nnx.Conv(in_chs, out_chs, kernel_size=kernel_size, strides=stride,
                                         kernel_dilation=(dilation, 1), padding=get_2d_padding(kernel_size, (dilation, 1)),rngs=rngs))
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(nnx.Conv(in_chs, out_chs, kernel_size=(kernel_size[0], kernel_size[0]),
                                     padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                     rngs=rngs))
        self.conv_post = nnx.Conv(out_chs, self.out_channels,
                                    kernel_size=(kernel_size[0], kernel_size[0]),
                                    padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                                    rngs=rngs)

    def __call__(self, x):
        fmap = []
        z = spectrogram(x,
            frame_length=self.n_fft, window_fn="hann_window",
            normalized=self.normalized, pad_mode="None", power=1)
        z = jnp.concatenate([z.real, z.imag], axis=1)
        # 'b c w t -> b c t w'
        z = z.transpose(0, 1, 3, 2)
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nnx.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """
    def __init__(self, filters: int, in_channels: int = 1, out_channels: int = 1,
                n_ffts = [1024, 2048, 512], hop_lengths = [256, 512, 128],
                win_lengths = [1024, 2048, 512], **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = [
            DiscriminatorSTFT(filters, in_channels=in_channels, out_channels=out_channels,
                              n_fft=n_ffts[i], win_length=win_lengths[i], hop_length=hop_lengths[i], **kwargs)
            for i in range(len(n_ffts))
        ]
        self.num_discriminators = len(self.discriminators)

    def __call__(self, x):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps


class EncodecDiscriminator(nnx.Module):

    def __init__(self, *args, **kwargs):

        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)

    def __call__(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, x, y):
        feature_matching_distance = 0.
        logits_true, feature_true = self(x)
        logits_fake, feature_fake = self(y)

        dis_loss = jnp.array(0.)
        adv_loss = jnp.array(0.)

        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda x, y: abs(x - y).mean(),
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            _dis, _adv = get_hinge_losses(
                logits_true[i],
                logits_fake[i],
            )

            dis_loss = dis_loss + _dis
            adv_loss = adv_loss + _adv

        return dis_loss, adv_loss, feature_matching_distance


class OobleckDiscriminator(nnx.Module):
    def __init__(self, in_channels=1):
        msd = MultiScaleDiscriminator(
            inchannels=in_channels,
            n_scales=3
        )
        mpd = MultiPeriodDiscriminator(
            in_channels=in_channels,
            periods=[2, 3, 5, 7, 11]
        )
        self.multi_discriminator = MultiDiscriminator(
            [msd, mpd], #, multi_resolution_discriminator],
            ["reals", "fakes"]
        )
        
    def loss(self, reals, fakes):
        inputs = {
            "reals": reals,
            "fakes": fakes,
        }

        inputs = self.multi_discriminator(inputs)

        scores_real = inputs["score_reals"]
        scores_fake = inputs["score_fakes"]

        features_real = inputs["features_reals"]
        features_fake = inputs["features_fakes"]

        dis_loss, gen_loss = get_hinge_losses(scores_real, scores_fake)
        feature_matching_distance = jnp.array(0.)

        for _, (scale_real, scale_fake) in enumerate(zip(features_real, features_fake)):

            feature_matching_distance = feature_matching_distance + sum(
                map(
                    lambda real, fake: abs(real - fake).mean(),
                    scale_real,
                    scale_fake,
                )) / len(scale_real)
            
        return dis_loss, gen_loss, feature_matching_distance
    