# ported from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/training/losses/auraloss.py
import jax
import jax.numpy as jnp
import scipy.signal
from flax import nnx
import librosa.filters


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
    signal = signal[0]
    signal_length = signal.shape[1]

    if frame_length is None:
        frame_length = n_fft
    if frame_step is None:
        frame_step = int(frame_length // 2)

    # Add the input channel dimension.
    signal = signal[:, :, jnp.newaxis]

    # Get the window function.
    window_fn = window_fn.replace('_window', '')
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
        print(signal.shape)
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


def mse_loss(x, y):
    return jnp.mean((x - y)**2)

def apply_reduction(losses, reduction="none"):
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    elif reduction == "none":
        pass
    return losses

def get_reduction(reduction):
    if reduction == "mean":
        return jnp.mean
    elif reduction == "sum":
        return jnp.sum
    elif reduction == "none":
        return lambda x: x
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class L1Loss(nnx.Module):
    def __init__(self, reduction="mean"):
        self.redfunc = get_reduction(reduction)

    def __call__(self, x, y):
        
        return self.redfunc(jnp.abs(x - y))

class L2Loss(nnx.Module):
    def __init__(self, reduction="mean"):
        self.redfunc = get_reduction(reduction)

    def __call__(self, x, y):
        return self.redfunc((x - y)**2)

class SpectralConvergenceLoss(nnx.Module):
    def __init__(self):
        pass
    
    def __call__(self, x_mag, y_mag):
        print(x_mag.shape, y_mag.shape)
        minlen = min(x_mag.shape[0], y_mag.shape[0])
        x_mag = x_mag[:minlen]
        y_mag = y_mag[:minlen]
        return (jnp.linalg.norm(y_mag - x_mag, ord="fro", axis=(-2, -1)) / jnp.linalg.norm(y_mag, ord="fro", axis=(-2, -1))).mean()

class STFTMagnitudeLoss(nnx.Module):
    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac
        if distance == "L1":
            self.distance = L1Loss(reduction)
        elif distance == "L2":
            self.distance = L2Loss(reduction)
        else:
            raise ValueError(f"Invalid distance: {distance}")
    def __call__(self, x, y):
        if self.log:
            x = jnp.log(self.log_eps + self.log_fac * x)
            y = jnp.log(self.log_eps + self.log_fac * y)
        minlen = min(x.shape[0], y.shape[0])
        x = x[:minlen]
        y = y[:minlen]
        return self.distance(x, y)
class SumAndDifferenceLoss(nnx.Module):
    def __init__(self):
        pass
    def __call__(self, x):
        #must be stereo
        if x.ndim != 3: 
            raise ValueError("input must be 3D (b, t, c)")
        if x.shape[2] != 2: 
            raise ValueError("input must be stereo")
        sum_sig = jnp.expand_dims(self.sum(x), 2)
        diff_sig = jnp.expand_dims(self.diff(x), 2)
        
        return sum_sig, diff_sig
    @staticmethod
    def sum(x):
        return x[:, :, 0] + x[:, :, 1]
    @staticmethod
    def diff(x):
        return x[:, :, 0] - x[:, :, 1]

class FIRFilter(nnx.Module):

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        import scipy.signal

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = jnp.reshape(jnp.array([1, 0, -coef]), (1, -1, 1))
        elif filter_type == "fd":
            self.fir = jnp.reshape(jnp.array([1, -coef, 0]), (1, -1, 1))
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * jnp.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = jnp.polymul(
                jnp.array([1, 4 * jnp.pi * f4, (2 * jnp.pi * f4) ** 2]),
                jnp.array([1, 4 * jnp.pi * f1, (2 * jnp.pi * f1) ** 2]),
            )
            DENs = jnp.polymul(
                jnp.polymul(DENs, jnp.array([1, 2 * jnp.pi * f3])), jnp.array([1, 2 * jnp.pi * f2])
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # b c l -> b l c
            # taps = taps.reshape(())
            self.fir = jnp.reshape(jnp.array(taps.astype("float32")), (1, -1, 1))


    def __call__(self, input, target):
        input = jax.lax.conv_general_dilated(
            input, self.fir, padding=[(self.ntaps // 2, 0)], window_strides=[1], dimension_numbers=('NHC', 'OHI', 'NHC')
        )
        target = jax.lax.conv_general_dilated(
            target, self.fir, padding=[(self.ntaps // 2, 0)], window_strides=[1], dimension_numbers=('NHC', 'OHI', 'NHC')
        )
        return input, target


class STFTLoss(nnx.Module):
    def __init__(self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        sample_rate: float|None = None,
        scale: str|None = None,
        n_bins: int|None = None,
        perceptual_weighting: bool = False,
        scale_invariance: bool = False,
        eps: float = 1e-8,
        output: str = "loss",
        reduction: str = "mean",
        mag_distance: str = "L1",
        name: str = "stft_loss"):
        self.name = name
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window_type = window
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.perceptual_weighting = perceptual_weighting
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.mag_distance = mag_distance
        
        self.phs_used = bool(self.w_phs)

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(
            log=True,
            reduction=reduction,
            distance=mag_distance
        )
        self.linstft = STFTMagnitudeLoss(
            log=False,
            reduction=reduction,
            distance=mag_distance
        )

        # setup mel filterbank
        if scale is not None:
            if self.scale == "mel":
                assert n_bins is not None
                assert sample_rate != None  # Must set sample rate to use mel scale
                assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
                fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
                fb = jnp.expand_dims(jnp.array(fb), 0)

            elif self.scale == "chroma":
                assert n_bins is not None
                assert sample_rate != None  # Must set sample rate to use chroma scale
                assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
                fb = librosa.filters.chroma(
                    sr=sample_rate, n_fft=fft_size, n_chroma=n_bins
                )

            else:
                raise ValueError(
                    f"Invalid scale: {self.scale}. Must be 'mel' or 'chroma'."
                )
            self.fb = fb
        else:
            self.fb = 1 # identity

        if self.perceptual_weighting:
            # raise ValueError(
                    # "no"
                # )
            assert sample_rate is not None
            self.prefilter = FIRFilter(filter_type="aw", fs=int(sample_rate))
    def _stft(self, x):
        x_stft = stft(
            x,
            n_fft=self.fft_size,
            frame_length=self.hop_size,
            window_fn=self.window_type,
        )
        x_mag = jnp.sqrt(
            jnp.clip((x_stft.real**2) + (x_stft.imag**2), min=self.eps)
        )

        if self.phs_used:
            x_phs = jnp.angle(x_stft)
        else:
            x_phs = None

        return x_mag, x_phs
    def __call__(self, input: jnp.ndarray, target):
        bs, seq_len, chs = input.shape
        if self.perceptual_weighting:  # apply optional A-weighting via FIR filter
            # since FIRFilter only support mono audio we will move channels to batch dim
            input = jnp.transpose(input, [2, 1, 0]) # hacky way to move channels to batch dim, breaks on bs > 1
            target = jnp.transpose(target, [2, 1, 0])
            # input = input.view(bs * chs, 1, -1)
            # target = target.view(bs * chs, 1, -1)

            # now apply the filter to both
            input, target = self.prefilter(input, target)

            # now move the channels back
            input = input.reshape(bs, -1, chs)
            target = target.reshape(bs, -1, chs)
        x_mag, x_phs = self._stft(input)
        y_mag, y_phs = self._stft(target)
        
        if self.scale is not None:
            x_mag = x_mag * self.fb
            y_mag = y_mag * self.fb
        
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag**2).sum([-2, -1]))
            y_mag = y_mag * jnp.expand_dims(alpha, -1)
        
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0
        phs_loss = mse_loss(x_phs, y_phs) if self.phs_used else 0
        
        loss = (
            (self.w_sc * sc_mag_loss) +
            (self.w_log_mag * log_mag_loss) +
            (self.w_lin_mag * lin_mag_loss) +
            (self.w_phs * phs_loss)
        )
        
        loss = apply_reduction(loss, self.reduction)
        
        if self.output == "loss":
            return loss
        else:
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
class MultiResolutionSTFTLoss(nnx.Module):
    def __init__(self,
                fft_sizes,
                hop_sizes,
                win_lengths,
                window='hann_window',
                w_sc=1.0,
                w_log_mag=1.0,
                w_lin_mag=1.0,
                w_phs = 0.0,
                sample_rate= None,
                scale= None,
                n_bins= None,
                perceptual_weighting= False,
                scale_invariance= False,
                **kwargs
                ):
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        self.stft_losses = []
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window=window,
                    w_sc=w_sc,
                    w_log_mag=w_log_mag,
                    w_lin_mag=w_lin_mag,
                    w_phs=w_phs,
                    sample_rate=sample_rate,
                    scale=scale,
                    n_bins=n_bins,
                    perceptual_weighting=perceptual_weighting,
                    scale_invariance=scale_invariance,
                    **kwargs
                )
            )

    def __call__(self, input, target):
        mrloss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for lossclass in self.stft_losses:
            if lossclass.output == "loss":
                loss = lossclass(input, target)
            else:
                loss, sc_mag_loss_d, log_mag_loss_d, lin_mag_loss_d, phs_loss_d = lossclass(input, target)
                sc_mag_loss.append(sc_mag_loss_d)
                log_mag_loss.append(log_mag_loss_d)
                lin_mag_loss.append(lin_mag_loss_d)
                phs_loss.append(phs_loss_d)
            mrloss += loss

        mrloss /= len(self.stft_losses)

        if lossclass.output == "loss": # type: ignore
            return mrloss
        else:
            return mrloss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss
class SumAndDifferenceSTFTLoss(nnx.Module):
    def __init__(self,
                fft_sizes,
                hop_sizes,
                win_lengths,
                window='hann_window',
                w_sum=1.0,
                w_diff=1.0,
                output='loss',
                name=None,
                **kwargs
                ):
        self.sd = SumAndDifferenceLoss()
        self.name = name
        self.w_sum = w_sum
        self.w_diff = w_diff
        self.output = output
        self.mrstft = MultiResolutionSTFTLoss(
            fft_sizes,
            hop_sizes,
            win_lengths,
            window=window,
            **kwargs
        )
    def __call__(self, input, target):
        print(input.shape, target.shape)
        # (b, l, c)
        if input.shape[1] != target.shape[1]:
            #no worries, crop
            minlen = min(input.shape[1], target.shape[1]) 
            input = input[:, :minlen, :]
            target = target[:, :minlen, :]
        # compute sum and difference signals for both
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)
        # compute error in STFT domain
        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2 # type: ignore
        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss
class AuralossLoss(nnx.Module):
    def __init__(self, auraloss_module, input_key: str, target_key: str, weight: float = 1, name: str|None = None):
        
        self.name = name
        self.weight = weight

        self.auraloss_module = auraloss_module

        self.input_key = input_key
        self.target_key = target_key

    def __call__(self, info):
        loss = self.auraloss_module(info[self.input_key], info[self.target_key])

        return self.weight * loss
class MultiLoss(nnx.Module):
    def __init__(self, losses, name: str|None = None):
        self.name = name
        self.losses = losses

    def __call__(self, info):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses

class ValueLoss(nnx.Module):
    def __init__(self, key: str, weight: float = 1.0, name: str|None = None):
        self.name = name
        self.weight = weight
        self.key = key
    
    def __call__(self, info):
        return self.weight * info[self.key]

class AltL1Loss(nnx.Module):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str|None = None, name: str|None = None):
        self.name = name 
        self.loss = L1Loss(reduction="none")
        self.key_a = key_a
        self.key_b = key_b
        self.weight = weight
        self.mask_key = mask_key
    
    def __call__(self, info):
        mse_loss = self.loss(info[self.key_a], info[self.key_b])    

        if self.mask_key is not None and self.mask_key in info:
            mse_loss = mse_loss[info[self.mask_key]]

        mse_loss = mse_loss.mean()

        return self.weight * mse_loss