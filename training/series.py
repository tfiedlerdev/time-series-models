"""Synthetic series used by every training task."""

import numpy as np
import torch


def lag(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    out[steps:] = values[:-steps]
    return out


def observe(values: np.ndarray, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise whose scale is `noise_frac` of this series' own spread."""
    out = values.astype(float).copy()
    seen = np.isfinite(out)
    out[seen] += rng.normal(0, noise_frac * float(np.std(out[seen])), int(seen.sum()))
    return out


def signed_power(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** power


def walk(rng: np.random.Generator, length: int, scale: float) -> np.ndarray:
    innov = rng.normal(0.0, scale, length)
    out = np.empty(length)
    out[0] = innov[0]
    for index in range(1, length):
        out[index] = 0.97 * out[index - 1] + innov[index]
    return out


def seasonal_parts(length: int, daily_phase: float, weekly_phase: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(length)
    daily = np.sin(2 * np.pi * time / 24.0 + daily_phase)
    weekly = 0.6 * np.sin(2 * np.pi * time / 168.0 + weekly_phase)
    return daily, weekly


def calendar(length: int, daily_phase: float, weekly_phase: float) -> np.ndarray:
    daily, weekly = seasonal_parts(length, daily_phase, weekly_phase)
    return daily + weekly


def make_signal(rng: np.random.Generator, length: int) -> np.ndarray:
    """Seasonal swing plus a mean-reverting walk, the signal measured in view_curve.py."""
    innov = rng.normal(0, 0.35, length)
    signal = np.empty(length)
    signal[0] = innov[0]
    for i in range(1, length):
        signal[i] = 0.98 * signal[i - 1] + innov[i]
    signal += 1.5 * np.sin(2 * np.pi * np.arange(length) / 64)
    return signal


PERIODS = (4, 7, 12, 24, 48, 52, 96, 168, 288, 336, 365)
FEATURES = 64


def _random_features(rng: np.random.Generator, time: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    phases = rng.uniform(0, 2 * np.pi, frequencies.shape[0])
    weights = rng.normal(size=frequencies.shape[0])
    return np.sqrt(2.0 / frequencies.shape[0]) * np.cos(np.outer(time, frequencies) + phases) @ weights


def _periodic(rng: np.random.Generator, time: np.ndarray, period: float, scale: float) -> np.ndarray:
    """Draw with covariance exp(-2 sin^2(pi d / period) / scale^2), from its Fourier series."""
    grid = np.arange(256) / 256
    spectrum = np.clip(np.fft.rfft(np.exp(-2 * np.sin(np.pi * grid) ** 2 / scale**2)).real / 256, 0, None)
    spectrum[1:] *= 2
    harmonics = np.arange(spectrum.shape[0])[:24]
    amplitude = np.sqrt(spectrum[:24])
    angle = 2 * np.pi * np.outer(time, harmonics) / period
    return np.cos(angle) @ (amplitude * rng.normal(size=24)) + np.sin(angle) @ (amplitude * rng.normal(size=24))


def _kernel_draw(rng: np.random.Generator, time: np.ndarray, length: int) -> np.ndarray:
    """One independent draw from a kernel in the KernelSynth bank. Lengthscales are in units of the series."""
    choice = int(rng.integers(0, 6))
    if choice == 0:
        return _periodic(rng, time, float(rng.choice(PERIODS)), float(rng.choice((0.5, 1.0, 2.0))))
    if choice == 1:
        center = float(rng.uniform(0, 1)) * length
        return float(rng.normal(0, float(rng.choice((0.5, 2.0, 5.0))))) * (time - center) / length
    if choice == 2:
        scale = float(rng.choice((0.01, 0.05, 0.1, 0.5, 1.0))) * length
        return _random_features(rng, time, rng.normal(0, 1 / scale, FEATURES))
    if choice == 3:
        alpha = float(rng.choice((0.1, 1.0, 10.0)))
        scale = float(rng.choice((0.05, 0.2, 1.0))) * length
        precision = rng.gamma(alpha, 1 / alpha, FEATURES)
        return _random_features(rng, time, rng.normal(0, 1, FEATURES) * np.sqrt(precision) / scale)
    if choice == 4:
        return rng.normal(0, float(rng.choice((0.05, 0.2))), length)
    return np.full(length, float(rng.normal(0, 1)))


def kernel_synth(rng: np.random.Generator, length: int) -> np.ndarray:
    """A random sum and product of one to five kernels, after the KernelSynth recipe of Chronos.

    Independent draws are added for a sum kernel and multiplied for a product kernel,
    which has the same covariance as the exact Gaussian-process draw and costs no
    Cholesky factor, so any length is cheap.
    """
    time = np.arange(length, dtype=np.float64)
    draw = _kernel_draw(rng, time, length)
    for _ in range(int(rng.integers(0, 5))):
        other = _kernel_draw(rng, time, length)
        draw = draw + other if rng.random() < 0.5 else draw * other
    spread = float(np.std(draw))
    if spread < 1e-6:
        draw = draw + rng.normal(0, 1, length)
        spread = float(np.std(draw))
    return (draw - draw.mean()) / spread + float(rng.normal(0, 1))


def _seasonal(rng: np.random.Generator, time: np.ndarray) -> np.ndarray:
    out = np.zeros(time.shape[0])
    for period in rng.choice(PERIODS, size=int(rng.integers(1, 4)), replace=False):
        amplitude = float(rng.uniform(0.2, 2.0))
        for harmonic in range(1, int(rng.integers(1, 4)) + 1):
            phase = float(rng.uniform(0, 2 * np.pi))
            out += amplitude / harmonic * np.sin(2 * np.pi * harmonic * time / float(period) + phase)
    return out


def _ar(rng: np.random.Generator, length: int, phi: float, scale: float) -> np.ndarray:
    innov = rng.normal(0.0, scale, length)
    out = np.empty(length)
    out[0] = innov[0]
    for index in range(1, length):
        out[index] = phi * out[index - 1] + innov[index]
    return out


def composite(rng: np.random.Generator, length: int) -> np.ndarray:
    """Seasons, a trend, an AR process, level shifts, and spikes, combined additively or multiplicatively."""
    time = np.arange(length)
    seasonal = _seasonal(rng, time) if rng.random() < 0.8 else np.zeros(length)
    trend = float(rng.normal(0, 2.0)) * time / length if rng.random() < 0.5 else np.zeros(length)
    noise = _ar(rng, length, float(rng.uniform(0.0, 0.99)), float(rng.uniform(0.05, 0.5)))
    shifts = np.zeros(length)
    for start in rng.integers(0, length, size=int(rng.poisson(0.7))):
        shifts[start:] += float(rng.normal(0, 1.0))
    body = seasonal + trend + noise + shifts
    if rng.random() < 0.3:
        body = np.exp(0.4 * (body - body.mean())) * float(rng.uniform(1, 10))
    spikes = rng.random(length) < float(rng.choice((0.0, 0.002, 0.01)))
    body[spikes] += rng.normal(0, 4 * float(np.std(body)) + 1e-3, int(spikes.sum()))
    return body


def intermittent(rng: np.random.Generator, length: int) -> np.ndarray:
    """Counts whose rate follows a season and a slow walk. Low rates give mostly zeros."""
    time = np.arange(length)
    log_rate = 0.5 * _seasonal(rng, time) + _ar(rng, length, 0.98, float(rng.uniform(0.02, 0.15)))
    rate = float(rng.uniform(0.05, 5.0)) * np.exp(log_rate - log_rate.mean())
    return rng.poisson(rate).astype(np.float64)


def regime(rng: np.random.Generator, length: int) -> np.ndarray:
    """An AR process that switches between two or three regimes, each with its own mean, memory and noise."""
    n_regimes = int(rng.integers(2, 4))
    means = rng.normal(0, 2.0, n_regimes)
    phis = rng.uniform(0.3, 0.99, n_regimes)
    scales = rng.uniform(0.05, 0.8, n_regimes)
    stay = float(rng.uniform(0.98, 0.999))
    seasonal = 0.5 * _seasonal(rng, np.arange(length)) if rng.random() < 0.5 else np.zeros(length)
    state = int(rng.integers(0, n_regimes))
    deviation = 0.0
    out = np.empty(length)
    for index in range(length):
        if rng.random() > stay:
            state = int(rng.integers(0, n_regimes))
        deviation = phis[state] * deviation + rng.normal(0, scales[state])
        out[index] = means[state] + deviation
    return out + seasonal


def _event_shape(rng: np.random.Generator, width: int) -> np.ndarray:
    steps = np.arange(width) / max(width - 1, 1)
    choice = int(rng.integers(0, 4))
    if choice == 0:
        return np.ones(width)
    if choice == 1:
        return 1 - np.abs(2 * steps - 1)
    if choice == 2:
        return np.exp(-steps * float(rng.uniform(2, 8)))
    return np.exp(-0.5 * ((steps - 0.5) / 0.18) ** 2)


def events(rng: np.random.Generator, length: int) -> np.ndarray:
    """A seasonal AR baseline with localised events: plateaus, ramps, decays and bumps of random width and sign."""
    time = np.arange(length)
    body = _seasonal(rng, time) + _ar(rng, length, float(rng.uniform(0.5, 0.99)), float(rng.uniform(0.05, 0.4)))
    for start in rng.integers(0, length, size=int(rng.poisson(length / 400) + 1)):
        width = int(rng.integers(3, max(4, length // 20)))
        end = min(length, int(start) + width)
        size = float(rng.normal(0, 3.0))
        body[start:end] += size * _event_shape(rng, width)[: end - int(start)]
    return body


def piecewise(rng: np.random.Generator, length: int) -> np.ndarray:
    """A piecewise-linear trend with changepoints, a season, and AR noise, sometimes multiplicative."""
    time = np.arange(length)
    knots = np.sort(rng.integers(1, length - 1, size=int(rng.integers(1, 6))))
    slopes = rng.normal(0, 3.0 / length, knots.shape[0] + 1)
    slope = np.full(length, slopes[0])
    for knot, value in zip(knots, slopes[1:]):
        slope[knot:] = value
    trend = np.cumsum(slope) * float(rng.uniform(0.5, 3.0))
    seasonal = _seasonal(rng, time) * float(rng.uniform(0, 1))
    noise = _ar(rng, length, float(rng.uniform(0, 0.95)), float(rng.uniform(0.05, 0.4)))
    if rng.random() < 0.3:
        level = trend - trend.min() + 1.0
        return level * (1 + 0.2 * seasonal) + noise
    return trend + seasonal + noise


def _waveform(rng: np.random.Generator, phase: np.ndarray) -> np.ndarray:
    """One cycle shape on phase in [0, 1): sawtooth, square, a clipped arch, or a narrow peak."""
    choice = int(rng.integers(0, 4))
    if choice == 0:
        return phase - 0.5
    if choice == 1:
        return np.where(phase < float(rng.uniform(0.2, 0.8)), 1.0, -1.0)
    if choice == 2:
        return np.maximum(np.sin(2 * np.pi * phase), 0.0)
    center = float(rng.uniform(0, 1))
    distance = np.minimum(np.abs(phase - center), 1 - np.abs(phase - center))
    return np.exp(-0.5 * (distance / float(rng.uniform(0.02, 0.1))) ** 2)


def profile(rng: np.random.Generator, length: int) -> np.ndarray:
    """A non-sinusoidal periodic profile whose amplitude drifts, with noise that grows with the level.

    Arched profiles stay at zero between cycles, like solar output.
    """
    time = np.arange(length)
    period = float(rng.choice(PERIODS[2:]))
    shape = _waveform(rng, (time / period + float(rng.uniform(0, 1))) % 1.0)
    amplitude = np.exp(0.3 * _ar(rng, length, 0.995, float(rng.uniform(0.01, 0.05))))
    body = amplitude * shape * float(rng.uniform(0.5, 3.0))
    if rng.random() < 0.5:
        body = body + 0.4 * _seasonal(rng, time)
    noise = rng.normal(0, 1, length) * (0.05 + 0.2 * np.abs(body)) * float(rng.uniform(0.2, 1.0))
    out = body + noise
    if shape.min() >= 0:
        out = np.maximum(out, 0.0)
    return out


def make_series(rng: np.random.Generator, length: int, kind: str) -> np.ndarray:
    """A series whose own future is the forecast target."""
    if kind == "train":
        return make_signal(rng, length)
    if kind == "kernel":
        return kernel_synth(rng, length)
    if kind == "composite":
        return composite(rng, length)
    if kind == "intermittent":
        return intermittent(rng, length)
    if kind == "regime":
        return regime(rng, length)
    if kind == "events":
        return events(rng, length)
    if kind == "piecewise":
        return piecewise(rng, length)
    if kind == "profile":
        return profile(rng, length)
    if kind == "walk":
        return np.cumsum(rng.normal(0, 0.4, length))
    if kind == "season":
        time = np.arange(length)
        return (
            2.0 * np.sin(2 * np.pi * time / 24)
            + 0.8 * np.sin(2 * np.pi * time / 168)
            + rng.normal(0, 0.3, length)
        )
    if kind == "trend":
        time = np.arange(length)
        return float(rng.normal()) + float(rng.normal(0, 0.01)) * time + rng.normal(0, 0.4, length)
    raise ValueError(kind)


def make_covariate(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One noisy nonlinear view. About half are delayed."""
    lag_k = 0 if rng.random() < 0.5 else int(rng.integers(1, 41))
    src = signal if lag_k == 0 else lag(signal, lag_k)
    kind = int(rng.integers(0, 3))
    if kind == 0:
        values = signed_power(src, float(rng.uniform(0.4, 1.8)))
    elif kind == 1:
        values = np.tanh(src / float(rng.uniform(0.5, 3.0)))
    else:
        values = src / (1 + float(rng.uniform(0.1, 0.8)) * src**2)
    return observe(values, float(rng.uniform(0.1, 0.7)), rng)


def pack(rows: list[np.ndarray], variates: bool, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(np.stack(rows), dtype=torch.float32, device=device)
    return tensor if variates else tensor[:, None, :]
