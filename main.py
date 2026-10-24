# %%
import matplotlib.pyplot as plt
import numpy as np
from timestral import TimestralForecaster

rng = np.random.default_rng(0)
context_len = 512
prediction_length = 96
n_covariates = 100
n_draws = 8
n = context_len + prediction_length
time = np.arange(n)

# Mean-reverting, so the scale does not depend on how long a history we keep.
innov = rng.normal(0, 0.35, n)
signal = np.empty(n)
signal[0] = innov[0]
for i in range(1, n):
    signal[i] = 0.98 * signal[i - 1] + innov[i]
signal += 1.5 * np.sin(2 * np.pi * time / 64)


def lag(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    out[steps:] = values[:-steps]
    return out


def observe(values: np.ndarray, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise whose scale is `noise_frac` of this series' own spread."""
    out = values.astype(float).copy()
    seen = np.isfinite(out)
    out[seen] += rng.normal(0, noise_frac * np.std(out[seen]), int(seen.sum()))
    return out


def signed_power(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** power


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


covariates = np.stack([make_covariate(signal, rng) for _ in range(n_covariates)])
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M")  # device="cuda" for a GPU
context = signal[:context_len][None, None, :]
truth = signal[context_len:]
# Every covariate is available over the past. A draw chooses which of them are also known on the horizon.
maes = np.empty((n_covariates + 1, n_draws))
draw_rng = np.random.default_rng(1)
for n_known in range(n_covariates, -1, -1):
    repeats = 1 if n_known in (0, n_covariates) else n_draws
    for draw in range(repeats):
        if n_known == 0:
            chosen = np.empty(0, dtype=int)
        elif n_known == n_covariates:
            chosen = np.arange(n_covariates)
        else:
            chosen = draw_rng.choice(n_covariates, n_known, replace=False)
        rest = np.setdiff1d(np.arange(n_covariates), chosen, assume_unique=False)
        kwargs = {"clip_nonnegative": False}
        if chosen.size:
            kwargs["known_covariates"] = covariates[chosen][None, :, :]
        if rest.size:
            kwargs["past_covariates"] = covariates[rest][None, :, :context_len]
        median = forecaster.forecast(context, prediction_length, **kwargs).median.detach().cpu().numpy()[0, 0]
        maes[n_known, draw] = float(np.mean(np.abs(median - truth)))
    if repeats == 1:
        maes[n_known, :] = maes[n_known, 0]
    if n_known % 10 == 0:
        print(f"n={n_known:3d}  median MAE {np.median(maes[n_known]):.3f}", flush=True)

counts = np.arange(n_covariates + 1)
median_mae = np.median(maes, axis=1)
q25, q75 = np.percentile(maes, [25, 75], axis=1)

fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
ax.fill_between(counts, q25, q75, color="C1", alpha=0.25, label=f"IQR across {n_draws} draws")
ax.plot(counts, median_mae, color="C1", lw=1.5, label="median MAE")
ax.set_xlim(0, n_covariates)
ax.set_xlabel("covariates known at forecast time")
ax.set_ylabel("MAE of the clean target")
ax.set_title("Error against how many noisy views are known ahead")
ax.legend(loc="upper right")
plt.show()

# %%
