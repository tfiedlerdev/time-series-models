"""How much the leading measurement can say about the early horizon, on the eval's own draws.

The eval series are drawn again with the same seed, keeping the pieces `make_lead_batch`
combines: the measurement, the response applied to it, and the target's own walk. The
bound forecasts the first `delay` horizon steps as the response of the noisy reading,
which is known there, plus the own walk decayed from the origin. Its error is the part of
the own walk that is still to come plus the reading noise. A second forecast reads the
own walk at the origin off the context, as the last target value minus the response of
the reading there, so it uses only what the model sees except the response and the
delay. Last value is the reference with no measurement.
"""

import argparse

import numpy as np

from training.data_config import LeadConfig
from training.eval import LEAD_DELAY
from training.schedule import CONTEXT_LEN, HORIZON
from training.series import lag, observe, walk


def response(rng: np.random.Generator):
    """The same draws as `batches._response`, returned as a function."""
    kind = int(rng.integers(0, 3))
    if kind == 0:
        gain, width = float(rng.uniform(1.0, 2.5)), float(rng.uniform(0.5, 2.0))
        return lambda values: gain * np.tanh(values / width)
    if kind == 1:
        power = float(rng.uniform(0.5, 1.8))
        return lambda values: np.sign(values) * np.abs(values) ** power
    gain, sign = float(rng.uniform(0.5, 1.5)), float(rng.choice((-1.0, 1.0)))
    return lambda values: gain * sign * values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123 + 50_000)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = LeadConfig(weight=1)
    delay, length = LEAD_DELAY, CONTEXT_LEN + HORIZON
    bound, seen, last = [], [], []
    for _ in range(args.series):
        measured = walk(rng, length + delay, 0.25)
        respond = response(rng)
        own = walk(rng, length, cfg.own_scale)
        target = respond(measured[:length]) + own
        reading = observe(measured[delay:], float(rng.uniform(cfg.noise_min, cfg.noise_max)), rng)
        lagged = lag(target, int(rng.integers(cfg.decoy_lag_min, cfg.decoy_lag_max + 1)))
        observe(lagged, float(rng.uniform(cfg.noise_min, cfg.noise_max)), rng)
        truth = target[CONTEXT_LEN : CONTEXT_LEN + delay]
        known = reading[CONTEXT_LEN - delay : CONTEXT_LEN]
        decay = 0.97 ** np.arange(1, delay + 1)
        forecast = respond(known) + own[CONTEXT_LEN - 1] * decay
        bound.append(float(np.abs(forecast - truth).mean()))
        residual = target[CONTEXT_LEN - 1] - respond(reading[CONTEXT_LEN - 1 - delay])
        seen.append(float(np.abs(respond(known) + residual * decay - truth).mean()))
        last.append(float(np.abs(target[CONTEXT_LEN - 1] - truth).mean()))
    print(f"early horizon, median MAE over {args.series} series, delay {delay}")
    print(f"true response of the noisy reading   {np.median(bound):.3f}")
    print(f"  own walk read off the context      {np.median(seen):.3f}")
    print(f"last value, no measurement           {np.median(last):.3f}")


if __name__ == "__main__":
    main()
