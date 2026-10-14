"""Model-free delta grids and PMM saturation rules shared by experiments."""
import numpy as np

SEARCH_METHODS = ("pmm", "lv", "qcor", "tv")
UNIT_DELTA_GRID = tuple(i / 20 for i in range(1, 20))


def pmm_saturation_q95(training_labels):
    if len(training_labels) != 5:
        raise ValueError("PMM saturation requires five training-label arrays")
    minimum = np.concatenate([p.min(axis=1) for p in training_labels])
    if not len(minimum) or np.any(~np.isfinite(minimum)) or np.any(minimum < 0):
        raise ValueError("Invalid training probabilities for PMM saturation")
    with np.errstate(divide="ignore", over="ignore"):
        saturation = np.sort(1.0 / minimum - 1.0)
    position = .95 * (len(saturation) - 1)
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if not np.isfinite(saturation[hi]):
        raise ValueError("PMM saturation Q95 is infinite; use labels with finite Q95")
    return float(saturation[lo] + (position - lo) * (saturation[hi] - saturation[lo]))


def pmm_delta_candidates(q95):
    if not np.isfinite(q95) or q95 <= 0:
        raise ValueError("PMM Q95 must be finite and positive")
    delta, step = min(.05, q95), .05
    while True:
        yield float(delta)
        step *= 1.5
        next_delta = delta + step
        delta = min(q95, next_delta) if delta < q95 else next_delta
        if not np.isfinite(delta):
            raise RuntimeError("PMM search overflowed before meeting both stopping criteria")


def pmm_stop_reason(previous_a, current_a, delta, q95):
    if previous_a is not None and current_a - previous_a < .005 and delta >= q95:
        return "q95_reached_and_adjacent_increment_below_0.005"
    return None
