"""Binary hand command stabilization, without robot I/O or persistent side effects."""
import numpy as np


def stabilize_binary_hand(probabilities, previous, *, state=None,
                          close_threshold=.65, open_threshold=.35,
                          close_frames=6, open_frames=21):
    """Use hysteresis and consecutive model-frame confirmation.

    Return candidate filter state, to be reused ONLY when this plan is the
    client's accepted parent. A discarded speculative plan cannot change it.
    No grasp latch: sustained open evidence still releases either hand.
    """
    p = np.asarray(probabilities, dtype=float)
    prev = np.asarray(previous, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or prev.shape != (2,):
        raise ValueError("Expected hand probabilities [T,2] and previous [2]")
    if not np.isfinite(p).all() or not np.isfinite(prev).all():
        raise ValueError("Non-finite hand input")
    if np.any((p < 0) | (p > 1)) or np.any((prev < 0) | (prev > 1)):
        raise ValueError("Hand closure must be within [0,1]")
    if not 0 <= open_threshold < close_threshold <= 1:
        raise ValueError("Require 0 <= open threshold < close threshold <= 1")
    if close_frames < 1 or open_frames < 1:
        raise ValueError("Confirmation frames must be positive")
    stable = (prev >= .5).astype(int)
    count = np.zeros(2, dtype=int)
    if state is not None:
        old_stable = np.asarray(state["stable"], dtype=int)
        old_count = np.asarray(state["count"], dtype=int)
        if old_stable.shape != (2,) or old_count.shape != (2,):
            raise ValueError("Invalid hand filter state")
        count = np.where(old_stable == stable, old_count, 0)
    out = np.empty(p.shape, dtype=np.float32)
    for i, row in enumerate(p):
        opposite = np.where(stable == 0, row >= close_threshold,
                            row <= open_threshold)
        count = np.where(opposite, count + 1, 0)
        required = np.where(stable == 0, close_frames, open_frames)
        switch = count >= required
        stable = np.where(switch, 1 - stable, stable)
        count = np.where(switch, 0, count)
        out[i] = stable
    return out, {"stable": stable.tolist(), "count": count.tolist()}
