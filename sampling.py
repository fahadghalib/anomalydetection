"""Zero-leakage hybrid sampling: ADASYN oversampling + capped undersampling.

Both steps must only ever be called on the TRAINING portion of a fold.
Validation and test folds must never pass through this module.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from . import config


def hybrid_resample(X: np.ndarray, y: np.ndarray, random_state: int = config.RANDOM_STATE):
    """Cap majority classes, then ADASYN-oversample minority classes.

    ADASYN is applied one class at a time (rather than as a single
    multi-class call) because its adaptive density ratio can legitimately
    compute "0 synthetic samples needed" for a class whose minority points
    are all deep in same-class neighborhoods (well-separated clusters) --
    imblearn raises ValueError in that case instead of just skipping the
    class. Per-class application lets a single such class fall back to
    plain random oversampling (duplication) without losing ADASYN's
    density-aware synthesis for every other class.

    Returns (X_res, y_res, before_counts, after_counts, fallback_classes).
    """
    classes, counts = np.unique(y, return_counts=True)
    before_counts = dict(zip(classes.tolist(), counts.tolist()))

    # Step 1: cap classes above UNDERSAMPLE_CAP via random undersampling.
    under_strategy = {
        c: config.UNDERSAMPLE_CAP
        for c, n in before_counts.items()
        if n > config.UNDERSAMPLE_CAP
    }
    if under_strategy:
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X, y = rus.fit_resample(X, y)

    # Step 2: oversample classes below ADASYN_TARGET, one class at a time.
    classes, counts = np.unique(y, return_counts=True)
    counts_map = dict(zip(classes.tolist(), counts.tolist()))
    minority_classes = [c for c, n in counts_map.items() if n < config.ADASYN_TARGET]

    fallback_classes = []
    for c in minority_classes:
        current_n = int((y == c).sum())
        if current_n >= config.ADASYN_TARGET:
            continue
        n_neighbors = min(config.ADASYN_N_NEIGHBORS, max(1, current_n - 1))
        try:
            adasyn = ADASYN(
                sampling_strategy={c: config.ADASYN_TARGET},
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            X, y = adasyn.fit_resample(X, y)
        except ValueError:
            # ADASYN found no borderline samples to synthesize from for this
            # class -- fall back to random oversampling (duplication) so the
            # pipeline still reaches the target count for this class.
            fallback_classes.append(int(c))
            ros = RandomOverSampler(
                sampling_strategy={c: config.ADASYN_TARGET}, random_state=random_state
            )
            X, y = ros.fit_resample(X, y)

    classes, counts = np.unique(y, return_counts=True)
    after_counts = dict(zip(classes.tolist(), counts.tolist()))

    return X, y, before_counts, after_counts, fallback_classes
