"""Chronological splits that respect label overlap.

A sample at time t has a label that is only known at ``label_end_ns`` (t + horizon). A training
sample is admissible for a test block starting at s only if ``label_end_ns + embargo <= s``:

* purging removes training samples whose label window overlaps the test block;
* the embargo adds a buffer for serial correlation in features (e.g. a 30 s rolling window
  computed just before s and just after s shares most of its inputs).

Walk-forward folds only ever train on the past, so there is nothing to purge after a test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Fold:
    train: IntArray
    test: IntArray
    test_start_ns: int
    test_end_ns: int
    purged: int  # training rows dropped by purge + embargo


def admissible(
    label_end_ns: IntArray, test_start_ns: int, embargo_ns: int
) -> npt.NDArray[np.bool_]:
    return label_end_ns + embargo_ns <= test_start_ns


def walk_forward(
    t_ns: IntArray, label_end_ns: IntArray, n_folds: int, embargo_ns: int
) -> list[Fold]:
    """Expanding-window folds: split time-sorted samples into n_folds + 1 equal blocks; fold k
    tests on block k and trains on admissible samples from blocks before it."""
    if np.any(np.diff(t_ns) < 0):
        raise ValueError("samples must be sorted by time")
    blocks = np.array_split(np.arange(len(t_ns)), n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        test = blocks[k]
        start = int(t_ns[test[0]])
        before = np.concatenate(blocks[:k])
        keep = admissible(label_end_ns[before], start, embargo_ns)
        folds.append(
            Fold(
                train=before[keep],
                test=test,
                test_start_ns=start,
                test_end_ns=int(t_ns[test[-1]]),
                purged=int((~keep).sum()),
            )
        )
    return folds
