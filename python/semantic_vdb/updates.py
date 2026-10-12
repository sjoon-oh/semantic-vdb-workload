"""Request mix and insert / delete schedule of a workload set.

A workload set performs ``total`` requests that split into reads (queries) and updates; one
update is one insertion plus one deletion (``request_mix``), as in the fugue workloads.

Global ids: base row ``i`` has id ``i``; the ``j``-th inserted vector (the ``j``-th row of
the base set's fixed insert order) has id ``num_base + j``. Ids are never reused.

Inside one step the order is: delete, then insert, then queries. Deletions of a step are
drawn uniformly at random from the vectors that are alive BEFORE that step's insertions,
so a vector inserted in step ``i`` cannot be deleted in step ``i``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import List, Optional

import numpy as np


class ScheduleError(ValueError):
    """The requested insert/delete schedule cannot be executed."""


def split_evenly(total: int, parts: int) -> np.ndarray:
    """Split ``total`` into ``parts`` near-equal integers; the first steps get the remainder."""
    if parts < 1:
        raise ScheduleError("the number of steps must be positive")
    if total < 0:
        raise ScheduleError("totals must not be negative")
    base, remainder = divmod(total, parts)
    sizes = np.full(parts, base, dtype=np.int64)
    # The first `remainder` steps take one extra item each.
    sizes[:remainder] += 1
    return sizes


@dataclass(frozen=True)
class RequestMix:
    """How the ``total`` requests of one workload set divide into reads and updates.

    One update is one insertion plus one deletion, so a set performs ``updates`` insertions,
    ``updates`` deletions and ``reads`` queries, and the number of alive vectors stays equal
    to the base size throughout.
    """

    name: str  # directory name of the set, e.g. "u25"
    update_percent: float
    reads: int
    updates: int


def update_set_name(update_percent: float) -> str:
    """Directory name of a set: ``u`` + the percentage (``u25``; 12.5 becomes ``u12p5``)."""
    # Format the shortest exact decimal text (no rounding to 6 digits, no exponent), so that
    # different percentages can never share a name.
    text = format(Decimal(repr(float(update_percent))).normalize(), "f")
    return "u" + text.replace(".", "p")


def request_mix(total: int, update_percent: float) -> RequestMix:
    """Split ``total`` requests into reads and updates for one update percentage.

    The split must be exact: ``total * update_percent / 100`` has to be a whole number,
    otherwise the requested ratio would silently differ from the delivered one.
    """
    if total < 1:
        raise ScheduleError("the total number of requests must be positive")
    if not math.isfinite(update_percent) or not 0.0 <= update_percent < 100.0:
        raise ScheduleError(f"an update percentage must be in [0, 100), got {update_percent}")

    # Go through the decimal text so that 0.1 means one tenth, not the nearest binary float.
    exact_updates = Fraction(total) * Fraction(repr(float(update_percent))) / 100
    if exact_updates.denominator != 1:
        raise ScheduleError(
            f"{update_percent}% of {total} requests is {float(exact_updates)}, not a whole number"
        )
    updates = int(exact_updates)
    return RequestMix(
        name=update_set_name(update_percent),
        update_percent=float(update_percent),
        reads=total - updates,
        updates=updates,
    )


@dataclass(frozen=True)
class UpdateSchedule:
    num_base: int
    inserts_per_step: np.ndarray  # int64, one entry per step
    deletes_per_step: np.ndarray  # int64, one entry per step

    @property
    def num_steps(self) -> int:
        return len(self.inserts_per_step)

    @property
    def total_inserts(self) -> int:
        return int(self.inserts_per_step.sum())

    @property
    def total_deletes(self) -> int:
        return int(self.deletes_per_step.sum())

    def inserted_before_step(self, step_index: int) -> int:
        """Number of vectors inserted in the steps before ``step_index`` (0-based)."""
        return int(self.inserts_per_step[:step_index].sum())


def make_schedule(
    num_base: int,
    total_inserts: int,
    total_deletes: int,
    num_steps: int,
    max_inserts: Optional[int] = None,
) -> UpdateSchedule:
    """Build and validate a schedule without touching any data file.

    Raises ``ScheduleError`` when more vectors are inserted than the base set holds, or
    when some step would have to delete more vectors than are alive at that moment.
    """
    if num_base < 1:
        raise ScheduleError("the base must contain at least one vector")
    if max_inserts is not None and total_inserts > max_inserts:
        raise ScheduleError(
            f"the workload inserts {total_inserts} vectors but the base set holds only {max_inserts}"
        )

    inserts = split_evenly(total_inserts, num_steps)
    deletes = split_evenly(total_deletes, num_steps)

    # Walk through the steps and track how many vectors are alive.
    alive = num_base
    for step_index in range(num_steps):
        if deletes[step_index] > alive:
            raise ScheduleError(
                f"step {step_index + 1} must delete {deletes[step_index]} vectors but only "
                f"{alive} are alive before its insertions"
            )
        alive = alive - int(deletes[step_index]) + int(inserts[step_index])

    return UpdateSchedule(num_base=num_base, inserts_per_step=inserts, deletes_per_step=deletes)


def draw_deletions(schedule: UpdateSchedule, seed: int) -> List[np.ndarray]:
    """Choose the ids deleted in every step (sorted int64 arrays, one per step)."""
    rng = np.random.default_rng(seed)
    alive_ids = np.arange(schedule.num_base, dtype=np.int64)
    deleted_per_step: List[np.ndarray] = []
    inserted_so_far = 0

    for step_index in range(schedule.num_steps):
        delete_count = int(schedule.deletes_per_step[step_index])
        if delete_count > 0:
            # Uniform subset of the vectors alive before this step's insertions.
            positions = rng.choice(len(alive_ids), size=delete_count, replace=False)
            deleted_per_step.append(np.sort(alive_ids[positions]))
            alive_ids = np.delete(alive_ids, positions)
        else:
            deleted_per_step.append(np.empty(0, dtype=np.int64))

        # Only now do this step's insertions become alive (and thus deletable later).
        insert_count = int(schedule.inserts_per_step[step_index])
        first_new_id = schedule.num_base + inserted_so_far
        new_ids = np.arange(first_new_id, first_new_id + insert_count, dtype=np.int64)
        alive_ids = np.concatenate([alive_ids, new_ids])
        inserted_so_far += insert_count

    return deleted_per_step
