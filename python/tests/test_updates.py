"""Tests for semantic_vdb.updates."""

import numpy as np
import pytest

from semantic_vdb import updates
from semantic_vdb.updates import ScheduleError


# ---------------------------------------------------------------------------
# request mix (reads + updates) and set names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "percent,reads,update_count,name",
    [
        (0, 1000, 0, "u0"),
        (25, 750, 250, "u25"),
        (50, 500, 500, "u50"),
        (12.5, 875, 125, "u12p5"),
        (99, 10, 990, "u99"),
    ],
)
def test_request_mix_splits_the_total_into_reads_and_updates(percent, reads, update_count, name):
    mix = updates.request_mix(1000, percent)
    assert (mix.reads, mix.updates, mix.name) == (reads, update_count, name)
    assert mix.reads + mix.updates == 1000  # the total is reads plus updates
    assert mix.update_percent == percent


def test_request_mix_is_exact_for_decimal_percentages():
    # 0.1% of 10000 is exactly 10; a float computation of 10000 * 0.1 / 100 is not exact.
    assert updates.request_mix(10_000, 0.1).updates == 10
    assert updates.request_mix(1_000_000, 0.7).updates == 7000
    assert updates.request_mix(300, 33.0).updates == 99


@pytest.mark.parametrize("percent", [0.05, 33.33, 0.01])
def test_request_mix_rejects_a_split_that_is_not_a_whole_number(percent):
    with pytest.raises(ScheduleError, match="not a whole number"):
        updates.request_mix(1000, percent)


@pytest.mark.parametrize("percent", [-1, 100, 150, float("nan"), float("inf")])
def test_request_mix_rejects_percentages_outside_zero_to_one_hundred(percent):
    with pytest.raises(ScheduleError):
        updates.request_mix(1000, percent)


def test_request_mix_rejects_an_empty_total():
    with pytest.raises(ScheduleError):
        updates.request_mix(0, 25)


def test_set_names_are_distinct_for_distinct_percentages():
    percents = [0, 1, 1.25, 10, 12, 12.5, 12.25, 25, 99, 0.1, 0.01, 12.34567, 12.34568]
    names = [updates.update_set_name(p) for p in percents]
    assert len(set(names)) == len(names)
    assert updates.update_set_name(25.0) == "u25"  # an integral float is written without ".0"
    assert "/" not in "".join(names) and "." not in "".join(names)


def test_split_evenly_gives_the_remainder_to_the_first_steps():
    assert updates.split_evenly(10, 4).tolist() == [3, 3, 2, 2]
    assert updates.split_evenly(8, 4).tolist() == [2, 2, 2, 2]
    assert updates.split_evenly(2, 5).tolist() == [1, 1, 0, 0, 0]
    assert updates.split_evenly(0, 3).tolist() == [0, 0, 0]
    with pytest.raises(ScheduleError):
        updates.split_evenly(5, 0)
    with pytest.raises(ScheduleError):
        updates.split_evenly(-1, 3)


def test_schedule_totals_match_the_request():
    schedule = updates.make_schedule(100, total_inserts=47, total_deletes=23, num_steps=6)
    assert schedule.total_inserts == 47 and schedule.total_deletes == 23
    assert schedule.num_steps == 6
    assert schedule.inserted_before_step(0) == 0
    assert schedule.inserted_before_step(2) == int(schedule.inserts_per_step[:2].sum())


def test_infeasible_schedules_are_rejected_before_any_data_is_read():
    # Too many deletions for the number of vectors that are alive.
    with pytest.raises(ScheduleError):
        updates.make_schedule(10, total_inserts=0, total_deletes=11, num_steps=1)
    # A step may delete only what is alive BEFORE its own insertions: 5 are alive, 6 asked.
    with pytest.raises(ScheduleError):
        updates.make_schedule(5, total_inserts=60, total_deletes=6, num_steps=1)
    # More insertions than the base set holds.
    with pytest.raises(ScheduleError):
        updates.make_schedule(10, total_inserts=11, total_deletes=0, num_steps=2, max_inserts=10)
    with pytest.raises(ScheduleError):
        updates.make_schedule(0, 0, 0, 1)


def test_feasible_when_earlier_insertions_provide_the_vectors_to_delete():
    # Base holds 4 vectors but 8 deletions are fine: inserted vectors become deletable.
    schedule = updates.make_schedule(4, total_inserts=20, total_deletes=8, num_steps=4)
    deleted = updates.draw_deletions(schedule, seed=1)
    assert sum(len(step) for step in deleted) == 8


def simulate_alive_sets(schedule, deleted_per_step):
    """Independent re-execution: returns the alive set seen by every step's deletions."""
    alive = set(range(schedule.num_base))
    inserted = 0
    alive_before_inserts = []
    for step_index in range(schedule.num_steps):
        alive_before_inserts.append(set(alive))
        alive -= set(deleted_per_step[step_index].tolist())
        count = int(schedule.inserts_per_step[step_index])
        alive |= set(range(schedule.num_base + inserted, schedule.num_base + inserted + count))
        inserted += count
    return alive_before_inserts


def test_deletions_come_only_from_alive_vectors_without_repeats():
    schedule = updates.make_schedule(50, total_inserts=80, total_deletes=90, num_steps=8)
    deleted = updates.draw_deletions(schedule, seed=3)
    alive_before = simulate_alive_sets(schedule, deleted)

    all_deleted = []
    for step_index, ids in enumerate(deleted):
        assert len(ids) == schedule.deletes_per_step[step_index]
        assert np.all(np.diff(ids) > 0)  # sorted and duplicate free inside a step
        # Every deleted id was alive before this step's insertions.
        assert set(ids.tolist()) <= alive_before[step_index]
        all_deleted.extend(ids.tolist())
    # A vector is never deleted twice, in any step.
    assert len(all_deleted) == len(set(all_deleted))


def test_vectors_inserted_in_a_step_cannot_be_deleted_in_that_step():
    # Deleting everything alive each step leaves no room for same-step insertions.
    schedule = updates.make_schedule(10, total_inserts=100, total_deletes=10 + 9 * 10, num_steps=10)
    deleted = updates.draw_deletions(schedule, seed=4)
    for step_index, ids in enumerate(deleted):
        first_new = schedule.num_base + schedule.inserted_before_step(step_index)
        # New ids of this step start at `first_new`; none of them may be deleted yet.
        assert np.all(ids < first_new)


def test_inserted_vectors_are_deleted_at_the_expected_rate():
    # Expected number of deletions that hit inserted vectors, from an exact expectation
    # recursion (each deletion removes alive vectors uniformly, so every class loses the
    # same fraction in expectation).
    num_base, steps = 200, 8
    schedule = updates.make_schedule(num_base, total_inserts=400, total_deletes=320, num_steps=steps)
    expected_inserted_deleted = 0.0
    alive_base, alive_inserted = float(num_base), 0.0
    for step_index in range(steps):
        deletes = float(schedule.deletes_per_step[step_index])
        alive = alive_base + alive_inserted
        fraction = deletes / alive
        expected_inserted_deleted += alive_inserted * fraction
        alive_base *= 1.0 - fraction
        alive_inserted = alive_inserted * (1.0 - fraction) + float(schedule.inserts_per_step[step_index])

    observed = []
    for seed in range(300):
        deleted = np.concatenate(updates.draw_deletions(schedule, seed))
        observed.append(int((deleted >= num_base).sum()))
    assert abs(np.mean(observed) - expected_inserted_deleted) < 0.05 * expected_inserted_deleted


def test_same_seed_is_deterministic_and_other_seed_differs():
    schedule = updates.make_schedule(100, 50, 60, 5)
    first = updates.draw_deletions(schedule, seed=9)
    again = updates.draw_deletions(schedule, seed=9)
    other = updates.draw_deletions(schedule, seed=10)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_zero_deletes_and_zero_inserts():
    schedule = updates.make_schedule(10, 0, 0, 3)
    assert [len(step) for step in updates.draw_deletions(schedule, 0)] == [0, 0, 0]
