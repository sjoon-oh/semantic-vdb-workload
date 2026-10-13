"""Registry of deliberate bugs ("mutations") that the test-suite must catch.

Each mutation replaces one exact snippet of one source file with a wrong version. The
runner (``run_mutations.py``) applies it to a scratch copy of the code base and expects
at least one test to fail. A mutation that no test notices is reported as SURVIVED.

New modules append their own mutations here when their tests are written.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    description: str
    # Path relative to the ``python/`` directory.
    file: str
    # Exact text to find (must occur exactly once) and its wrong replacement.
    old: str
    new: str
    # Further (file, old, new) edits applied together with the first one, for bugs that
    # need a coordinated change in two places (e.g. an encoder and its decoder).
    also: tuple = ()


BINIO = "semantic_vdb/binio.py"

MUTATIONS = [
    Mutation(
        "B1",
        "header writes cols before rows",
        BINIO,
        "self._file.write(struct.pack(HEADER_FORMAT, num_rows, num_cols))",
        "self._file.write(struct.pack(HEADER_FORMAT, num_cols, num_rows))",
    ),
    Mutation(
        "B2",
        "read_header no longer checks the file size",
        BINIO,
        "    expected_size = HEADER_BYTES + num_rows * num_cols * item_size\n"
        "    if file_size != expected_size:",
        "    expected_size = file_size\n    if file_size != expected_size:",
    ),
    Mutation(
        "B3",
        "writer publishes the file even when fewer rows were written",
        BINIO,
        "        if self._rows_written != self.num_rows:\n"
        "            self._partial_path.unlink(missing_ok=True)",
        "        if False:\n            self._partial_path.unlink(missing_ok=True)",
    ),
    Mutation(
        "B4",
        "ground truth stores distances before ids",
        BINIO,
        "        np.ascontiguousarray(neighbour_ids).tofile(file)\n"
        "        np.ascontiguousarray(neighbour_dists).tofile(file)",
        "        np.ascontiguousarray(neighbour_dists).tofile(file)\n"
        "        np.ascontiguousarray(neighbour_ids).tofile(file)",
    ),
    Mutation(
        "B5",
        "uint8 and int8 extensions are swapped",
        BINIO,
        '    ".u8bin": np.dtype(np.uint8),\n    ".i8bin": np.dtype(np.int8),',
        '    ".u8bin": np.dtype(np.int8),\n    ".i8bin": np.dtype(np.uint8),',
    ),
    Mutation(
        "B6",
        "read_rows accepts out-of-range and negative row ids",
        BINIO,
        "        if row_ids.size and (row_ids.min() < 0 or row_ids.max() >= self.num_rows):",
        "        if False:",
    ),
    Mutation(
        "B7",
        "writer silently accepts rows of the wrong dtype",
        BINIO,
        "        if rows.dtype != self.dtype:\n",
        "        if False:\n",
    ),
]


PARTITION = "semantic_vdb/partition.py"

MUTATIONS += [
    Mutation(
        "P1",
        "base and insert rows are drawn independently, so they can overlap",
        PARTITION,
        "        drawn = _draw_distinct_rows(rng, 0, total_rows, needed_rows)\n"
        "        base_row_ids = drawn[:num_base]\n"
        "        query_row_ids = drawn[num_base : num_base + num_query_rows]\n"
        "        insert_row_ids = drawn[num_base + num_query_rows :]",
        "        base_row_ids = _draw_distinct_rows(rng, 0, total_rows, num_base)\n"
        "        query_row_ids = _draw_distinct_rows(rng, 0, total_rows, num_query_rows)\n"
        "        insert_row_ids = _draw_distinct_rows(rng, 0, total_rows, max_inserts)",
    ),
    Mutation(
        "P2",
        "prefix selection takes the LAST rows instead of the first",
        PARTITION,
        "        base_row_ids = np.arange(num_base, dtype=np.int64)",
        "        base_row_ids = np.arange(total_rows - num_base, total_rows, dtype=np.int64)",
    ),
    Mutation(
        "P3",
        '"rest" forgets the last remaining row',
        PARTITION,
        "        max_inserts = total_rows - num_base - num_query_rows\n",
        "        max_inserts = total_rows - num_base - num_query_rows - 1\n",
    ),
    Mutation(
        "P4",
        "base rows are returned in random order instead of ascending",
        PARTITION,
        "        base_row_ids=np.sort(base_row_ids),",
        "        base_row_ids=base_row_ids,",
    ),
    Mutation(
        "P5",
        "the seed is ignored",
        PARTITION,
        "    rng = np.random.default_rng(seed)",
        "    rng = np.random.default_rng(12345)",
    ),
    Mutation(
        "P6",
        "insert order is sorted, destroying the random insertion sequence",
        PARTITION,
        "        insert_row_ids=insert_row_ids,",
        "        insert_row_ids=np.sort(insert_row_ids),",
    ),
    Mutation(
        "P7",
        "the capacity check is removed, so oversized requests are accepted",
        PARTITION,
        "    if max_inserts < 0 or needed_rows > total_rows:",
        "    if False:",
    ),
]

GROUNDTRUTH = "semantic_vdb/groundtruth.py"

MUTATIONS += [
    Mutation(
        "G1",
        "inner-product distance has the wrong sign",
        GROUNDTRUTH,
        "        return -(queries @ block.T)",
        "        return queries @ block.T",
    ),
    Mutation(
        "G2", "each dense merge keeps only k-1 winners instead of k (off-by-one in the slice)",
        GROUNDTRUTH,
        "            merged = xp.partition(merged, self.k - 1, axis=1)[:, : self.k]",
        "            merged = xp.partition(merged, self.k - 1, axis=1)[:, : self.k - 1]",
    ),
    Mutation(
        "G3",
        "equal distances prefer the LARGER id (encoder and decoder changed together)",
        GROUNDTRUTH,
        "| global_ids.astype(xp.uint64)[None, :]",
        "| (xp.uint64(0xFFFFFFFF) - global_ids.astype(xp.uint64))[None, :]",
        also=(
            (
                GROUNDTRUTH,
                "    ids = (keys & xp.uint64(0xFFFFFFFF)).astype(xp.int64)",
                "    ids = (xp.uint64(0xFFFFFFFF) - (keys & xp.uint64(0xFFFFFFFF))).astype(xp.int64)",
            ),
        ),
    ),
    Mutation(
        "G4",
        "deleted ids are not removed from the active set",
        GROUNDTRUTH,
        "            alive = ~deleted_mask[ids]",
        "            alive = np.ones(ids.shape[0], dtype=bool)",
    ),
    Mutation(
        "G5",
        "inserted vectors are left out of the active set",
        GROUNDTRUTH,
        "    if num_inserted > 0:\n        segments.append",
        "    if False:\n        segments.append",
    ),
    Mutation(
        "G8",
        "-0.0 is not normalised to +0.0 before encoding",
        GROUNDTRUTH,
        "xp.ascontiguousarray(distances, dtype=xp.float32) + xp.float32(0.0)",
        "xp.ascontiguousarray(distances, dtype=xp.float32)",
    ),
    Mutation(
        "G9", "negative distances are encoded with the wrong bit transform",
        GROUNDTRUTH,
        "    return xp.where(is_negative, ~bits, bits | xp.uint32(0x80000000))",
        "    return xp.where(is_negative, bits, bits | xp.uint32(0x80000000))",
    ),
    Mutation(
        "G10",
        "L2 distances are not clamped at zero",
        GROUNDTRUTH,
        "        return xp.maximum(distances, 0.0)",
        "        return distances",
    ),
    Mutation(
        "G11", "the running best is dropped when a new block is merged",
        GROUNDTRUTH,
        "        merged = xp.concatenate([self._best[chunk_index], keys], axis=1)",
        "        merged = keys",
    ),
]

UPDATES = "semantic_vdb/updates.py"

MUTATIONS += [
    Mutation(
        "U1",
        "inserted vectors never become deletable (deletes come from the base only)",
        UPDATES,
        "        alive_ids = np.concatenate([alive_ids, new_ids])",
        "        alive_ids = alive_ids",
    ),
    Mutation(
        "U2",
        "deleted vectors stay in the candidate list, so they can be deleted again",
        UPDATES,
        "            alive_ids = np.delete(alive_ids, positions)",
        "            pass",
    ),
    Mutation(
        "U3",
        "vectors inserted in a step can already be deleted in that same step",
        UPDATES,
        "            positions = rng.choice(len(alive_ids), size=delete_count, replace=False)\n",
        "            first_id = schedule.num_base + inserted_so_far\n"
        "            alive_ids = np.concatenate(\n"
        "                [alive_ids, np.arange(first_id, first_id + int(schedule.inserts_per_step[step_index]),"
        " dtype=np.int64)]\n"
        "            )\n"
        "            positions = rng.choice(len(alive_ids), size=delete_count, replace=False)\n",
        also=(
            (
                UPDATES,
                "        alive_ids = np.concatenate([alive_ids, new_ids])",
                "        alive_ids = alive_ids if delete_count > 0 else np.concatenate([alive_ids, new_ids])",
            ),
        ),
    ),
    Mutation(
        "U4",
        "the remainder of an uneven split goes to the LAST steps",
        UPDATES,
        "    sizes[:remainder] += 1",
        "    sizes[parts - remainder :] += 1",
    ),
    Mutation(
        "U5",
        "the feasibility check ignores deletions that exceed the alive count",
        UPDATES,
        "        if deletes[step_index] > alive:",
        "        if False:",
    ),
]

PERTURB = "semantic_vdb/perturb.py"
CALIBRATE = "semantic_vdb/calibrate.py"

MUTATIONS += [
    Mutation(
        "N1",
        "N2 uses the WEAKEST r axes instead of the strongest",
        PERTURB,
        "        axes=vt[:num_axes].copy(),",
        "        axes=vt[-num_axes:].copy(),",
    ),
    Mutation(
        "N2",
        "noise is isotropic over all dimensions instead of along the neighbour axes",
        PERTURB,
        "    noise = (draw.axis_noise * (alpha * basis.std)) @ basis.axes\n",
        "    noise = draw.residual_noise * (alpha * float(basis.std[0]))\n",
    ),
    Mutation(
        "N3",
        "alpha is ignored when scaling the noise",
        PERTURB,
        "    noise = (draw.axis_noise * (alpha * basis.std)) @ basis.axes",
        "    noise = (draw.axis_noise * basis.std) @ basis.axes",
    ),
    Mutation(
        "N4",
        "variants are truncated (floor) instead of rounded to the nearest value",
        PERTURB,
        "np.clip(np.rint(vectors), info.min, info.max)",
        "np.clip(np.floor(vectors), info.min, info.max)",
    ),
    Mutation(
        "N5",
        "variants are not clipped to the dtype range (values wrap around)",
        PERTURB,
        "        return np.clip(np.rint(vectors), info.min, info.max).astype(dtype)",
        "        return np.rint(vectors).astype(dtype)",
    ),
    Mutation(
        "N6",
        "the per-axis spread uses the variance instead of the standard deviation",
        PERTURB,
        "        std=np.sqrt(variances[:num_axes]),",
        "        std=variances[:num_axes],",
    ),
    Mutation(
        "N7",
        "the residual option is ignored",
        PERTURB,
        "    if use_residual:\n",
        "    if False:\n",
    ),
    Mutation(
        "N8",
        "the neighbourhood is not centred before the PCA",
        PERTURB,
        "    centred -= centred.mean(axis=0)",
        "    pass",
    ),
    Mutation(
        "A1",
        "the bisection moves in the wrong direction",
        CALIBRATE,
        "        if overlap > target_overlap:\n            low = middle  # still too similar: more noise needed\n        else:\n"
        "            high = middle  # too different: less noise",
        "        if overlap > target_overlap:\n            high = middle  # still too similar: more noise needed\n        else:\n"
        "            low = middle  # too different: less noise",
    ),
    Mutation(
        "A2",
        "calibration accepts the first trial value regardless of the tolerance",
        CALIBRATE,
        "        if abs(overlap - target_overlap) <= tolerance:",
        "        if True:",
    ),
    Mutation(
        "A3",
        "calibration no longer checks that the noise-free overlap reaches the target",
        CALIBRATE,
        "    if measure(low) < target_overlap:",
        "    if False:",
    ),
    Mutation(
        "A4",
        "mean overlap is divided by the number of queries instead of k",
        CALIBRATE,
        "    return shared / neighbours_a.shape[1]",
        "    return shared / neighbours_a.shape[0]",
    ),
]

STREAM = "semantic_vdb/stream.py"

MUTATIONS += [
    Mutation(
        "Z1",
        "Zipf probabilities are not normalised",
        STREAM,
        "    return weights / weights.sum()",
        "    return weights",
    ),
    Mutation(
        "Z2",
        "the exact-repeat diagnostic counts groups instead of vectors",
        STREAM,
        "        exact_repeat_fraction=1.0 - distinct_vectors / stream.length,",
        "        exact_repeat_fraction=1.0 - distinct_groups / stream.length,",
    ),
    Mutation(
        "Z3",
        "members are drawn from a skewed instead of a uniform distribution",
        STREAM,
        "        (rng.random(length) * group_sizes[group_ids]).astype(np.int64),",
        "        (rng.random(length) ** 3 * group_sizes[group_ids]).astype(np.int64),",
    ),
    Mutation(
        "Z4",
        "the pool row ignores the group offset",
        STREAM,
        "    pool_index = offsets[group_ids] + member_ids",
        "    pool_index = member_ids",
    ),
    Mutation(
        "Z5",
        "group popularity follows the order of the groups (no random rank assignment)",
        STREAM,
        "    group_of_rank = rng.permutation(num_groups)",
        "    group_of_rank = np.arange(num_groups)",
    ),
    Mutation(
        "Z6",
        "step boundaries give the remainder to the last steps",
        STREAM,
        "    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)",
        "    return np.concatenate([[0], np.cumsum(sizes[::-1])]).astype(np.int64)",
    ),
    Mutation(
        "Z7",
        "the Zipf exponent is inverted (rare ranks become popular)",
        STREAM,
        "    weights = ranks ** (-exponent)",
        "    weights = ranks ** exponent",
    ),
]

BASE_SET = "semantic_vdb/base_set.py"

MUTATIONS += [
    Mutation(
        "S1",
        "the j-th inserted vector is read from row j+1 of the insert pool",
        BASE_SET,
        "            return pool.read_range(first, count)",
        "            return pool.read_range(first + 1, count)",
    ),
    Mutation(
        "S2",
        "base vectors are written in reverse order relative to base_row_ids",
        BASE_SET,
        "            raw, partition.base_row_ids, partition.query_row_ids, partition.insert_row_ids,",
        "            raw, partition.base_row_ids[::-1], partition.query_row_ids, partition.insert_row_ids,",
    ),
    Mutation(
        "S3",
        "loading a base set no longer verifies the hashes of the id files",
        BASE_SET,
        '        if sha256_file(path) != expected["sha256"]:',
        "        if False:",
    ),
    Mutation(
        "S4",
        "building into an existing base set directory is allowed",
        BASE_SET,
        "    if final_dir.exists():",
        "    if False:",
    ),
    Mutation(
        "S5",
        "a failed build leaves its work directory behind",
        BASE_SET,
        "        shutil.rmtree(work_dir, ignore_errors=True)\n        raise",
        "        raise",
    ),
]

WORKLOAD = "semantic_vdb/workload.py"
SOURCES = "semantic_vdb/sources.py"

MUTATIONS += [
    Mutation(
        "W1",
        "deletions are never applied to the active set used for ground truth",
        WORKLOAD,
        "            deleted_mask[deletions[step - 1]] = True\n",
        "            pass\n",
    ),
    Mutation(
        "W2",
        "insertions are never added to the active set used for ground truth",
        WORKLOAD,
        "            inserted += int(schedule.inserts_per_step[step - 1])",
        "            inserted += 0",
    ),
    Mutation(
        "W3",
        "evaluation ground truth is computed BEFORE the step's updates (violates the rule)",
        WORKLOAD,
        "                # Order inside a step: delete, insert, then the queries of the step.\n",
        "                if computing_shared and step in checkpoints:\n                    eval_fallbacks += eval_answer(inserted, deleted_mask, mix_shared, step)\n",
        also=(
            (
                WORKLOAD,
                "                if computing_shared and step in checkpoints:\n                    eval_fallbacks += eval_answer(inserted, deleted_mask, mix_shared, step)\n                if step % 10",
                "                if step % 10",
            ),
        ),
    ),
    Mutation(
        "W4", "the answers of the stream queries are looked up in reverse order",
        "semantic_vdb/stream_gt.py",
        "            rows = store.find(digests_as_array(row_digests(queries)))\n",
        "            rows = store.find(digests_as_array(row_digests(queries)))[::-1]\n",
    ),
    Mutation(
        "W5",
        "stream step files hold the pool row after the requested one",
        WORKLOAD,
        "                            pool.vectors[stream.pool_index[first:last]],\n",
        "                            pool.vectors[(stream.pool_index[first:last] + 1) % len(pool.vectors)],\n",
    ),
    Mutation(
        "W6",
        "include_original is ignored (the seed is not a member of its group)",
        "semantic_vdb/n2.py",
        "        if job.include_original:\n            parts.append(seed_vector[None, :])",
        "        if False:\n            parts.append(seed_vector[None, :])",
    ),
    Mutation(
        "W7",
        "evaluation ground truth uses the stream k instead of eval_k",
        WORKLOAD,
        "lambda rows: context.eval_queries[rows], eval_k, inserted,",
        "lambda rows: context.eval_queries[rows], stream_k, inserted,",
    ),
    Mutation(
        "W8",
        "the initial-state (step 0) evaluation ground truth is skipped",
        WORKLOAD,
        '            step0_fallbacks = eval_answer(0, no_deletions, store.eval_step0_dir, 0)\n',
        "            step0_fallbacks = 0\n",
    ),
    Mutation(
        "W9",
        "the last step is not an evaluation checkpoint",
        WORKLOAD,
        "    steps = {0, num_steps}",
        "    steps = {0}",
    ),
    Mutation(
        "W10",
        "pool group ids are written in the wrong order",
        WORKLOAD,
        "np.repeat(np.arange(len(pool.group_sizes)), pool.group_sizes),",
        "np.repeat(np.arange(len(pool.group_sizes)), pool.group_sizes)[::-1],",
    ),
    Mutation(
        "X1",
        "num_groups is ignored and every query becomes a seed",
        SOURCES,
        "        chosen = _choose_subset(len(queries), seeds.num_groups, subset_sequence)\n        return SeedSource(seeds.mode, queries, queries[chosen], chosen, None, None)\n\n    if seeds.mode == \"base_rows\":",
        "        chosen = np.arange(len(queries))\n        return SeedSource(seeds.mode, queries, queries[chosen], chosen, None, None)\n\n    if seeds.mode == \"base_rows\":",
    ),
    Mutation(
        "X2",
        "external vectors are not regrouped, so groups are not contiguous",
        SOURCES,
        '        order = np.argsort(group_ids, kind="stable")',
        "        order = np.arange(len(group_ids))",
    ),
    Mutation(
        "X3",
        "query files with another dtype are accepted",
        SOURCES,
        "        if reader.dtype != base_set.dtype:",
        "        if False:",
    ),
    Mutation(
        "X4",
        "query files with another dimension are accepted",
        SOURCES,
        "        if reader.num_cols != base_set.dim:",
        "        if False:",
    ),
]

CONFIG = "semantic_vdb/config.py"
CLI = "semantic_vdb/cli.py"

MUTATIONS += [
    Mutation(
        "C1",
        "num_steps larger than the stream length is accepted",
        CONFIG,
        "    if num_steps > total_requests:",
        "    if False:",
    ),
    Mutation(
        "C2",
        "axes larger than neighbours - 1 are accepted",
        CONFIG,
        "    if groups.axes > groups.neighbours - 1:",
        "    if False:",
    ),
    Mutation(
        "C3",
        "inline comments are no longer stripped from values",
        CONFIG,
        "        inline_comment_prefixes=(\";\",),",
        "        inline_comment_prefixes=(),",
    ),
    Mutation(
        "C4",
        "names with path separators are accepted",
        CONFIG,
        '    if not name or name in (".", "..") or "/" in name or "\\\\" in name:',
        "    if not name:",
    ),
    Mutation(
        "C5",
        "a target overlap of exactly 1.0 is accepted",
        CONFIG,
        "        if any(not 0.0 < value < 1.0 for value in self.target_overlaps):",
        "        if any(not 0.0 < value <= 1.0 for value in self.target_overlaps):",
    ),
    Mutation(
        "C6",
        "external seed mode no longer requires evaluation.source = external",
        CONFIG,
        '    if seeds.mode == "external" and evaluation.source != "external":',
        "    if False:",
    ),
    Mutation(
        "I1",
        "configuration errors are reported with exit code 0",
        CLI,
        "        return 2\n",
        "        return 0\n",
    ),

]

# ---------------------------------------------------------------------------
# update percentages: reads + updates, set names, prefix-sharing, families (round 00065)
# ---------------------------------------------------------------------------

MUTATIONS += [
    Mutation(
        "R1",
        "reads are not reduced by the updates (reads + updates no longer equal the total)",
        UPDATES,
        "        reads=total - updates,\n",
        "        reads=total,\n",
    ),
    Mutation(
        "R2",
        "reads and updates are swapped",
        UPDATES,
        "        reads=total - updates,\n        updates=updates,\n",
        "        reads=updates,\n        updates=total - updates,\n",
    ),
    Mutation(
        "R3",
        "the percentage is divided by 10 instead of 100",
        UPDATES,
        "    exact_updates = Fraction(total) * Fraction(repr(float(update_percent))) / 100",
        "    exact_updates = Fraction(total) * Fraction(repr(float(update_percent))) / 10",
    ),
    Mutation(
        "R4",
        "a split that is not a whole number is truncated instead of rejected",
        UPDATES,
        "    if exact_updates.denominator != 1:",
        "    if False:",
    ),
    Mutation(
        "R5",
        "a percentage of exactly 100 (no reads) is accepted",
        UPDATES,
        "not 0.0 <= update_percent < 100.0:",
        "not 0.0 <= update_percent <= 100.0:",
    ),
    Mutation(
        "R6",
        "the decimal point is dropped from set names",
        UPDATES,
        '    return "u" + text.replace(".", "p")',
        '    return "u" + text.replace(".", "")',
    ),
    Mutation(
        "R7",
        "set names round the percentage to 6 significant digits",
        UPDATES,
        '    text = format(Decimal(repr(float(update_percent))).normalize(), "f")',
        '    text = f"{update_percent:g}"',
    ),
    Mutation(
        "R8",
        "Stream.head returns the last requests instead of the first ones",
        STREAM,
        "            group_ids=self.group_ids[:count],\n"
        "            member_ids=self.member_ids[:count],\n"
        "            pool_index=self.pool_index[:count],\n",
        "            group_ids=self.group_ids[-count:],\n"
        "            member_ids=self.member_ids[-count:],\n"
        "            pool_index=self.pool_index[-count:],\n",
    ),
    Mutation(
        "R9",
        "Stream.head accepts counts beyond the stream length",
        STREAM,
        "        if not 1 <= count <= self.length:",
        "        if not 1 <= count:",
    ),
    Mutation(
        "R10",
        "settings of the previous version are silently ignored",
        CONFIG,
        "    if parser.has_option(section, key):\n        raise ConfigError(f\"[{section}] {key} is no longer supported",
        "    if False:\n        raise ConfigError(f\"[{section}] {key} is no longer supported",
    ),
    Mutation(
        "R11",
        "a percentage that leaves fewer reads than steps is accepted",
        CONFIG,
        "            if mix.reads < self.stream.num_steps:",
        "            if False:",
    ),
    Mutation(
        "R12",
        "the same percentage may be listed twice",
        CONFIG,
        "            if mix.name in names:",
        "            if False:",
    ),
    Mutation(
        "R13",
        "the shared Zipf streams are drawn at the shortest instead of the longest read count",
        WORKLOAD,
        "    longest_reads = max(mix.reads for mix in mixes)",
        "    longest_reads = min(mix.reads for mix in mixes)",
    ),
    Mutation(
        "R14",
        "every set uses the whole shared stream instead of its prefix",
        WORKLOAD,
        "name: master.head(mix.reads) for name, master in p.master_streams.items()",
        "name: master for name, master in p.master_streams.items()",
    ),
    Mutation(
        "R15",
        "deletions are one fewer than insertions",
        WORKLOAD,
        "                mix.updates,\n                mix.updates,\n                config.stream.num_steps,",
        "                mix.updates,\n                mix.updates - 1,\n                config.stream.num_steps,",
    ),
    Mutation(
        "R16",
        "a read-only set still scores the evaluation queries at every checkpoint",
        WORKLOAD,
        "    if mix.updates == 0:",
        "    if False:",
    ),
    Mutation(
        "R17",
        "all sets of a pool are written into the same directory",
        WORKLOAD,
        "context.work_dir / p.spec.name / mix.name for p in context.pools}",
        "context.work_dir / p.spec.name for p in context.pools}",
    ),
    Mutation(
        "R18",
        "every set reuses the update schedule of the first set",
        WORKLOAD,
        "            schedule = context.schedules[mix.name]\n",
        "            schedule = context.schedules[context.mixes[0].name]\n",
    ),
    Mutation(
        "R19",
        "the manifest records the wrong number of insertions",
        WORKLOAD,
        '            "insertions": schedule.total_inserts,',
        '            "insertions": mix.reads,',
    ),
    Mutation(
        "R20",
        "a failed family build leaves its partial directory behind",
        WORKLOAD,
        "    else:\n        shutil.rmtree(work_dir, ignore_errors=True)\n\n\ndef build_workload(",
        "    else:\n        pass\n\n\ndef build_workload(",
    ),
    Mutation(
        "R21",
        "the planned schedule is not checked against the insert pool size",
        WORKLOAD,
        "                max_inserts=base_set.num_inserts,\n            )\n        except ValueError",
        "                max_inserts=None,\n            )\n        except ValueError",
    ),

]

# ---------------------------------------------------------------------------
# duplicate-free uniform stream, variants = none, documented example (round 00066)
# ---------------------------------------------------------------------------

EXAMPLE_INI = "../config/workload.example.ini"

MUTATIONS += [
    Mutation(
        "V1",
        "the uniform_unique stream is drawn WITH replacement (exact repeats appear)",
        STREAM,
        "rng.choice(pool_size, size=length, replace=False)",
        "rng.choice(pool_size, size=length, replace=True)",
    ),
    Mutation(
        "V2",
        "a uniform_unique stream longer than the pool is no longer rejected by the builder",
        STREAM,
        "    if length > pool_size:",
        "    if False:",
    ),
    Mutation(
        "V3",
        "pool rows on a group boundary are assigned to the previous group",
        STREAM,
        'np.searchsorted(offsets, pool_index, side="right") - 1',
        'np.searchsorted(offsets, pool_index, side="left") - 1',
    ),
    Mutation(
        "V4",
        "the pool-size check before the expensive work is removed",
        WORKLOAD,
        "    too_small = [spec for spec in specs if spec.size < longest_reads]",
        "    too_small = []",
    ),
    Mutation(
        "V5",
        "variants = none still builds N2 variants and calibrates",
        WORKLOAD,
        '        elif spec.kind == "none":',
        "        elif False:",
    ),
    Mutation(
        "V6",
        "variants = none puts the given queries into the pool in reverse order",
        WORKLOAD,
        "            vectors = source.seed_vectors\n",
        "            vectors = source.seed_vectors[::-1]\n",
    ),
    Mutation(
        "V7",
        "variants = none accepts a group size other than 1",
        CONFIG,
        '        if self.variants == "none" and tuple(self.group_sizes) != (1,):',
        "        if False:",
    ),
    Mutation(
        "V8",
        "a configuration without any stream is accepted",
        CONFIG,
        "        if not self.zipf_exponents and not self.uniform_unique:",
        "        if False:",
    ),
    Mutation(
        "V9",
        "uniform_unique = true does not create the stream",
        WORKLOAD,
        "    if config.stream.uniform_unique:\n        rng = np.random.default_rng(",
        "    if False:\n        rng = np.random.default_rng(",
    ),
    Mutation(
        "V10",
        "the manifest labels every stream as zipf",
        WORKLOAD,
        '            "kind": stream.kind,',
        '            "kind": "zipf",',
    ),
    Mutation(
        "V11",
        "Stream.head forgets the stream kind",
        STREAM,
        "            kind=self.kind,\n",
        "",
    ),
    Mutation(
        "V12",
        "the example no longer documents overlap_tolerance",
        EXAMPLE_INI,
        "; overlap_tolerance = 0.02\n",
        "",
    ),
    Mutation(
        "V13",
        "the example no longer documents query_chunk_rows",
        EXAMPLE_INI,
        "; query_chunk_rows = auto\n",
        "",
    ),
    Mutation(
        "V14",
        "the example no longer documents group_id_file",
        EXAMPLE_INI,
        "; group_id_file = /path/to/groups.ibin\n",
        "",
    ),
    Mutation(
        "V15",
        "the example no longer documents variants",
        EXAMPLE_INI,
        "; variants = n2\n",
        "",
    ),
    Mutation(
        "V16",
        "the example documents a key under the wrong section",
        EXAMPLE_INI,
        "; eval_k = 100\n",
        "",
        also=((EXAMPLE_INI, "; stream_k = 120\n", "; stream_k = 120\n; eval_k = 100\n"),),
    ),

]

# ---------------------------------------------------------------------------
# universe-list ground truth, group-size axis, resources, parallel workers (round 00067)
# ---------------------------------------------------------------------------

UNIVERSE = "semantic_vdb/universe.py"
PARALLEL = "semantic_vdb/parallel.py"
GT_ENGINE = "semantic_vdb/groundtruth.py"
LAYOUT = "semantic_vdb/layout.py"
N2_MODULE = "semantic_vdb/n2.py"

MUTATIONS += [
    Mutation(
        "Y1",
        "the list length is one too small",
        UNIVERSE,
        "    return _bisect_length(enough, low, high)\n",
        "    return _bisect_length(enough, low, high) - 1\n",
    ),
    Mutation(
        "Y2",
        "the chance of a short list is computed as its complement",
        UNIVERSE,
        "    return min(total, 1.0)\n",
        "    return max(1.0 - total, 0.0)\n",
    ),
    Mutation(
        "Y3",
        "a vector that is inserted LATER already counts as alive (off by one)",
        UNIVERSE,
        "    alive = list_ids < alive_limit\n",
        "    alive = list_ids <= alive_limit\n",
    ),
    Mutation(
        "Y4",
        "deleted vectors stay in the answer",
        UNIVERSE,
        "    alive &= ~deleted_mask[list_ids]\n",
        "    alive &= np.ones_like(alive)\n",
    ),
    Mutation(
        "Y5",
        "the alive entries are picked with an unstable sort (order within ties of the key is lost)",
        UNIVERSE,
        'order = np.argsort(~alive, axis=1, kind="stable")[:, :k]',
        'order = np.argsort(~alive, axis=1, kind="quicksort")[:, :k]',
    ),
    Mutation(
        "Y6",
        "a list with exactly k alive entries is treated as too short",
        UNIVERSE,
        "    enough = alive.sum(axis=1) >= k\n",
        "    enough = alive.sum(axis=1) > k\n",
    ),
    Mutation(
        "Y7",
        "the scan of the universe leaves out the inserted vectors",
        UNIVERSE,
        "base, insert_pool if num_inserts > 0 else None, num_inserts, nothing_deleted, block_rows",
        "base, None, 0, nothing_deleted, block_rows",
    ),
    Mutation(
        "Y8",
        "a worker stores the answers of its queries in reverse order",
        UNIVERSE,
        "                list_ids[first:last] = result.neighbour_ids\n",
        "                list_ids[first:last] = result.neighbour_ids[::-1]\n",
    ),
    Mutation(
        "Y12", "a request whose list is too short is not answered by scanning (the unusable row is returned)",
        UNIVERSE,
        "        missing = np.flatnonzero(~enough)\n",
        "        missing = np.flatnonzero(~enough)[:0]\n",
    ),
    Mutation(
        "Y13", "the scan of the active set ignores the insertions of the step",
        UNIVERSE,
        "            result = self._scanner.top(unique_vectors, k, inserted, deleted_mask)\n",
        "            result = self._scanner.top(unique_vectors, k, 0, deleted_mask)\n",
    ),
    Mutation(
        "Y14", "the step's insertions are alive one step too early",
        UNIVERSE,
        "        alive_limit = self._num_base + inserted\n",
        "        alive_limit = self._num_base + inserted + 1\n",
    ),
    Mutation(
        "Y15",
        "a step answers with the deletions of the previous step",
        WORKLOAD,
        "                deleted_mask[deletions[step - 1]] = True\n",
        "                deleted_mask[deletions[max(step - 2, 0)]] = True\n",
    ),
    Mutation(
        "Y16",
        "the universe holds the insertions of the SMALLEST set only",
        WORKLOAD,
        "universe_inserts = max(schedule.total_inserts for schedule in schedules.values())",
        "universe_inserts = min(schedule.total_inserts for schedule in schedules.values())",
    ),
    Mutation(
        "Y17",
        "the evaluation lists are as short as the stream lists",
        WORKLOAD,
        "    eval_list_length = choose_list_length(config.evaluation.eval_k,",
        "    eval_list_length = choose_list_length(gt.stream_k,",
    ),
    Mutation(
        "Y18",
        "every set links the pool of the first pool directory",
        WORKLOAD,
        '                link_tree(context.shared_dir / f"pool_{pool.spec.key}", set_dir)\n',
        '                link_tree(context.shared_dir / f"pool_{context.pools[0].spec.key}", set_dir)\n',
    ),
    Mutation(
        "Y19",
        "the deleted ids of a percentage are not linked into the sets",
        WORKLOAD,
        "                link_tree(mix_shared, set_dir)\n",
        "                pass\n",
    ),
    Mutation(
        "Y20",
        "the shared step-0 evaluation ground truth is not linked into the sets",
        WORKLOAD,
        "                link_tree(link_store.eval_step0_dir, set_dir)\n",
        "                pass\n",
    ),
    Mutation(
        "Y21",
        "family.json writes set paths as <percentage>/<pool>",
        WORKLOAD,
        '"path": f"{record.pool.spec.name}/{record.mix.name}",',
        '"path": f"{record.mix.name}/{record.pool.spec.name}",',
    ),
    Mutation(
        "Y22",
        "a core is counted for every logical CPU (hyper-threads are not merged)",
        PARALLEL,
        "        cores.setdefault(siblings, set()).add(cpu)\n",
        "        cores.setdefault(frozenset({cpu}), set()).add(cpu)\n",
    ),
    Mutation(
        "Y23",
        "a GPU that is in use can still be named",
        PARALLEL,
        "        if gpu.used_mb >= max_used_mb:\n",
        "        if False:\n",
    ),
    Mutation(
        "Y24",
        "the batch size budget ignores the workers that share the GPU",
        PARALLEL,
        "free_bytes * GPU_MEMORY_FRACTION / max(workers, 1)",
        "free_bytes * GPU_MEMORY_FRACTION",
    ),
    Mutation(
        "Y25",
        "the process is not pinned to its NUMA node",
        PARALLEL,
        "        pin(cpus)\n",
        "        pass\n",
    ),
    Mutation(
        "Y26",
        "PCI bus ids keep the long domain and no longer match across tools",
        PARALLEL,
        '    return f"{domain[-4:]}:{rest}"',
        '    return f"{domain}:{rest}"',
    ),
    Mutation(
        "Y27",
        "spawned workers keep the default number of BLAS threads",
        PARALLEL,
        "    values = dict(SINGLE_THREAD_ENVIRONMENT)\n",
        "    values = {}\n",
    ),
    Mutation(
        "Y28",
        "the environment of a pool is not restored in the parent",
        PARALLEL,
        "        for name, value in saved.items():\n            if value is None:\n                os.environ.pop(name, None)\n            else:\n                os.environ[name] = value\n",
        "        pass\n",
    ),
    Mutation(
        "Y29",
        "a block that does not fit is split but its second half is forgotten",
        GT_ENGINE,
        "            self._merge_into_chunk(chunk_index, ids[half:], block[half:])\n",
        "            pass\n",
    ),
    Mutation(
        "Y30",
        "the prefetch drops the last block",
        GT_ENGINE,
        "            for ids, vectors in blocks:\n                converted",
        "            for ids, vectors in list(blocks)[:-1]:\n                converted",
    ),
    Mutation(
        "Y31",
        "every seed draws its variants from the random stream of the first seed of its worker",
        N2_MODULE,
        "np.random.SeedSequence([job.seed, 1, index])",
        "np.random.SeedSequence([job.seed, 1, job.start])",
    ),
    Mutation(
        "Y32",
        "the basis of a seed is stored with its axes in reverse order",
        N2_MODULE,
        "            axes_out[index] = basis.axes\n",
        "            axes_out[index] = basis.axes[::-1]\n",
    ),
    Mutation(
        "Y33",
        "the same group size may be listed twice",
        CONFIG,
        "        if len(set(self.group_sizes)) != len(self.group_sizes):",
        "        if False:",
    ),
    Mutation(
        "Y34",
        "an automatic batch size becomes a fixed number instead of being chosen at run time",
        CONFIG,
        '    return None if value == "auto" else value',
        '    return 100000 if value == "auto" else value',
    ),
    Mutation(
        "Y35",
        "the pool directory name uses the wrong group size",
        LAYOUT,
        '    return f"g{group_size}"',
        '    return f"g{group_size + 1}"',
    ),
    Mutation(
        "Y36",
        "a hard link is made to the wrong file",
        LAYOUT,
        "        os.link(source, destination)\n",
        "        os.link(source.parent / (source.name + 'x'), destination)\n",
    ),
    Mutation(
        "Y37",
        "files linked to each other are all hashed from their first name only (wrong size reported)",
        LAYOUT,
        "        result[path] = (info.st_size, hashes[(info.st_dev, info.st_ino)])",
        "        result[path] = (info.st_size + 1, hashes[(info.st_dev, info.st_ino)])",
    ),
    Mutation(
        "Y38",
        "the bounded writer drops the error of a failed job",
        LAYOUT,
        "            for future in self._pending:\n                future.result()\n",
        "            pass\n",
    ),
]

RESUME = "semantic_vdb/resume.py"
N2 = "semantic_vdb/n2.py"

MUTATIONS += [
    Mutation(
        "Q1",
        "a finished stage is never recognised, so an interrupted build redoes everything",
        RESUME,
        '        return (self._done / f"{self._checked(stage)}.json").is_file()',
        "        return False",
    ),
    Mutation(
        "Q2",
        "a stored value that differs from the present one is accepted",
        RESUME,
        "            if stored != present:",
        "            if False:",
    ),
    Mutation(
        "Q3",
        "an interrupted build deletes the stages it finished",
        WORKLOAD,
        "    if resume.any_progress():\n        # Keep what is finished;",
        "    if False:\n        # Keep what is finished;",
    ),
    Mutation(
        "Q4",
        "a build that finished no stage leaves its .building directory behind",
        WORKLOAD,
        "    if resume.any_progress():\n        # Keep what is finished;",
        "    if True:\n        # Keep what is finished;",
    ),
    Mutation(
        "Q5",
        "the query digests of an unfinished build are not compared",
        WORKLOAD,
        '            resume.check_or_set("inputs", planned.inputs)',
        "            pass",
    ),
    Mutation(
        "Q6",
        "a finished query group is scanned again",
        UNIVERSE,
        "                if marker is not None and marker.exists():\n"
        "                    continue  # finished by an earlier run of this build",
        "                if False:\n"
        "                    continue  # finished by an earlier run of this build",
    ),
    Mutation(
        "Q7",
        "the marker of a query group names the wrong range",
        UNIVERSE,
        '                marker = Path(job.done_dir) / f"{first}_{last}.done" if job.done_dir else None',
        '                marker = Path(job.done_dir) / f"{first}_{last + 1}.done" if job.done_dir else None',
    ),
    Mutation(
        "Q8",
        "list files of an earlier attempt are reused even when they are missing or have another shape",
        UNIVERSE,
        "        keep_rows = _list_files_fit(ids_path, dists_path, count, list_length)",
        "        keep_rows = True",
    ),
    Mutation(
        "Q9",
        "the N2 bases and alpha are computed again although they were finished",
        N2,
        '    if resume is not None and resume.done("n2_bases"):',
        "    if False:",
    ),
    Mutation(
        "Q10",
        "the exact neighbours of the seeds are searched again although they were finished",
        N2,
        '        if resume is not None and resume.done("seed_neighbours"):',
        "        if False:",
    ),
    Mutation(
        "Q11",
        "a finished pool is made again",
        WORKLOAD,
        "            if resume is not None and resume.done(stage):\n"
        '                log(f"[resume] pool {spec.name} and its statistics were finished earlier")',
        "            if False:\n"
        '                log(f"[resume] pool {spec.name} and its statistics were finished earlier")',
    ),
    Mutation(
        "Q12",
        "a finished update set is built again",
        WORKLOAD,
        "            if resume is not None and resume.done(stage):\n"
        '                context.log(f"[resume] set {mix.name}: finished earlier, kept")',
        "            if False:\n"
        '                context.log(f"[resume] set {mix.name}: finished earlier, kept")',
    ),
    Mutation(
        "Q13",
        "the files shared by the sets are written again after an interruption (their hard links break)",
        WORKLOAD,
        '    if resume is None or not resume.done("common"):',
        "    if True:",
    ),
    Mutation(
        "Q14",
        "what an interrupted attempt left of an unfinished set is not removed",
        WORKLOAD,
        "                    shutil.rmtree(context.work_dir / pool.spec.name / mix.name, ignore_errors=True)\n",
        "                    pass\n",
    ),
]

MUTATIONS += [
    Mutation(
        "S6",
        "an inserted vector is stored at its place in raw order instead of its place in the insertion order",
        BASE_SET,
        "                        inserts[order[i_lo:i_hi]] = block[sorted_inserts[i_lo:i_hi] - span_first]",
        "                        inserts[np.arange(i_lo, i_hi)] = block[sorted_inserts[i_lo:i_hi] - span_first]",
    ),
    Mutation(
        "S7",
        "a stretch of the raw file that holds only insertion rows is skipped",
        BASE_SET,
        "                if wanted:\n                    # Read only the span",
        "                if wanted and b_hi > b_lo:\n                    # Read only the span",
    ),
    Mutation(
        "S8",
        "the span that is read ends one row before the last wanted row",
        BASE_SET,
        "                    span_last = int(max(ids[-1] for ids in wanted)) + 1",
        "                    span_last = int(max(ids[-1] for ids in wanted))",
    ),
    Mutation(
        "S9",
        "consecutive wanted rows lose their last row",
        BASE_SET,
        "        return block[first - block_first : last - block_first + 1]",
        "        return block[first - block_first : last - block_first]",
    ),
    Mutation(
        "S10",
        "the temporary file of a large insertion set is left behind",
        BASE_SET,
        "            temporary.unlink(missing_ok=True)",
        "            pass",
    ),
    Mutation(
        "S11",
        "the sweep no longer checks that the base row ids ascend",
        BASE_SET,
        '        if len(ids) > 1 and not bool((np.diff(ids) > 0).all()):',
        "        if False:",
    ),
    Mutation(
        "S12",
        "the insertion rows of a stretch are taken from the wrong end of the sorted insertion ids",
        BASE_SET,
        "                i_lo, i_hi = np.searchsorted(sorted_inserts, [chunk_first, chunk_last])",
        "                i_lo, i_hi = np.searchsorted(sorted_inserts, [chunk_first, chunk_last + 1])",
    ),
    Mutation(
        "B8",
        "read_block ignores the 8-byte header",
        BINIO,
        "        offset = HEADER_BYTES + start * self.num_cols * self.dtype.itemsize",
        "        offset = start * self.num_cols * self.dtype.itemsize",
    ),
    Mutation(
        "B9",
        "read_block does not check its range",
        BINIO,
        "        if start < 0 or count < 0 or start + count > self.num_rows:\n"
        "            raise IndexError(\n"
        '                f"{self.path}: range [{start}, {start + count}) is outside 0..{self.num_rows}"\n'
        "            )\n"
        "        block = np.empty",
        "        block = np.empty",
    ),
]

# ---------------------------------------------------------------------------
# round 00073: the target overlap as an axis, the shared/ results, extending a family
# ---------------------------------------------------------------------------

SHARED_STORE = "semantic_vdb/shared_store.py"
FAMILY = "semantic_vdb/family.py"
EXTEND = "semantic_vdb/extend.py"
STREAM_GT_FILE = "semantic_vdb/stream_gt.py"

MUTATIONS += [
    # ---- names and configuration ------------------------------------------------------
    Mutation(
        "L1", "the overlap directory name forgets that the overlap is a fraction (x10, not x100)",
        LAYOUT,
        "    percent = Decimal(repr(float(target_overlap))) * 100",
        "    percent = Decimal(repr(float(target_overlap))) * 10",
    ),
    Mutation(
        "L2", "the overlap directory name drops the decimal point (o12p5 becomes o125)",
        LAYOUT,
        '    return "o" + format(percent.normalize(), "f").replace(".", "p")',
        '    return "o" + format(percent.normalize(), "f").replace(".", "")',
    ),
    Mutation(
        "L3", "an N2 pool directory has no overlap level",
        LAYOUT,
        '        return f"{overlap_name}/g{group_size}"',
        '        return f"g{group_size}"',
    ),
    Mutation(
        "L4", "a target overlap outside (0, 1) is accepted",
        CONFIG,
        "        if any(not 0.0 < value < 1.0 for value in self.target_overlaps):",
        "        if False:",
    ),
    Mutation(
        "L5", "two target overlaps with the same directory name are accepted",
        CONFIG,
        "        if len(set(names)) != len(names):",
        "        if False:",
    ),
    Mutation(
        "L6", "the .ini loader keeps only the first target overlap of a list",
        CONFIG,
        '        target_overlaps=_real_list(parser, "groups", "target_overlap", "0.8"),',
        '        target_overlaps=_real_list(parser, "groups", "target_overlap", "0.8")[:1],',
    ),
    Mutation(
        "L7", "an unknown [workload] existing policy is accepted",
        CONFIG,
        "        if self.existing not in EXISTING_POLICIES:",
        "        if False:",
    ),
    Mutation(
        "L8", "the default of [workload] existing is extend (it must be error)",
        CONFIG,
        'existing=_choice(parser, "workload", "existing", EXISTING_POLICIES, default="error"),',
        'existing=_choice(parser, "workload", "existing", EXISTING_POLICIES, default="extend"),',
    ),
    # ---- the noise scale per overlap -------------------------------------------------------
    Mutation(
        "O1", "every pool uses the noise scale of the first overlap",
        WORKLOAD,
        "            calibration = calibrations[spec.overlap_name]\n",
        "            calibration = next(iter(calibrations.values()))\n",
    ),
    Mutation(
        "O2", "a stored noise scale is used although it was found with other settings",
        N2,
        "    if stored != settings:",
        "    if False:",
    ),
    Mutation(
        "O3", "a stored noise scale is ignored: alpha is searched again",
        N2,
        "    if stored is not None:\n        log(f\"[groups] using the stored noise scale",
        "    if False:\n        log(f\"[groups] using the stored noise scale",
    ),
    Mutation(
        "O4", "the noise scale of a new overlap is not recorded in cache.json",
        WORKLOAD,
        "                store.set_calibrations({spec.overlap_name: calibration.to_info(settings)})",
        "                pass",
    ),
    Mutation(
        "O5", "the settings a noise scale depends on omit the residual flag",
        N2,
        '        "residual": config.groups.residual,\n        "seed": config.seeds.seed,',
        '        "residual": False,\n        "seed": config.seeds.seed,',
    ),
    Mutation(
        "O6", "the statistics of a pool report the first target overlap of the configuration",
        N2,
        '        "target_overlap": calibration.target_overlap,',
        '        "target_overlap": config.groups.target_overlaps[0],',
    ),
    Mutation(
        "O8", "a noise scale that was found before the interruption is searched again",
        N2,
        "    if resume is not None and resume.done(stage):\n        log(f\"[resume] the noise scale of",
        "    if False:\n        log(f\"[resume] the noise scale of",
    ),
    Mutation(
        "O7", "a later overlap searches its noise scale with the overlap of the first one",
        N2,
        "        log(f\"[groups] calibrating alpha for target overlap {target_overlap} ({overlap_name})\")\n        result = find_alpha(overlap_at, target_overlap, groups.overlap_tolerance)",
        "        log(f\"[groups] calibrating alpha for target overlap {target_overlap} ({overlap_name})\")\n        result = find_alpha(overlap_at, 0.8, groups.overlap_tolerance)",
    ),
    # ---- pools, sets and the family index of a joint build -----------------------------
    Mutation(
        "W16", "the pools are ordered size by size instead of overlap by overlap",
        WORKLOAD,
        "    for target in config.groups.target_overlaps:\n        overlap_name = overlap_dir_name(target)\n        for size in config.groups.group_sizes:\n            specs.append(",
        "    for size in config.groups.group_sizes:\n        for target in config.groups.target_overlaps:\n            overlap_name = overlap_dir_name(target)\n            specs.append(",
    ),
    Mutation(
        "W19", "workload.json keeps the whole list of target overlaps of the build",
        WORKLOAD,
        "        description[\"groups\"][\"target_overlaps\"] = (spec.target_overlap,)",
        "        pass",
    ),
    Mutation(
        "W20", "workload.json records the existing policy of the run",
        WORKLOAD,
        "    description.pop(\"existing\", None)\n    for key in STREAM_STAGE_SETTINGS:",
        "    for key in STREAM_STAGE_SETTINGS:",
    ),
    Mutation(
        "W21", "a family index of a new family has no common fingerprint",
        WORKLOAD,
        "            index[\"common\"] = common_fingerprint(config, base_set, planned.inputs)\n",
        "            index[\"common\"] = {}\n",
    ),
    Mutation(
        "W22", "an extension writes family.json instead of the fragment",
        WORKLOAD,
        "        if existing is None:\n            index[\"common\"]",
        "        if True:\n            index[\"common\"]",
    ),
    Mutation(
        "W23", "an extension builds the pools that the family already has",
        WORKLOAD,
        "            specs = [spec for spec in specs if spec.name not in existing.pool_names]",
        "            specs = specs",
    ),
    Mutation(
        "W24", "a set of an extension takes its evaluation answers from this build, not from the family",
        WORKLOAD,
        "            mix_shared = link_store.mix_dir(mix.name)  # files every pool's set links to",
        "            mix_shared = store.mix_dir(mix.name)  # files every pool's set links to",
    ),
    Mutation(
        "W25", "the scan counts of the evaluation answers of an extension are reported as 0",
        WORKLOAD,
        "                eval_fallbacks = int(read_json(info_path)[\"eval_requests_answered_by_scan\"])",
        "                eval_fallbacks = 0",
    ),
    Mutation(
        "W26", "a new family keeps the queries outside shared/common",
        WORKLOAD,
        "        common = context.store.common_dir\n",
        "        common = context.shared_dir / \"common\"\n",
    ),
    # ---- the stored neighbours and bases --------------------------------------------------
    Mutation(
        "S20", "stored neighbours are not used by an extension",
        N2,
        "    if reuse is not None and reuse.has_n2():",
        "    if False:",
    ),
    Mutation(
        "S21", "stored neighbours of an extension are used without checking them",
        N2,
        "            reuse.verify_n2(fingerprint)\n",
        "            pass\n",
    ),
    Mutation(
        "S22", "the stored files are checked for their size but not for their sha256",
        SHARED_STORE,
        '            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:',
        '            if path.stat().st_size != expected["bytes"]:',
    ),
    Mutation(
        "S23", "stored neighbours made for other inputs are accepted",
        SHARED_STORE,
        "        if stored != canonical(fingerprint):",
        "        if False:",
    ),
    Mutation(
        "S24", "a missing stored file is not reported as such",
        SHARED_STORE,
        "            if not path.is_file():\n                raise SharedStoreError(f\"{path} is missing\")",
        "            if False:\n                raise SharedStoreError(f\"{path} is missing\")",
    ),
    Mutation(
        "S25", "writing the manifest of the neighbours forgets the stored noise scales",
        SHARED_STORE,
        '        cache = self.read_cache() or {"format_version": SHARED_FORMAT_VERSION, "calibrations": {}}',
        '        cache = {"format_version": SHARED_FORMAT_VERSION, "calibrations": {}}',
    ),
    Mutation(
        "S26", "storing a noise scale replaces the ones that are already stored",
        SHARED_STORE,
        "        merged.update(canonical(calibrations))",
        "        merged = canonical(calibrations)",
    ),
    Mutation(
        "S27", "the fingerprint of the neighbours ignores the seed vectors",
        SHARED_STORE,
        '        "seeds_sha256": array_digest(seeds),',
        '        "seeds_sha256": "",',
    ),
    Mutation(
        "S28", "the fingerprint of the neighbours ignores the number of PCA axes",
        SHARED_STORE,
        '        "axes": config.groups.axes,',
        '        "axes": 0,',
    ),
    # ---- what a family and its extension have in common ----------------------------------------
    Mutation(
        "F1", "the existing policy is part of the common fingerprint",
        FAMILY,
        '_IGNORED_FOR_COMMON = ("parallel", "existing", "name", "base_set_dir")',
        '_IGNORED_FOR_COMMON = ("parallel", "name", "base_set_dir")',
    ),
    Mutation(
        "F2", "the target overlaps are part of the common fingerprint",
        FAMILY,
        '_POOL_AXES = ("group_sizes", "target_overlaps")',
        '_POOL_AXES = ("group_sizes",)',
    ),
    Mutation(
        "F3", "the group sizes are part of the common fingerprint",
        FAMILY,
        '_POOL_AXES = ("group_sizes", "target_overlaps")',
        '_POOL_AXES = ("target_overlaps",)',
    ),
    Mutation(
        "F4", "the update percentages and seeds may differ in an extension",
        FAMILY,
        '_IGNORED_FOR_COMMON = ("parallel", "existing", "name", "base_set_dir")',
        '_IGNORED_FOR_COMMON = ("parallel", "existing", "name", "base_set_dir", "updates")',
    ),
    Mutation(
        "F10", "the path of a query file is part of the common fingerprint",
        FAMILY,
        '_PATH_SETTINGS = {"seeds": ("query_file", "group_id_file"), "evaluation": ("eval_file",)}',
        "_PATH_SETTINGS = {}",
    ),
    Mutation(
        "F11", "the directory of the base set is part of the common fingerprint",
        FAMILY,
        '_IGNORED_FOR_COMMON = ("parallel", "existing", "name", "base_set_dir")',
        '_IGNORED_FOR_COMMON = ("parallel", "existing", "name")',
    ),
    Mutation(
        "F5", "an extension that differs in a shared setting is not refused",
        FAMILY,
        "    if stored != current:",
        "    if False:",
    ),
    Mutation(
        "F6", "a family of another format is opened for extension",
        FAMILY,
        "    if version != FAMILY_FORMAT_VERSION:",
        "    if False:",
    ),
    Mutation(
        "F7", "the merged index lists an overlap twice when a pool is added to an existing overlap",
        FAMILY,
        "        entry for entry in fragment[\"overlaps\"] if entry[\"name\"] not in known",
        "        entry for entry in fragment[\"overlaps\"]",
    ),
    Mutation(
        "F8", "the merged index keeps the pools in the order they were added",
        FAMILY,
        "    pools_in_order = sorted(merged[\"pools\"], key=pool_key)  # stable: unknown ones stay in order",
        "    pools_in_order = list(merged[\"pools\"])",
    ),
    Mutation(
        "F9", "the merged index keeps the sets in the order they were added",
        FAMILY,
        "    merged[\"sets\"] = sorted(\n        merged[\"sets\"],",
        "    merged[\"sets\"] = list(\n        merged[\"sets\"],",
    ),
    # ---- extending a family ----------------------------------------------------------------------
    Mutation(
        "E1", "the pools that exist are built again",
        EXTEND,
        "    missing = [name for name in wanted if name not in existing.pool_names]",
        "    missing = wanted",
    ),
    Mutation(
        "E2", "the report does not name the pools that were skipped",
        EXTEND,
        "    skipped = [name for name in wanted if name in existing.pool_names]",
        "    skipped = []",
    ),
    Mutation(
        "E3", "a run with nothing to add builds anyway",
        EXTEND,
        "        if not missing:\n",
        "        if False:\n",
    ),
    Mutation(
        "E4", "the lock of an extension is never released",
        EXTEND,
        "    finally:\n        lock.release()",
        "    finally:\n        pass",
    ),
    Mutation(
        "E5", "every lock counts as the lock of a running process",
        EXTEND,
        "    except ProcessLookupError:\n        return False",
        "    except ProcessLookupError:\n        return True",
    ),
    Mutation(
        "E6", "the lock of a running process is replaced",
        EXTEND,
        "                if owner is not None and _process_exists(owner):",
        "                if False:",
    ),
    Mutation(
        "E7", "family.json is replaced before the pools are moved in",
        EXTEND,
        "    # 1. The new pool directories (o<overlap>/g<size>, with all their sets).\n",
        "    _write_text(final_dir / \"family.json\", journal[\"family_json\"])\n",
    ),
    Mutation(
        "E8", "a pool directory that is in the way is overwritten or reported as another error",
        EXTEND,
        "    if destination.exists():\n        raise WorkloadError(\n            f\"{destination} already exists although",
        "    if False:\n        raise WorkloadError(\n            f\"{destination} already exists although",
    ),
    Mutation(
        "E9", "the commit is not described in a journal before it starts",
        EXTEND,
        "    write_text_durable(work_dir / JOURNAL_FILE, json_text(journal))\n",
        "",
    ),
    Mutation(
        "E10", "an unfinished commit is not completed by the next run",
        EXTEND,
        "        if (work_dir / JOURNAL_FILE).is_file():\n            # An earlier run built",
        "        if False:\n            # An earlier run built",
    ),
    Mutation(
        "E11", "a repeated move of a pool that was already moved is an error",
        EXTEND,
        "        if destination.exists():\n            return\n",
        "        if destination.exists():\n            raise WorkloadError(\"already moved\")\n",
    ),
    Mutation(
        "E12", "every extension writes the same report file",
        EXTEND,
        "    number = 1 + len(list(final_dir.glob(\"build_report_extension_*.json\")))",
        "    number = 1",
    ),
    Mutation(
        "E13", "the noise scales of the new overlaps are not merged into the family's shared results",
        EXTEND,
        "    calibrations.update(staged.calibrations())",
        "    pass",
    ),
    Mutation(
        "E14", "neighbours that an extension had to compute are not kept in the family",
        EXTEND,
        "    move_n2 = \"n2\" not in cache and staged.has_n2()",
        "    move_n2 = False",
    ),
    Mutation(
        "E15", "an unfinished extension is continued although the family changed since",
        EXTEND,
        "            \"family_json_sha256\": sha256_file(final_dir / \"family.json\"),",
        "            \"family_json_sha256\": \"\",",
    ),
    Mutation(
        "E16", "the working directory of an extension is left behind after the commit",
        EXTEND,
        "    _apply_journal(work_dir, final_dir, journal, log)\n    shutil.rmtree(work_dir, ignore_errors=True)\n\n\ndef _finish_unfinished_commit",
        "    _apply_journal(work_dir, final_dir, journal, log)\n\n\ndef _finish_unfinished_commit",
    ),
    Mutation(
        "E17", "a refusal of the command line is reported as success",
        CLI,
        "        _log(f\"error: {error}\")\n        return 1",
        "        _log(f\"error: {error}\")\n        return 0",
    ),
    # ---- round 00077: the filtered merge of the scan (Part A) -----------------------------------
    Mutation(
        "K1", "the filter drops entries that tie with the threshold (ties are lost)",
        GT_ENGINE,
        "        keep = ~(distances > self._threshold[chunk_index][:, None])",
        "        keep = distances < self._threshold[chunk_index][:, None]",
    ),
    Mutation(
        "K2", "the portable filter uses the threshold of another query row",
        GT_ENGINE,
        "        keep = ~(distances > self._threshold[chunk_index][:, None])",
        "        keep = ~(distances > xp.roll(self._threshold[chunk_index], 1)[:, None])",
    ),
    Mutation(
        "K3", "the portable filter forgets the current winners when it merges the survivors",
        GT_ENGINE,
        "        candidates[:, :k] = best\n        candidates[rows, k + slot] = keys",
        "        candidates[rows, k + slot] = keys",
    ),
    Mutation(
        "K4", "survivors of a query are written to the wrong slot (every query starts at slot 1)",
        GT_ENGINE,
        "        slot = xp.arange(total) - first_of_row[rows]",
        "        slot = xp.arange(total) - first_of_row[rows] + 1\n        width = width + 1",
    ),
    Mutation(
        "K5", "the threshold is never refreshed after a merge",
        GT_ENGINE,
        "        self._threshold[chunk_index] = threshold\n",
        "",
    ),
    Mutation(
        "K6", "a chunk is treated as full before every query holds k entries",
        GT_ENGINE,
        "        return threshold, not bool(is_empty.any())",
        "        return threshold, True",
    ),
    Mutation(
        "K7", "an empty slot is read as a real k-th distance",
        GT_ENGINE,
        "        threshold = xp.where(is_empty, xp.float32(xp.inf), kth_distance).astype(xp.float32)",
        "        threshold = kth_distance.astype(xp.float32)",
    ),
    Mutation(
        "K8", "the kernel filter drops entries that tie with the threshold",
        GT_ENGINE,
        "    if (distance > threshold[row]) {\n        return;\n    }",
        "    if (distance >= threshold[row]) {\n        return;\n    }",
    ),
    Mutation(
        "K9", "the kernel skips the clamp of negative l2 values at zero",
        GT_ENGINE,
        "        distance = (value != value) ? value : (value > 0.0f ? value : 0.0f);",
        "        distance = value;",
    ),
    Mutation(
        "K10", "the kernel writes survivors one column too far to the right",
        GT_ENGINE,
        "    candidates[(long long)row * (num_best + width) + num_best + slot] =",
        "    candidates[(long long)row * (num_best + width) + num_best + slot + 1] =",
    ),
    Mutation(
        "K11", "the kernel keeps the sign of an l2 zero out of the key (-0.0 sorts apart)",
        GT_ENGINE,
        "    const unsigned int bits = __float_as_uint(__fadd_rn(distance, 0.0f));",
        "    const unsigned int bits = __float_as_uint(distance);",
    ),
    Mutation(
        "K12", "an overflowing query (more survivors than slots) is not sent to the dense merge",
        GT_ENGINE,
        "        if most > width:\n            return None",
        "        if False:\n            return None",
    ),
    Mutation(
        "K13", "the inner-product kernel path uses +dot instead of -dot",
        GT_ENGINE,
        "        distance = -dot[index];",
        "        distance = dot[index];",
    ),
    Mutation(
        "K14", "the dense merge is skipped for a chunk that is not full yet",
        GT_ENGINE,
        "            if self._chunk_is_full[chunk_index] and self._device_kernel:",
        "            if self._device_kernel:",
    ),
    # ---- round 00077: the stage that answers the stream queries (Part B) ------------------------
    Mutation(
        "H1", "the answer of a step ignores the deletions of that step",
        STREAM_GT_FILE,
        "            if delete_path.exists():\n                deleted_mask[read_ids(delete_path)] = True",
        "            if False:\n                deleted_mask[read_ids(delete_path)] = True",
    ),
    Mutation(
        "H2", "the inserted vectors of a step are counted one step late",
        STREAM_GT_FILE,
        "            inserted += int(schedule[\"inserts_per_step\"][step - 1])\n\n            queries",
        "            queries",
    ),
    Mutation(
        "H3", "the stage records a pair as complete before its files are renamed into place",
        STREAM_GT_FILE,
        "    shutil.rmtree(pair.final_dir, ignore_errors=True)\n    os.replace(pair.tmp_dir, pair.final_dir)",
        "    shutil.rmtree(pair.final_dir, ignore_errors=True)\n    pair.final_dir.mkdir(parents=True)",
    ),
    Mutation(
        "H4", "the digest ignores the element type and length of a vector",
        STREAM_GT_FILE,
        "    prefix = f\"{rows.dtype.str}:{rows.shape[1]}:\".encode()",
        "    prefix = b\"\"",
    ),
    Mutation(
        "H5", "the digest index returns the position in sorted order instead of the input",
        STREAM_GT_FILE,
        "        result[single[match]] = self._position[low[single[match]]]",
        "        result[single[match]] = low[single[match]]",
    ),
    Mutation(
        "H6", "equal first halves of two digests are not told apart",
        STREAM_GT_FILE,
        "        for index in np.flatnonzero(run > 1):\n            for candidate in range(low[index], high[index]):\n                if self._second[candidate] == digests[index, 1]:",
        "        for index in np.flatnonzero(run > 1):\n            for candidate in range(low[index], high[index]):\n                if True:",
    ),
    Mutation(
        "H7", "an answer made for another stream_k is taken for complete",
        STREAM_GT_FILE,
        "        return \"complete\" if record[\"stream_k\"] == stream_k else \"mismatch\"",
        "        return \"complete\"",
    ),
    Mutation(
        "H8", "pairs that are complete are computed again",
        STREAM_GT_FILE,
        "            if state == \"complete\":\n                complete.append(pair)",
        "            if state == \"complete\":\n                pending.append(pair)",
    ),
    Mutation(
        "H9", "a pool name that matches nothing is silently ignored",
        STREAM_GT_FILE,
        "        if not matches:\n            raise StreamGtError(",
        "        if False:\n            raise StreamGtError(",
    ),
    Mutation(
        "H10", "a finished segment of lists is checked against a size only, not its hash",
        STREAM_GT_FILE,
        "            if sha256_file(path) != recorded[\"sha256\"]:",
        "            if False:",
    ),
    Mutation(
        "H11", "lists made for another universe are accepted",
        STREAM_GT_FILE,
        "            if meta[\"universe\"] != universe:",
        "            if False:",
    ),
    Mutation(
        "H12", "the stage does not remove the .tmp_ folders of an interrupted run",
        STREAM_GT_FILE,
        "        for leftover in sorted(self.directory.glob(TMP_PREFIX + \"*\")):\n            shutil.rmtree(leftover, ignore_errors=True)",
        "        pass",
    ),
    Mutation(
        "H13", "the family index keeps the old hash of a workload.json that the stage changed",
        STREAM_GT_FILE,
        "        if entry.get(\"workload_json_sha256\") != current:\n            entry[\"workload_json_sha256\"] = current\n            changed += 1",
        "        if entry.get(\"workload_json_sha256\") is None:\n            entry[\"workload_json_sha256\"] = current\n            changed += 1",
    ),
    Mutation(
        "H14", "the disk space check is skipped",
        STREAM_GT_FILE,
        "    if (answer_bytes + list_bytes) * 1.05 > free:",
        "    if False:",
    ),
    Mutation(
        "H15", "the progress line reports the speed of the whole run as the recent speed",
        STREAM_GT_FILE,
        "        first_time, first_done = self._samples[0]",
        "        first_time, first_done = self._started, self._start_done",
    ),
    Mutation(
        "H16", "the time left is computed from the wrong remaining count",
        STREAM_GT_FILE,
        "            format_duration((self._total - self._done) / recent)",
        "            format_duration(self._total / recent)",
    ),
    Mutation(
        "H17", "the stage settings are part of the common fingerprint again",
        FAMILY,
        "    description[\"groundtruth\"] = _without_stream_stage(description[\"groundtruth\"])\n",
        "",
    ),
    Mutation(
        "H18", "a family of the previous version is refused because it stored the stage settings",
        FAMILY,
        "    if \"groundtruth\" in stored.get(\"config\", {}):",
        "    if False:",
    ),
    Mutation(
        "H19", "the follow-up stage is not run for an extended family",
        WORKLOAD,
        "        _answer_stream_queries(config, base_set, family_dir, log)\n        return family_dir",
        "        return family_dir",
    ),
    Mutation(
        "H20", "the follow-up stage ignores stream_groundtruth = false",
        WORKLOAD,
        "    if not config.groundtruth.stream_groundtruth:\n        return\n    # Imported here because stream_gt.py builds on this module.",
        "    # Imported here because stream_gt.py builds on this module.",
    ),
    Mutation(
        "H21", "stream_k smaller than eval_k is accepted",
        CONFIG,
        "    if gt.stream_k < eval_k:\n        raise ConfigError(",
        "    if False:\n        raise ConfigError(",
    ),
    Mutation(
        "H22", "a pool pattern is matched as a prefix instead of a glob",
        STREAM_GT_FILE,
        "        matches = [name for name in pool_names if fnmatch.fnmatchcase(name, pattern)]",
        "        matches = [name for name in pool_names if name.startswith(pattern.rstrip(\"*\"))]",
    ),
    Mutation(
        "H23", "the stage does not give back the GPU memory that the build cached",
        STREAM_GT_FILE,
        "    release_device_memory()  # the build's cached blocks must not make the GPU look busy\n",
        "",
    ),
    Mutation(
        "H24", "the default depth of the stream answers is shallower than the default eval_k",
        CONFIG,
        "DEFAULT_STREAM_K = 120",
        "DEFAULT_STREAM_K = 10",
    ),
    Mutation(
        "H25", "the digest covers only the first half of a vector (vectors that share it share a list)",
        STREAM_GT_FILE,
        "        blake(prefix + raw[start : start + stride], digest_size=DIGEST_BYTES).digest()",
        "        blake(prefix + raw[start : start + stride // 2], digest_size=DIGEST_BYTES).digest()",
    ),
    Mutation(
        "K15", "the portable filtered merge keeps only k-1 winners",
        GT_ENGINE,
        "        candidates[rows, k + slot] = keys\n        merged = xp.partition(candidates, k - 1, axis=1)[:, :k]",
        "        candidates[rows, k + slot] = keys\n        merged = xp.partition(candidates, k - 1, axis=1)[:, : k - 1]",
    ),
    Mutation(
        "K16", "the kernel filtered merge keeps only k-1 winners",
        GT_ENGINE,
        "            return self._best[chunk_index], 0\n        merged = xp.partition(candidates, k - 1, axis=1)[:, :k]",
        "            return self._best[chunk_index], 0\n        merged = xp.partition(candidates, k - 1, axis=1)[:, : k - 1]",
    ),
]
