"""Loading and validation of the two ``.ini`` configuration files.

* ``base_set.ini``  describes the shared base set (row partition of the raw file).
* ``workload.ini``  describes one workload set that references a base set.

All validation that needs no data file happens here, so a bad setting fails before any
large file is opened.
"""

from __future__ import annotations

import configparser
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from semantic_vdb.layout import overlap_dir_name
from semantic_vdb.partition import BASE_SELECTIONS
from semantic_vdb.updates import RequestMix, ScheduleError, request_mix

METRICS = ("l2", "ip")
SEED_MODES = ("public_queries", "base_rows", "external")
EVAL_SOURCES = ("original", "external")
# How the members of a group are made: "n2" = local-PCA noise around the seed query,
# "none" = no variants, the given queries themselves are the pool (one query per group).
VARIANTS = ("n2", "none")
DEVICES = ("gpu", "cpu")
# What to do when the family directory of this name already exists: refuse (the default) or
# add only the pools that are missing (see extend.py).
EXISTING_POLICIES = ("error", "extend")
# Neighbours stored per stream query unless [groundtruth] stream_k says otherwise: deeper than
# the default evaluation k (100) so that recall@100 can be scored with some tolerance.
DEFAULT_STREAM_K = 120


class ConfigError(ValueError):
    """A setting is missing, malformed or out of range."""


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseSetConfig:
    raw_file: Path
    metric: str
    seed: int
    num_base: int
    base_selection: str
    num_query_rows: int
    max_inserts: Optional[int]  # None means "every remaining row"
    output_dir: Path
    name: str


@dataclass(frozen=True)
class SeedsConfig:
    mode: str
    query_file: Optional[Path]
    group_id_file: Optional[Path]
    num_groups: Optional[int]  # None means "all"
    seed: int


@dataclass(frozen=True)
class GroupsConfig:
    # One pool per (target overlap, group size) pair, in the directory o<overlap>/g<size>;
    # every pool is crossed with every update percentage. One value of each is the usual case.
    group_sizes: Tuple[int, ...]
    include_original: bool
    neighbours: int
    axes: int
    residual: bool
    target_overlaps: Tuple[float, ...]
    calibration_queries: int
    overlap_tolerance: float
    variants: str = "n2"

    def __post_init__(self) -> None:
        # Checked here (not only in the .ini loader) so that a configuration built in code
        # is validated like one read from a file.
        if self.variants not in VARIANTS:
            raise ConfigError(
                f"[groups] variants must be one of {list(VARIANTS)}, got {self.variants!r}"
            )
        if not self.group_sizes:
            raise ConfigError("[groups] group_size must list at least one size")
        if any(size < 1 for size in self.group_sizes):
            raise ConfigError(f"[groups] group_size values must be >= 1, got {list(self.group_sizes)}")
        if len(set(self.group_sizes)) != len(self.group_sizes):
            raise ConfigError(f"[groups] group_size must not repeat a value: {list(self.group_sizes)}")
        if self.variants == "none" and tuple(self.group_sizes) != (1,):
            raise ConfigError(
                "[groups] variants = none uses the given queries as they are (one per group), "
                f"so group_size must be 1 or omitted, got {list(self.group_sizes)}"
            )
        if not self.target_overlaps:
            raise ConfigError("[groups] target_overlap must list at least one value")
        if any(not 0.0 < value < 1.0 for value in self.target_overlaps):
            raise ConfigError(
                "[groups] target_overlap must be strictly between 0 and 1, "
                f"got {list(self.target_overlaps)}"
            )
        names = [overlap_dir_name(value) for value in self.target_overlaps]
        if len(set(names)) != len(names):
            raise ConfigError(
                f"[groups] target_overlap must not repeat a value: {list(self.target_overlaps)}"
            )


@dataclass(frozen=True)
class StreamConfig:
    total_requests: int  # reads + updates of ONE workload set
    num_steps: int
    zipf_exponents: Tuple[float, ...]  # empty only when uniform_unique is true
    member_selection: str
    seed: int
    # Also make a stream of DIFFERENT vectors drawn uniformly from the whole pool.
    uniform_unique: bool = False

    def __post_init__(self) -> None:
        if not self.zipf_exponents and not self.uniform_unique:
            raise ConfigError(
                "[stream] no stream was requested: give zipf_exponents or uniform_unique = true"
            )


@dataclass(frozen=True)
class UpdatesConfig:
    seed: int
    # One workload set per percentage; an update is one insertion plus one deletion.
    update_percents: Tuple[float, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    source: str
    eval_file: Optional[Path]
    checkpoint_every: int
    eval_k: int


@dataclass(frozen=True)
class GroundTruthConfig:
    stream_k: int
    device: str
    # None means "auto": chosen from the free memory of the selected GPU at run time.
    base_batch_rows: Optional[int]
    query_chunk_rows: Optional[int]
    # True: after the workload is built, a follow-up stage computes the exact answers of the
    # stream queries (the requests of the Zipf / uniform_unique streams); sets that already
    # have them are skipped (see stream_gt.py). False (the default, because the stage can take
    # a long time on large data): the stream queries get no ground truth, only the evaluation
    # queries are scored at their checkpoints.
    stream_groundtruth: bool = False
    # Pools that get stream answers: ("all",) or pool names / glob patterns (o50/g200, o80/*).
    # A chosen pool gets all its streams and all its update percentages.
    stream_pools: Tuple[str, ...] = ("all",)
    # Seconds between the progress lines of the follow-up stage.
    progress_seconds: int = 60


@dataclass(frozen=True)
class ParallelConfig:
    """Which machine resources the build may use (see ``parallel.py``).

    The defaults are the values of the development machine: GPU 1, which sits on NUMA
    node 1. On another machine a missing GPU or node is an error, never silently skipped.
    """

    # "auto" (every free GPU), "none", or the nvidia-smi indexes to use.
    gpus: Union[str, Tuple[int, ...]] = (1,)
    gpu_max_used_mb: int = 1024  # a GPU using less memory than this counts as free
    workers_per_gpu: int = 1
    cpu_workers: Union[str, int] = "auto"  # "auto" = physical cores of numa_node
    numa_node: Union[str, int] = 1  # "none", "auto" (the node of the GPU) or a node number

    def __post_init__(self) -> None:
        if isinstance(self.gpus, str):
            if self.gpus not in ("auto", "none"):
                raise ConfigError(f"[parallel] gpus must be auto, none or GPU numbers, got {self.gpus!r}")
        else:
            if not self.gpus or any(index < 0 for index in self.gpus):
                raise ConfigError(f"[parallel] gpus must list GPU numbers >= 0, got {list(self.gpus)}")
            if len(set(self.gpus)) != len(self.gpus):
                raise ConfigError(f"[parallel] gpus must not repeat a GPU: {list(self.gpus)}")
        if self.gpu_max_used_mb < 0:
            raise ConfigError("[parallel] gpu_max_used_mb must be >= 0")
        if self.workers_per_gpu < 1:
            raise ConfigError("[parallel] workers_per_gpu must be >= 1")
        if isinstance(self.cpu_workers, str):
            if self.cpu_workers != "auto":
                raise ConfigError(f"[parallel] cpu_workers must be auto or a number, got {self.cpu_workers!r}")
        elif self.cpu_workers < 1:
            raise ConfigError("[parallel] cpu_workers must be >= 1")
        if isinstance(self.numa_node, str):
            if self.numa_node not in ("none", "auto"):
                raise ConfigError(f"[parallel] numa_node must be none, auto or a number, got {self.numa_node!r}")
        elif self.numa_node < 0:
            raise ConfigError("[parallel] numa_node must be >= 0")


@dataclass(frozen=True)
class WorkloadConfig:
    name: str  # the family directory; its sets are named u<percent>
    base_set_dir: Path
    seeds: SeedsConfig
    groups: GroupsConfig
    stream: StreamConfig
    updates: UpdatesConfig
    evaluation: EvaluationConfig
    groundtruth: GroundTruthConfig
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    # What to do when the family already exists: "error" or "extend" (add missing pools).
    existing: str = "error"

    def __post_init__(self) -> None:
        if self.existing not in EXISTING_POLICIES:
            raise ConfigError(
                f"[workload] existing must be one of {list(EXISTING_POLICIES)}, "
                f"got {self.existing!r}"
            )

    def request_mixes(self) -> List[RequestMix]:
        """The workload sets to build, one per update percentage, in the configured order.

        Raises ``ConfigError`` when a percentage cannot be delivered exactly, so a
        configuration built in code is checked like one read from an ``.ini`` file.
        """
        mixes: List[RequestMix] = []
        names = set()
        for percent in self.updates.update_percents:
            try:
                mix = request_mix(self.stream.total_requests, percent)
            except ScheduleError as error:
                raise ConfigError(f"[updates] update_percents: {error}") from error
            if mix.reads < self.stream.num_steps:
                raise ConfigError(
                    f"[updates] {percent}% leaves {mix.reads} reads for {self.stream.num_steps} "
                    "steps; every step needs at least one read"
                )
            if mix.name in names:
                raise ConfigError(f"[updates] update_percents lists {percent}% more than once")
            names.add(mix.name)
            mixes.append(mix)
        return mixes


# ---------------------------------------------------------------------------
# small parsing helpers
# ---------------------------------------------------------------------------


def _read_ini(path: str | os.PathLike) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(
        interpolation=None,  # paths and values may contain '%'
        inline_comment_prefixes=(";",),
        comment_prefixes=("#", ";"),
    )
    if not parser.read(path, encoding="utf-8"):
        raise ConfigError(f"cannot read configuration file {path}")
    return parser


def _raw(parser: configparser.ConfigParser, section: str, key: str, default=None) -> str:
    if parser.has_option(section, key):
        value = parser.get(section, key).strip()
        if value != "":
            return value
    if default is not None:
        return default
    raise ConfigError(f"missing required setting [{section}] {key}")


def _integer(parser, section: str, key: str, default=None, minimum: Optional[int] = None) -> int:
    text = _raw(parser, section, key, None if default is None else str(default))
    try:
        value = int(text)
    except ValueError as error:
        raise ConfigError(f"[{section}] {key} must be an integer, got {text!r}") from error
    if minimum is not None and value < minimum:
        raise ConfigError(f"[{section}] {key} must be >= {minimum}, got {value}")
    return value


def _real(parser, section: str, key: str, default=None) -> float:
    text = _raw(parser, section, key, None if default is None else str(default))
    try:
        return float(text)
    except ValueError as error:
        raise ConfigError(f"[{section}] {key} must be a number, got {text!r}") from error


def _boolean(parser, section: str, key: str, default: bool) -> bool:
    text = _raw(parser, section, key, "true" if default else "false").lower()
    if text in ("true", "yes", "1", "on"):
        return True
    if text in ("false", "no", "0", "off"):
        return False
    raise ConfigError(f"[{section}] {key} must be true or false, got {text!r}")


def _choice(parser, section: str, key: str, choices, default=None) -> str:
    value = _raw(parser, section, key, default).lower()
    if value not in choices:
        raise ConfigError(f"[{section}] {key} must be one of {list(choices)}, got {value!r}")
    return value


def _path(parser, section: str, key: str) -> Path:
    return Path(_raw(parser, section, key))


def _optional_path(parser, section: str, key: str) -> Optional[Path]:
    if parser.has_option(section, key) and parser.get(section, key).strip():
        return Path(parser.get(section, key).strip())
    return None


def _safe_name(name: str, what: str) -> str:
    # A name becomes a directory name; reject anything that could leave its parent.
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ConfigError(f"{what} must be a plain directory name, got {name!r}")
    return name


# ---------------------------------------------------------------------------
# base_set.ini
# ---------------------------------------------------------------------------


def load_base_set_config(path: str | os.PathLike) -> BaseSetConfig:
    parser = _read_ini(path)

    max_inserts_text = _raw(parser, "partition", "max_inserts").lower()
    if max_inserts_text == "rest":
        max_inserts = None
    else:
        max_inserts = _integer(parser, "partition", "max_inserts", minimum=0)

    return BaseSetConfig(
        raw_file=_path(parser, "input", "raw_file"),
        metric=_choice(parser, "input", "metric", METRICS, default="l2"),
        seed=_integer(parser, "partition", "seed"),
        num_base=_integer(parser, "partition", "num_base", minimum=1),
        base_selection=_choice(
            parser, "partition", "base_selection", BASE_SELECTIONS, default="random"
        ),
        num_query_rows=_integer(parser, "partition", "num_query_rows", default=0, minimum=0),
        max_inserts=max_inserts,
        output_dir=_path(parser, "output", "dir"),
        name=_safe_name(_raw(parser, "output", "name"), "[output] name"),
    )


# ---------------------------------------------------------------------------
# workload.ini
# ---------------------------------------------------------------------------


def _load_seeds(parser) -> SeedsConfig:
    mode = _choice(parser, "seeds", "mode", SEED_MODES)

    query_file = _optional_path(parser, "seeds", "query_file")
    group_id_file = _optional_path(parser, "seeds", "group_id_file")
    if mode in ("public_queries", "external") and query_file is None:
        raise ConfigError(f"[seeds] query_file is required for mode {mode}")
    if mode == "external" and group_id_file is None:
        raise ConfigError("[seeds] group_id_file is required for mode external")

    groups_text = _raw(parser, "seeds", "num_groups", "all").lower()
    if groups_text == "all":
        num_groups = None
    else:
        num_groups = _integer(parser, "seeds", "num_groups", minimum=1)

    return SeedsConfig(
        mode=mode,
        query_file=query_file,
        group_id_file=group_id_file,
        num_groups=num_groups,
        seed=_integer(parser, "seeds", "seed"),
    )


def _integer_list(parser, section: str, key: str, default: str) -> Tuple[int, ...]:
    """A comma-separated list of integers, e.g. ``50, 200`` (one number is a list of one)."""
    text = _raw(parser, section, key, default)
    try:
        return tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ConfigError(f"[{section}] {key} must be integers separated by commas, got {text!r}") from error


def _real_list(parser, section: str, key: str, default: str) -> Tuple[float, ...]:
    """A comma-separated list of numbers, e.g. ``0.8, 0.5`` (one number is a list of one)."""
    text = _raw(parser, section, key, default)
    try:
        return tuple(float(item) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise ConfigError(
            f"[{section}] {key} must be numbers separated by commas, got {text!r}"
        ) from error


def _load_groups(parser) -> GroupsConfig:
    variants = _choice(parser, "groups", "variants", VARIANTS, default="n2")
    # Without variants every group is the given query alone, so the size defaults to 1.
    default_group_size = "1" if variants == "none" else "200"
    groups = GroupsConfig(
        group_sizes=_integer_list(parser, "groups", "group_size", default_group_size),
        include_original=_boolean(parser, "groups", "include_original", False),
        neighbours=_integer(parser, "groups", "neighbours", default=500, minimum=2),
        axes=_integer(parser, "groups", "axes", default=16, minimum=1),
        residual=_boolean(parser, "groups", "residual", False),
        target_overlaps=_real_list(parser, "groups", "target_overlap", "0.8"),
        calibration_queries=_integer(parser, "groups", "calibration_queries", default=500, minimum=1),
        overlap_tolerance=_real(parser, "groups", "overlap_tolerance", default=0.02),
        variants=variants,
    )
    if groups.axes > groups.neighbours - 1:
        raise ConfigError(
            f"[groups] axes ({groups.axes}) cannot exceed neighbours - 1 ({groups.neighbours - 1})"
        )
    if not 0.0 < groups.overlap_tolerance < 0.5:
        raise ConfigError("[groups] overlap_tolerance must be in (0, 0.5)")
    return groups


def _reject_removed_key(parser, section: str, key: str, hint: str) -> None:
    # Settings of the previous version are refused with a pointer to their replacement,
    # because silently ignoring them would build a different workload than the user wrote.
    if parser.has_option(section, key):
        raise ConfigError(f"[{section}] {key} is no longer supported: {hint}")


def _load_stream(parser) -> StreamConfig:
    _reject_removed_key(
        parser,
        "stream",
        "length",
        "use total_requests (reads + updates) and [updates] update_percents",
    )
    total_requests = _integer(parser, "stream", "total_requests", minimum=1)
    num_steps = _integer(parser, "stream", "num_steps", minimum=1)
    if num_steps > total_requests:
        raise ConfigError(
            f"[stream] num_steps ({num_steps}) cannot exceed total_requests ({total_requests})"
        )

    uniform_unique = _boolean(parser, "stream", "uniform_unique", False)

    # "none" asks for no Zipf stream at all (then uniform_unique must provide the stream).
    exponent_text = _raw(parser, "stream", "zipf_exponents")
    if exponent_text.lower() == "none":
        exponents: Tuple[float, ...] = ()
    else:
        try:
            exponents = tuple(float(item) for item in exponent_text.split(",") if item.strip())
        except ValueError as error:
            raise ConfigError(
                f"[stream] zipf_exponents must be numbers or none, got {exponent_text!r}"
            ) from error
        if not exponents or any(value < 0.0 for value in exponents):
            raise ConfigError("[stream] zipf_exponents must be a non-empty list of values >= 0")
        if len(set(exponents)) != len(exponents):
            raise ConfigError("[stream] zipf_exponents must not repeat a value")

    return StreamConfig(
        total_requests=total_requests,
        num_steps=num_steps,
        zipf_exponents=exponents,
        member_selection=_choice(parser, "stream", "member_selection", ("uniform",), "uniform"),
        seed=_integer(parser, "stream", "seed", default=31),
        uniform_unique=uniform_unique,
    )


def _load_updates(parser) -> UpdatesConfig:
    for key in ("total_inserts", "total_deletes"):
        _reject_removed_key(
            parser,
            "updates",
            key,
            "give update_percents; every update is one insertion plus one deletion",
        )
    percent_text = _raw(parser, "updates", "update_percents")
    try:
        percents = tuple(float(item) for item in percent_text.split(",") if item.strip())
    except ValueError as error:
        raise ConfigError(
            f"[updates] update_percents must be numbers, got {percent_text!r}"
        ) from error
    if not percents:
        raise ConfigError("[updates] update_percents must list at least one percentage")
    return UpdatesConfig(
        seed=_integer(parser, "updates", "seed", default=21), update_percents=percents
    )


def _load_evaluation(parser) -> EvaluationConfig:
    source = _choice(parser, "evaluation", "source", EVAL_SOURCES, default="original")
    eval_file = _optional_path(parser, "evaluation", "eval_file")
    if source == "external" and eval_file is None:
        raise ConfigError("[evaluation] eval_file is required when source = external")
    return EvaluationConfig(
        source=source,
        eval_file=eval_file,
        checkpoint_every=_integer(parser, "evaluation", "checkpoint_every", default=1, minimum=1),
        eval_k=_integer(parser, "evaluation", "eval_k", default=100, minimum=1),
    )


def _pool_patterns(parser) -> Tuple[str, ...]:
    """``[groundtruth] stream_pools``: ``all`` or a comma-separated list of names / globs."""
    text = _raw(parser, "groundtruth", "stream_pools", "all")
    names = tuple(item.strip() for item in text.split(",") if item.strip())
    if not names:
        raise ConfigError("[groundtruth] stream_pools must be all or a list of pool names")
    if "all" in names and len(names) > 1:
        raise ConfigError("[groundtruth] stream_pools: all cannot be combined with pool names")
    for name in names:
        if name.startswith("/") or ".." in name or " " in name:
            raise ConfigError(f"[groundtruth] stream_pools: {name!r} is not a pool name")
    return names


def _check_stream_k(workload_config) -> None:
    """The stored neighbours must reach at least as deep as the recall that is scored.

    Only relevant when the follow-up stage will run. ``stream_k`` equal to ``eval_k`` works
    but leaves no room for a tolerant comparison, so it is only a warning.
    """
    gt, eval_k = workload_config.groundtruth, workload_config.evaluation.eval_k
    if not gt.stream_groundtruth:
        return
    if gt.stream_k < eval_k:
        raise ConfigError(
            f"[groundtruth] stream_k ({gt.stream_k}) must not be smaller than [evaluation] "
            f"eval_k ({eval_k}); set it larger than the k at which recall is scored"
        )
    if gt.stream_k == eval_k:
        warnings.warn(
            f"[groundtruth] stream_k equals eval_k ({eval_k}); a larger value leaves room "
            "for answers that are slightly out of the top-k",
            UserWarning,
            stacklevel=2,
        )


def _batch_rows(parser, key: str) -> Optional[int]:
    """A GT batch size: ``auto`` (the default, returned as None) or an integer >= 1."""
    value = _auto_or_integer(parser, "groundtruth", key, "auto", minimum=1)
    return None if value == "auto" else value


def _auto_or_integer(parser, section: str, key: str, default: str, minimum: int):
    """``auto`` (returned as the string) or an integer >= minimum."""
    text = _raw(parser, section, key, default)
    if text.lower() == "auto":
        return "auto"
    return _integer(parser, section, key, default=default, minimum=minimum)


def _load_parallel(parser) -> ParallelConfig:
    defaults = ParallelConfig()

    gpus_text = _raw(parser, "parallel", "gpus", "1").lower()
    if gpus_text in ("auto", "none"):
        gpus: Union[str, Tuple[int, ...]] = gpus_text
    else:
        gpus = _integer_list(parser, "parallel", "gpus", "1")

    numa_text = _raw(parser, "parallel", "numa_node", str(defaults.numa_node)).lower()
    numa_node: Union[str, int] = (
        numa_text if numa_text in ("none", "auto") else _integer(
            parser, "parallel", "numa_node", default=defaults.numa_node, minimum=0
        )
    )
    return ParallelConfig(
        gpus=gpus,
        gpu_max_used_mb=_integer(
            parser, "parallel", "gpu_max_used_mb", default=defaults.gpu_max_used_mb, minimum=0
        ),
        workers_per_gpu=_integer(parser, "parallel", "workers_per_gpu", default=1, minimum=1),
        cpu_workers=_auto_or_integer(parser, "parallel", "cpu_workers", "auto", minimum=1),
        numa_node=numa_node,
    )


def load_workload_config(path: str | os.PathLike) -> WorkloadConfig:
    parser = _read_ini(path)

    seeds = _load_seeds(parser)
    evaluation = _load_evaluation(parser)
    if seeds.mode == "external" and evaluation.source != "external":
        raise ConfigError("[evaluation] source must be external when [seeds] mode = external")

    workload_config = WorkloadConfig(
        name=_safe_name(_raw(parser, "workload", "name"), "[workload] name"),
        base_set_dir=_path(parser, "base_set", "dir"),
        seeds=seeds,
        groups=_load_groups(parser),
        stream=_load_stream(parser),
        updates=_load_updates(parser),
        evaluation=evaluation,
        groundtruth=GroundTruthConfig(
            stream_k=_integer(parser, "groundtruth", "stream_k", default=DEFAULT_STREAM_K, minimum=1),
            device=_choice(parser, "groundtruth", "device", DEVICES, default="gpu"),
            base_batch_rows=_batch_rows(parser, "base_batch_rows"),
            query_chunk_rows=_batch_rows(parser, "query_chunk_rows"),
            stream_groundtruth=_boolean(parser, "groundtruth", "stream_groundtruth", False),
            stream_pools=_pool_patterns(parser),
            progress_seconds=_integer(
                parser, "groundtruth", "progress_seconds", default=60, minimum=1
            ),
        ),
        parallel=_load_parallel(parser),
        existing=_choice(parser, "workload", "existing", EXISTING_POLICIES, default="error"),
    )
    # Fail now, before any data file is opened, if a percentage cannot be delivered exactly.
    workload_config.request_mixes()
    _check_stream_k(workload_config)
    return workload_config
