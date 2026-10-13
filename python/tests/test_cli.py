"""Tests for the command line interface and the example configuration files."""

import configparser
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from semantic_vdb import binio, cli, config, manifest

REPOSITORY = Path(__file__).resolve().parents[2]


def test_example_configs_in_the_repository_are_valid():
    # The examples are documentation; they must keep parsing as the code evolves.
    base = config.load_base_set_config(REPOSITORY / "config" / "base_set.example.ini")
    assert base.num_base == 1_000_000 and base.max_inserts == 1_000_000
    workload = config.load_workload_config(REPOSITORY / "config" / "workload.example.ini")
    assert workload.groups.group_sizes == (200,)
    assert workload.seeds.num_groups is None
    assert workload.stream.zipf_exponents == (0.3, 0.6, 0.99)
    assert workload.groups.target_overlaps == (0.8,)
    assert workload.existing == "error"


def test_example_workload_config_documents_every_option(monkeypatch):
    """Every (section, key) the loader reads must be written in the example, in its section.

    A key counts as documented when the example has a line ``key = ...`` (active) or
    ``; key = ...`` (commented out, optional) inside the section's block.
    """
    queried = set()
    original_parser = configparser.ConfigParser

    class RecordingParser(original_parser):
        def has_option(self, section, option):
            queried.add((section, option))
            return super().has_option(section, option)

        def get(self, section, option, **kwargs):
            queried.add((section, option))
            return super().get(section, option, **kwargs)

    monkeypatch.setattr(config.configparser, "ConfigParser", RecordingParser)
    config.load_workload_config(REPOSITORY / "config" / "workload.example.ini")

    # Keys of the previous version are only read to be refused; they must not be advertised.
    removed = {("stream", "length"), ("updates", "total_inserts"), ("updates", "total_deletes")}
    text = (REPOSITORY / "config" / "workload.example.ini").read_text(encoding="utf-8")
    blocks = {}
    for match in re.finditer(r"(?m)^\[(\w+)\][ \t]*$", text):
        blocks[match.group(1)] = match.end()
    ordered = sorted(blocks.items(), key=lambda item: item[1])
    sections = {}
    for position, (name, start) in enumerate(ordered):
        stop = ordered[position + 1][1] if position + 1 < len(ordered) else len(text)
        sections[name] = text[start:stop]

    assert queried - removed, "the spy recorded nothing"
    undocumented = sorted(
        (section, key)
        for section, key in queried - removed
        if not re.search(rf"(?m)^[ \t]*;?[ \t]*{re.escape(key)}[ \t]*=", sections.get(section, ""))
    )
    assert undocumented == []
    for section, key in removed:
        assert not re.search(rf"(?m)^[ \t]*;?[ \t]*{key}[ \t]*=", sections.get(section, ""))


def write_synthetic_inputs(tmp_path):
    rng = np.random.default_rng(0)
    centres = rng.integers(40, 200, size=(10, 8))

    def rows(count):
        points = centres[rng.integers(0, 10, size=count)] + rng.normal(size=(count, 8)) * 8.0
        return np.clip(np.rint(points), 0, 255).astype(np.uint8)

    binio.write_array(tmp_path / "raw.u8bin", rows(700))
    binio.write_array(tmp_path / "query.u8bin", rows(20))


def test_cli_builds_a_base_set_and_a_workload_from_ini_files(tmp_path, capsys):
    write_synthetic_inputs(tmp_path)
    (tmp_path / "base.ini").write_text(
        f"""
[input]
raw_file = {tmp_path / "raw.u8bin"}
[partition]
seed = 1
num_base = 400
max_inserts = 100
[output]
dir = {tmp_path / "out"}
name = b
""",
        encoding="utf-8",
    )
    (tmp_path / "workload.ini").write_text(
        f"""
[workload]
name = w
[base_set]
dir = {tmp_path / "out" / "b"}
[seeds]
mode = public_queries
query_file = {tmp_path / "query.u8bin"}
seed = 2
[groups]
group_size = 4
neighbours = 30
axes = 3
target_overlap = 0.7
calibration_queries = 10
overlap_tolerance = 0.06
[parallel]
gpus = none
numa_node = none
cpu_workers = 1
[stream]
total_requests = 120
num_steps = 4
zipf_exponents = 0.9
[updates]
update_percents = 0, 50
[evaluation]
checkpoint_every = 2
eval_k = 5
[groundtruth]
stream_groundtruth = true
stream_k = 6
device = cpu
""",
        encoding="utf-8",
    )

    assert cli.main(["base-set", "--config", str(tmp_path / "base.ini")]) == 0
    base_path = Path(capsys.readouterr().out.strip())
    assert base_path == tmp_path / "out" / "b"

    assert cli.main(["workload", "--config", str(tmp_path / "workload.ini")]) == 0
    family_path = Path(capsys.readouterr().out.strip())
    assert family_path == tmp_path / "out" / "b" / "workloads" / "w"
    # One pool (group size 4), two sets inside it, and the index files of the family.
    assert sorted(path.name for path in family_path.iterdir()) == [
        "build_report.json", "build_report_stream_groundtruth_1.json", "family.json", "o70",
        "shared",
    ]
    # Overlap level o70 (target_overlap 0.7), group size 4, then the two update sets.
    assert sorted(path.name for path in (family_path / "o70").iterdir()) == ["g4"]
    assert sorted(path.name for path in (family_path / "o70" / "g4").iterdir()) == ["u0", "u50"]

    # u50: 120 requests = 60 reads + 60 updates (60 insertions and 60 deletions).
    info = manifest.read_json(family_path / "o70" / "g4" / "u50" / "workload.json")
    assert info["name"] == "w" and info["set_name"] == "u50"
    assert info["update_set"] == {
        "update_percent": 50.0, "total_requests": 120, "reads": 60, "insertions": 60, "deletions": 60,
    }
    assert info["schedule"]["checkpoints"] == [0, 2, 4]
    assert (family_path / "o70" / "g4" / "u50" / "groundtruth" / "stream_zipf_0.9" / "step_00004.bin").exists()

    # u0 is read-only: all 120 requests are reads and nothing is inserted or deleted.
    read_only = manifest.read_json(family_path / "o70" / "g4" / "u0" / "workload.json")
    assert read_only["update_set"]["reads"] == 120 and read_only["update_set"]["insertions"] == 0


def test_cli_adds_a_target_overlap_to_an_existing_family_with_existing_extend(tmp_path, capsys):
    """The same command with a longer overlap list and ``existing = extend`` adds the new pool."""
    write_synthetic_inputs(tmp_path)
    (tmp_path / "base.ini").write_text(
        f"""
[input]
raw_file = {tmp_path / "raw.u8bin"}
[partition]
seed = 1
num_base = 400
max_inserts = 100
[output]
dir = {tmp_path / "out"}
name = b
""",
        encoding="utf-8",
    )

    def workload_ini(overlaps, existing):
        return f"""
[workload]
name = w
existing = {existing}
[base_set]
dir = {tmp_path / "out" / "b"}
[seeds]
mode = public_queries
query_file = {tmp_path / "query.u8bin"}
seed = 2
[groups]
group_size = 4
neighbours = 30
axes = 3
target_overlap = {overlaps}
calibration_queries = 10
overlap_tolerance = 0.06
[parallel]
gpus = none
numa_node = none
cpu_workers = 1
[stream]
total_requests = 120
num_steps = 4
zipf_exponents = 0.9
[updates]
update_percents = 0, 50
[evaluation]
checkpoint_every = 2
eval_k = 5
[groundtruth]
stream_groundtruth = true
stream_k = 6
device = cpu
"""

    assert cli.main(["base-set", "--config", str(tmp_path / "base.ini")]) == 0
    capsys.readouterr()
    ini = tmp_path / "workload.ini"
    ini.write_text(workload_ini("0.7", "error"), encoding="utf-8")
    assert cli.main(["workload", "--config", str(ini)]) == 0
    family_path = Path(capsys.readouterr().out.strip())

    # The same ini again: an existing family is refused (the default, "error") ...
    assert cli.main(["workload", "--config", str(ini)]) == 1
    assert "existing = extend" in capsys.readouterr().err

    # ... and with a second overlap and "extend" only the new pool is made.
    ini.write_text(workload_ini("0.7, 0.5", "extend"), encoding="utf-8")
    assert cli.main(["workload", "--config", str(ini)]) == 0
    assert Path(capsys.readouterr().out.strip()) == family_path
    index = manifest.read_json(family_path / "family.json")
    assert [pool["name"] for pool in index["pools"]] == ["o70/g4", "o50/g4"]
    assert (family_path / "o50" / "g4" / "u50" / "workload.json").is_file()
    assert (family_path / "build_report_extension_1.json").is_file()


def test_cli_reports_configuration_errors_with_exit_code_two(tmp_path, capsys):
    (tmp_path / "bad.ini").write_text("[input]\nraw_file = x\n", encoding="utf-8")
    assert cli.main(["base-set", "--config", str(tmp_path / "bad.ini")]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_the_command_line_runs_with_several_cpu_workers_without_restarting_itself(tmp_path):
    """`python -m semantic_vdb` starts spawned workers, which import the main module again.

    The package entry point must tolerate that (multiprocessing skips a package's
    ``__main__``); a script without a main guard would build the workload once per worker.
    """
    write_synthetic_inputs(tmp_path)
    (tmp_path / "base.ini").write_text(
        f"[input]\nraw_file = {tmp_path / 'raw.u8bin'}\n[partition]\nseed = 1\nnum_base = 400\n"
        f"max_inserts = 100\n[output]\ndir = {tmp_path / 'out'}\nname = b\n",
        encoding="utf-8",
    )
    (tmp_path / "workload.ini").write_text(
        f"""
[workload]
name = w
[base_set]
dir = {tmp_path / "out" / "b"}
[seeds]
mode = public_queries
query_file = {tmp_path / "query.u8bin"}
seed = 2
[groups]
group_size = 3, 4
neighbours = 30
axes = 3
target_overlap = 0.7
calibration_queries = 10
overlap_tolerance = 0.06
[parallel]
gpus = none
numa_node = none
cpu_workers = 2
[stream]
total_requests = 120
num_steps = 4
zipf_exponents = 0.9
[updates]
update_percents = 0, 50
[groundtruth]
device = cpu
""",
        encoding="utf-8",
    )
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY / "python"))
    for command, config_name in (("base-set", "base.ini"), ("workload", "workload.ini")):
        completed = subprocess.run(
            [sys.executable, "-m", "semantic_vdb", command, "--config", str(tmp_path / config_name)],
            env=environment, capture_output=True, text=True, timeout=300,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]

    family = tmp_path / "out" / "b" / "workloads" / "w"
    assert sorted(path.name for path in (family / "o70").iterdir()) == ["g3", "g4"]
    report = manifest.read_json(family / "build_report.json")
    assert report["parallel"]["cpu_workers"] == 2
    assert not list((tmp_path / "out" / "b" / "workloads").glob("*.building"))
