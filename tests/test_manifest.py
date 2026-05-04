"""Run manifest: hashing, file listing, and integrity verification."""

from __future__ import annotations

from pathlib import Path

from training_pipeline.manifest import (
    RunManifest,
    StageEntry,
    discover_files,
    file_entries,
    hash_bytes,
    hash_file,
    hash_obj,
    load_manifest,
    make_run_id,
    now_utc,
    verify_manifest,
    write_manifest,
)


def test_hash_obj_is_stable_across_key_order():
    a = hash_obj({"a": 1, "b": [2, 3]})
    b = hash_obj({"b": [2, 3], "a": 1})
    assert a == b


def test_hash_file_matches_content_hash(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    assert hash_file(p) == hash_bytes(b"hello")


def test_file_entries_relative_path(tmp_path: Path):
    p = tmp_path / "a" / "b.jsonl"
    p.parent.mkdir()
    p.write_bytes(b"line\n")
    entries = file_entries([p], base_dir=tmp_path)
    assert entries[0].path == "a/b.jsonl"
    assert entries[0].bytes == 5
    assert len(entries[0].sha256) == 64


def test_discover_files_globs_jsonl_and_gz(tmp_path: Path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "shard-00000.jsonl").write_bytes(b"a\n")
    (tmp_path / "out" / "shard-00001.jsonl.gz").write_bytes(b"b\n")
    (tmp_path / "out" / "ignored.txt").write_bytes(b"c\n")
    entries = discover_files(tmp_path / "out")
    paths = sorted(e.path for e in entries)
    assert paths == ["shard-00000.jsonl", "shard-00001.jsonl.gz"]


def test_make_run_id_combines_time_and_hash():
    now = now_utc()
    rid = make_run_id("0123456789abcdef" * 4, now)
    assert rid.startswith("run-")
    # Hash prefix preserved.
    assert "01234567" in rid


def test_write_then_load_roundtrip(tmp_path: Path):
    started = now_utc()
    m = RunManifest(
        run_id="run-x",
        pipeline_version="0.0.0",
        created_at=started,
        config_hash="abcd",
        stages=[
            StageEntry(name="ingest", started_at=started, finished_at=started, files=[])
        ],
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, m)
    loaded = load_manifest(out)
    assert loaded.run_id == "run-x"
    assert loaded.stages[0].name == "ingest"


def test_verify_manifest_passes_when_files_match(tmp_path: Path):
    f = tmp_path / "shard-00000.jsonl"
    f.write_bytes(b"hello\n")
    started = now_utc()
    m = RunManifest(
        run_id="r",
        pipeline_version="0.0.0",
        created_at=started,
        stages=[
            StageEntry(
                name="export",
                started_at=started,
                finished_at=started,
                files=file_entries([f], base_dir=tmp_path),
            )
        ],
    )
    assert verify_manifest(m, base_dir=tmp_path) == []


def test_verify_manifest_detects_tampering(tmp_path: Path):
    f = tmp_path / "shard-00000.jsonl"
    f.write_bytes(b"hello\n")
    started = now_utc()
    m = RunManifest(
        run_id="r",
        pipeline_version="0.0.0",
        created_at=started,
        stages=[
            StageEntry(
                name="export",
                started_at=started,
                finished_at=started,
                files=file_entries([f], base_dir=tmp_path),
            )
        ],
    )
    # Modify the file after manifest was written.
    f.write_bytes(b"tampered\n")
    errors = verify_manifest(m, base_dir=tmp_path)
    assert errors
    assert "hash mismatch" in errors[0]


def test_verify_manifest_detects_missing(tmp_path: Path):
    f = tmp_path / "shard-00000.jsonl"
    f.write_bytes(b"hello\n")
    started = now_utc()
    m = RunManifest(
        run_id="r",
        pipeline_version="0.0.0",
        created_at=started,
        stages=[
            StageEntry(
                name="export",
                started_at=started,
                finished_at=started,
                files=file_entries([f], base_dir=tmp_path),
            )
        ],
    )
    f.unlink()
    errors = verify_manifest(m, base_dir=tmp_path)
    assert errors
    assert "missing" in errors[0]
