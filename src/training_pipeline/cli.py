"""CLI entry point.

Each subcommand is a thin shell over the corresponding library function so
the same code paths are tested by ``pytest`` and exercised in production.

::

    tp ingest             --input PATH --output PATH [--source NAME]
    tp redact             --input PATH --output PATH [--rules FILE] [--audit FILE]
                          [--quarantine PATH] [--fail-on-leak]
    tp tag                --input PATH --output PATH
    tp validate           --input PATH [--tool-registry FILE] [--issues FILE]
    tp validate-template  --input PATH [--template chatml] [--tokenizer hf:<id>]
                          [--max-tokens 8192] [--report FILE]
    tp split              --input PATH --output-dir DIR [--fractions 0.8 0.1 0.1]
    tp export sft         --input PATH --output-dir DIR [--template chatml]
                          [--shard-size 5000] [--loss-policy assistant_only]
    tp export dpo         --input PATH --output-dir DIR [--strategy feedback]
    tp run                --config FILE [--manifest PATH]   # full pipeline
    tp manifest show      PATH
    tp manifest verify    PATH [--base-dir DIR]
    tp hash-config        --config FILE
    tp eval               --student FILE --teacher FILE [--report FILE]
    tp generate seeds         --input PATH --output PATH [--embedder NAME]
                              [--cluster-method NAME] [--n-clusters K]
    tp generate trajectories  --seeds PATH --output PATH --tool-registry FILE
                              [--backend stub|transformers|vllm] [--model ID]
                              [--fixtures-dir DIR] [--max-steps N]
    tp generate stratify      --input PATH --output PATH [--cap-per-bucket N]
    tp version
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from training_pipeline import __version__
from training_pipeline.config import load_pipeline_config
from training_pipeline.export.dpo import DPOPairStrategy
from training_pipeline.export.sft import iter_sft_records
from training_pipeline.export.shards import ShardWriter, write_dataset_card
from training_pipeline.ingest.normalizer import NormalizationError, normalize_records
from training_pipeline.ingest.parsers import iter_records, write_jsonl
from training_pipeline.manifest import (
    RunManifest,
    StageEntry,
    discover_files,
    file_entries,
    hash_obj,
    load_manifest,
    make_run_id,
    now_utc,
    verify_manifest,
    write_manifest,
)
from training_pipeline.pii.audit import AuditSampler
from training_pipeline.pii.redactor import Redactor
from training_pipeline.pii.rules import BUILTIN_RULES, load_rules
from training_pipeline.schemas.events import Trajectory
from training_pipeline.tagging.complexity import tag_trajectory
from training_pipeline.tagging.stratify import stratified_split
from training_pipeline.validate.consistency import (
    ToolRegistry,
    validate_consistency,
)
from training_pipeline.validate.splits import split_integrity_report
from training_pipeline.validate.template_dryrun import dryrun_jsonl

app = typer.Typer(
    name="tp",
    help="training-pipeline: logs to SFT/DPO datasets for agentic LLMs.",
    no_args_is_help=True,
    add_completion=False,
)
export_app = typer.Typer(name="export", help="Build SFT or DPO datasets.")
app.add_typer(export_app, name="export")
generate_app = typer.Typer(
    name="generate",
    help="Synthesise diverse trajectories from log seeds + a mock tool environment.",
)
app.add_typer(generate_app, name="generate")

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _iter_trajectories(path: str | Path) -> Iterator[Trajectory]:
    for rec in iter_records(path):
        yield Trajectory.model_validate(rec)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"training-pipeline {__version__}")


@app.command()
def ingest(
    input: Annotated[Path, typer.Option("--input", "-i", help="File or directory of raw logs")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Canonical JSONL output")],
    source: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Force a source adapter (default: auto)"),
    ] = None,
    quarantine: Annotated[
        Path | None,
        typer.Option("--quarantine", "-q", help="Where to write records that failed to normalize"),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Read raw logs and write canonical Trajectory JSONL."""
    _setup_logging(verbose)
    n_ok = 0
    n_err = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    quarantine_fh = None
    if quarantine:
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        quarantine_fh = quarantine.open("wb")

    def _gen() -> Iterator[Any]:
        nonlocal n_ok, n_err
        for item in normalize_records(iter_records(input), source=source):
            if isinstance(item, NormalizationError):
                n_err += 1
                if quarantine_fh:
                    import orjson

                    quarantine_fh.write(
                        orjson.dumps(
                            {"index": item.index, "error": item.error, "record": item.record},
                            option=orjson.OPT_APPEND_NEWLINE,
                        )
                    )
                continue
            n_ok += 1
            yield item

    try:
        write_jsonl(output, _gen())
    finally:
        if quarantine_fh:
            quarantine_fh.close()
    console.print(f"[green]ingest:[/green] {n_ok} normalized, {n_err} quarantined → {output}")


@app.command()
def redact(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    rules: Annotated[
        Path | None, typer.Option("--rules", help="YAML rule file (extends built-ins)")
    ] = None,
    audit: Annotated[Path | None, typer.Option("--audit", help="Audit JSONL output")] = None,
    audit_rate: Annotated[float, typer.Option("--audit-rate")] = 0.05,
    audit_seed: Annotated[int, typer.Option("--audit-seed")] = 0,
    audit_cap: Annotated[int, typer.Option("--audit-cap")] = 1000,
    quarantine: Annotated[
        Path | None,
        typer.Option(
            "--quarantine",
            help=(
                "Where to write trajectories with surviving PII after redaction. "
                "When set, leaked rows are routed here and not to --output."
            ),
        ),
    ] = None,
    fail_on_leak: Annotated[
        bool,
        typer.Option(
            "--fail-on-leak",
            help="Exit non-zero if any trajectory has surviving PII after redaction.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Detect PII and write redacted Trajectory JSONL with placeholders.

    A defense-in-depth pass re-runs the detectors on the redacted output. Rows
    that still match a rule are tagged ``pii_leak_count`` and — when
    ``--quarantine`` is set — written there instead of the main output.
    """
    _setup_logging(verbose)
    rule_set = load_rules(rules) if rules else BUILTIN_RULES
    redactor = Redactor(rules=rule_set)
    sampler = AuditSampler(rate=audit_rate, seed=audit_seed, cap=audit_cap)
    totals: dict[str, int] = {}
    n_leaks = 0
    leak_records: list[dict[str, Any]] = []

    def _gen() -> Iterator[Trajectory]:
        nonlocal n_leaks
        for traj in _iter_trajectories(input):
            res = redactor.redact_trajectory(traj, record_for_audit=bool(audit))
            for k, v in res.report.items():
                totals[k] = totals.get(k, 0) + v
            if audit and res.sample:
                rec = {
                    "session_id": traj.session_id,
                    "domain": traj.domain,
                    "events": res.sample,
                }
                sampler.consider(rec, key=traj.session_id)
            if res.has_leaks:
                n_leaks += 1
                leak_records.append(
                    {
                        "session_id": traj.session_id,
                        "lineage_id": traj.lineage_id,
                        "leaks": [
                            {
                                "event_id": lk.event_id,
                                "rule": lk.rule,
                                "category": lk.category,
                                "field": lk.field,
                            }
                            for lk in res.leaks
                        ],
                    }
                )
                if quarantine:
                    # Skip leaked rows from the main stream.
                    continue
            yield res.trajectory

    write_jsonl(output, _gen())
    if audit:
        write_jsonl(audit, sampler.consume())
    if quarantine and leak_records:
        import orjson

        quarantine.parent.mkdir(parents=True, exist_ok=True)
        with quarantine.open("wb") as fh:
            for r in leak_records:
                fh.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))

    table = Table(title="PII redaction summary")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    for k, v in sorted(totals.items()):
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[green]redact:[/green] → {output}")
    if n_leaks:
        msg = f"[red]leakage:[/red] {n_leaks} trajectories had surviving PII"
        if quarantine:
            msg += f" → {quarantine}"
        console.print(msg)
        if fail_on_leak:
            raise typer.Exit(code=2)
    else:
        console.print("[green]leakage:[/green] none")


@app.command()
def tag(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Compute trajectory complexity, recovery, and ambiguity tags."""
    _setup_logging(verbose)
    counts: dict[str, int] = {}

    def _gen() -> Iterator[Trajectory]:
        for traj in _iter_trajectories(input):
            tagged = tag_trajectory(traj)
            band = tagged.tags["complexity"]["complexity_band"]
            counts[band] = counts.get(band, 0) + 1
            yield tagged

    write_jsonl(output, _gen())
    table = Table(title="Complexity bands")
    table.add_column("Band")
    table.add_column("Count", justify="right")
    for band, c in sorted(counts.items(), key=lambda x: x[0]):
        table.add_row(band, str(c))
    console.print(table)


@app.command()
def validate(
    input: Annotated[Path, typer.Option("--input", "-i")],
    tool_registry: Annotated[
        Path | None,
        typer.Option("--tool-registry", help="YAML registry of valid tools and arg schemas"),
    ] = None,
    issues_output: Annotated[
        Path | None,
        typer.Option("--issues-output", help="Write all issues to this JSONL"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="If set, write trajectories *without* errors to this JSONL (drop-on-error).",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Validate trajectories against tool registry and consistency rules."""
    _setup_logging(verbose)
    registry = ToolRegistry.from_yaml(tool_registry) if tool_registry else None
    issue_counts: dict[str, int] = {}
    n_total = 0
    n_with_error = 0
    issues_fh = issues_output.open("wb") if issues_output else None

    def _gen() -> Iterator[Trajectory]:
        nonlocal n_total, n_with_error
        for traj in _iter_trajectories(input):
            n_total += 1
            issues = validate_consistency(traj, registry=registry)
            has_error = any(i.severity == "error" for i in issues)
            if has_error:
                n_with_error += 1
            for i in issues:
                issue_counts[i.code] = issue_counts.get(i.code, 0) + 1
                if issues_fh:
                    import orjson

                    issues_fh.write(
                        orjson.dumps(
                            {
                                "session_id": traj.session_id,
                                "code": i.code,
                                "severity": i.severity,
                                "message": i.message,
                                "event_id": i.event_id,
                                "extra": i.extra,
                            },
                            option=orjson.OPT_APPEND_NEWLINE,
                        )
                    )
            if output and has_error:
                continue
            yield traj

    if output:
        write_jsonl(output, _gen())
    else:
        for _ in _gen():
            pass

    if issues_fh:
        issues_fh.close()

    table = Table(title=f"Validation summary ({n_total} trajectories, {n_with_error} with errors)")
    table.add_column("Issue code")
    table.add_column("Count", justify="right")
    for k, v in sorted(issue_counts.items()):
        table.add_row(k, str(v))
    console.print(table)
    if n_with_error and not output:
        console.print(
            f"[yellow]validate:[/yellow] {n_with_error}/{n_total} trajectories have errors. "
            "Pass --output to drop them."
        )


@app.command()
def split(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    fractions: Annotated[
        tuple[float, float, float],
        typer.Option("--fractions", help="train/val/test fractions"),
    ] = (0.8, 0.1, 0.1),
    seed: Annotated[int, typer.Option("--seed")] = 0,
    keys: Annotated[
        list[str] | None,
        typer.Option("--key", "-k", help="Stratification keys (repeatable)"),
    ] = None,
    threshold: Annotated[float, typer.Option("--near-duplicate-threshold")] = 0.85,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Stratified split + near-duplicate leakage report."""
    _setup_logging(verbose)
    trajectories = list(_iter_trajectories(input))
    keys_list = keys or ["complexity_band", "domain"]
    s = stratified_split(trajectories, fractions=fractions, seed=seed, keys=keys_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    sets = {
        "train": [trajectories[i] for i in s.train],
        "val": [trajectories[i] for i in s.val],
        "test": [trajectories[i] for i in s.test],
    }
    for name, items in sets.items():
        write_jsonl(output_dir / f"{name}.jsonl", items)

    leak_report = split_integrity_report(sets, threshold=threshold)

    import orjson

    (output_dir / "split_report.json").write_bytes(
        orjson.dumps(
            {
                "fractions": list(fractions),
                "seed": seed,
                "keys": keys_list,
                "strata": s.strata,
                "near_duplicate_report": leak_report,
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    table = Table(title="Split summary")
    table.add_column("Split")
    table.add_column("Count", justify="right")
    for name, items in sets.items():
        table.add_row(name, str(len(items)))
    console.print(table)
    if leak_report["total_leaks"]:
        console.print(
            f"[red]split:[/red] {leak_report['total_leaks']} near-duplicate leaks across pairs!"
        )
    else:
        console.print("[green]split:[/green] no near-duplicate leakage")


@export_app.command("sft")
def export_sft_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    template: Annotated[str, typer.Option("--template")] = "chatml",
    system_prompt: Annotated[str | None, typer.Option("--system-prompt")] = None,
    shard_size: Annotated[int, typer.Option("--shard-size")] = 5000,
    compress: Annotated[bool, typer.Option("--compress")] = False,
    loss_policy: Annotated[
        str,
        typer.Option(
            "--loss-policy",
            help=(
                "Per-message loss weighting policy: 'assistant_only' (default), "
                "'assistant_text_only', or 'none' to skip emission."
            ),
        ),
    ] = "assistant_only",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build SFT JSONL shards (chat-template aligned) with a dataset card."""
    _setup_logging(verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    policy: str | None = None if loss_policy == "none" else loss_policy
    with ShardWriter(
        output_dir,
        shard_size=shard_size,
        prefix="sft",
        compress=compress,
    ) as writer:
        for record in iter_sft_records(
            _iter_trajectories(input),
            system_prompt=system_prompt,
            loss_policy=policy,
        ):
            writer.write(record)
            n += 1
        fingerprint = writer.fingerprint()
    write_dataset_card(
        output_dir,
        name="sft",
        record_count=n,
        fingerprint=fingerprint,
        fields=["messages", "metadata"],
        chat_template=template,
        extra={"loss_policy": loss_policy},
    )
    console.print(f"[green]export sft:[/green] {n} records → {output_dir}")


@export_app.command("dpo")
def export_dpo_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    strategy: Annotated[str, typer.Option("--strategy")] = "feedback",
    system_prompt: Annotated[str | None, typer.Option("--system-prompt")] = None,
    shard_size: Annotated[int, typer.Option("--shard-size")] = 5000,
    compress: Annotated[bool, typer.Option("--compress")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build DPO JSONL shards with prompt/chosen/rejected and a dataset card."""
    _setup_logging(verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    # We need a sharded writer but DPO records flow from export_dpo_jsonl;
    # implement sharding via a generator + ShardWriter directly.
    from training_pipeline.export.dpo import (
        _from_failure_recovery,
        _from_feedback,
    )

    strat = DPOPairStrategy(strategy)
    n = 0
    with ShardWriter(
        output_dir,
        shard_size=shard_size,
        prefix="dpo",
        compress=compress,
    ) as writer:
        for traj in _iter_trajectories(input):
            if strat is DPOPairStrategy.FEEDBACK:
                rows = _from_feedback(traj, system_prompt=system_prompt)
            elif strat is DPOPairStrategy.FAILURE_RECOVERY:
                rows = _from_failure_recovery(traj, system_prompt=system_prompt)
            else:
                console.print(f"[red]export dpo:[/red] strategy {strategy!r} not supported via CLI")
                raise typer.Exit(code=2)
            for record in rows:
                writer.write(record)
                n += 1
        fingerprint = writer.fingerprint()
    write_dataset_card(
        output_dir,
        name="dpo",
        record_count=n,
        fingerprint=fingerprint,
        fields=["prompt", "chosen", "rejected", "metadata"],
        extra={"strategy": strategy},
    )
    console.print(f"[green]export dpo:[/green] {n} records → {output_dir}")


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c")],
    manifest_out: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help=(
                "Write a run manifest (config hash, code version, per-stage file "
                "hashes) to this path for reproducibility and audit."
            ),
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full pipeline from a YAML config.

    With ``--manifest`` set, every stage's outputs are hashed and recorded
    alongside the config snapshot. ``tp manifest verify`` later confirms the
    on-disk build matches the manifest, catching silent drift.
    """
    _setup_logging(verbose)
    cfg = load_pipeline_config(config)
    cfg_dict = cfg.model_dump(mode="json")
    cfg_hash = hash_obj(cfg_dict)
    started = now_utc()
    # Anchor every manifest path to the cwd so a single --base-dir resolves all
    # of them at verify time. Without this, ``discover_files`` and
    # ``file_entries`` use different bases and verify can't find anything.
    manifest_anchor = Path.cwd()
    manifest = RunManifest(
        run_id=make_run_id(cfg_hash, started),
        pipeline_version=__version__,
        created_at=started,
        config_hash=cfg_hash,
        config_snapshot=cfg_dict,
    )

    def _stage(name: str, started_at: Any, files: list[Any], counters: dict[str, int] | None = None) -> None:
        if manifest_out:
            manifest.stages.append(
                StageEntry(
                    name=name,
                    started_at=started_at,
                    finished_at=now_utc(),
                    files=files,
                    counters=counters or {},
                )
            )

    n_optional = sum(
        bool(x) for x in (cfg.seeds.enabled, cfg.generate.enabled, cfg.stratify.enabled)
    )
    n_total_stages = 6 + n_optional
    stage_idx = 0

    def _step(label: str) -> str:
        nonlocal stage_idx
        stage_idx += 1
        return f"[{stage_idx}/{n_total_stages}] {label}"

    console.rule(f"[bold]{cfg.name}")
    console.print(_step("ingest"))
    s_started = now_utc()
    ingest(
        input=Path(cfg.ingest.input),
        output=Path(cfg.ingest.output),
        source=cfg.ingest.source,
        quarantine=Path(cfg.ingest.quarantine) if cfg.ingest.quarantine else None,
        verbose=verbose,
    )
    if manifest_out:
        files = file_entries([cfg.ingest.output], role="output", base_dir=manifest_anchor)
        if cfg.ingest.quarantine and Path(cfg.ingest.quarantine).exists():
            files.extend(
                file_entries([cfg.ingest.quarantine], role="quarantine", base_dir=manifest_anchor)
            )
        _stage("ingest", s_started, files)

    console.print(_step("redact"))
    s_started = now_utc()
    redact(
        input=Path(cfg.pii.input),
        output=Path(cfg.pii.output),
        rules=Path(cfg.pii.rules_file) if cfg.pii.rules_file else None,
        audit=Path(cfg.pii.audit_output) if cfg.pii.audit_output else None,
        audit_rate=cfg.pii.audit_rate,
        audit_seed=cfg.pii.audit_seed,
        audit_cap=cfg.pii.audit_cap,
        quarantine=Path(cfg.pii.quarantine) if cfg.pii.quarantine else None,
        fail_on_leak=cfg.pii.fail_on_leak,
        verbose=verbose,
    )
    if manifest_out:
        files = file_entries([cfg.pii.output], role="output", base_dir=manifest_anchor)
        if cfg.pii.audit_output and Path(cfg.pii.audit_output).exists():
            files.extend(
                file_entries([cfg.pii.audit_output], role="audit", base_dir=manifest_anchor)
            )
        if cfg.pii.quarantine and Path(cfg.pii.quarantine).exists():
            files.extend(
                file_entries([cfg.pii.quarantine], role="quarantine", base_dir=manifest_anchor)
            )
        _stage("redact", s_started, files)

    if cfg.seeds.enabled:
        console.print(_step("seeds"))
        s_started = now_utc()
        generate_seeds_cmd(
            input=Path(cfg.seeds.input),
            output=Path(cfg.seeds.output),
            embedder=cfg.seeds.embedder,
            embedder_model=cfg.seeds.embedder_model,
            cluster_method=cfg.seeds.cluster_method,
            n_clusters=cfg.seeds.n_clusters,
            similarity_threshold=cfg.seeds.similarity_threshold,
            seed=cfg.seeds.seed,
            verbose=verbose,
        )
        if manifest_out:
            _stage(
                "seeds",
                s_started,
                file_entries([cfg.seeds.output], role="output", base_dir=manifest_anchor),
            )

    if cfg.generate.enabled:
        console.print(_step("generate"))
        s_started = now_utc()
        if not cfg.generate.tool_registry:
            raise typer.BadParameter(
                "generate.tool_registry is required when generate.enabled is true"
            )
        generate_trajectories_cmd(
            seeds=Path(cfg.generate.seeds_input),
            output=Path(cfg.generate.output),
            tool_registry=Path(cfg.generate.tool_registry),
            fixtures_dir=Path(cfg.generate.fixtures_dir) if cfg.generate.fixtures_dir else None,
            backend=cfg.generate.backend,
            model_id=cfg.generate.model_id,
            max_steps=cfg.generate.max_steps,
            drop_on_invalid_args=cfg.generate.drop_on_invalid_args,
            system_prompt=cfg.generate.system_prompt,
            seed=cfg.generate.seed,
            verbose=verbose,
        )
        if manifest_out:
            _stage(
                "generate",
                s_started,
                file_entries([cfg.generate.output], role="output", base_dir=manifest_anchor),
            )

    console.print(_step("tag"))
    s_started = now_utc()
    tag(
        input=Path(cfg.tag.input),
        output=Path(cfg.tag.output),
        verbose=verbose,
    )
    if manifest_out:
        _stage(
            "tag",
            s_started,
            file_entries([cfg.tag.output], role="output", base_dir=manifest_anchor),
        )

    console.print(_step("validate"))
    s_started = now_utc()
    validate(
        input=Path(cfg.validation.input),
        tool_registry=Path(cfg.validation.tool_registry) if cfg.validation.tool_registry else None,
        issues_output=Path(cfg.validation.issues_output) if cfg.validation.issues_output else None,
        output=Path(cfg.validation.output)
        if cfg.validation.output and cfg.validation.drop_on_error
        else None,
        verbose=verbose,
    )
    if manifest_out:
        files = []
        if cfg.validation.output and Path(cfg.validation.output).exists():
            files.extend(
                file_entries([cfg.validation.output], role="output", base_dir=manifest_anchor)
            )
        if cfg.validation.issues_output and Path(cfg.validation.issues_output).exists():
            files.extend(
                file_entries(
                    [cfg.validation.issues_output], role="issues", base_dir=manifest_anchor
                )
            )
        _stage("validate", s_started, files)

    if cfg.stratify.enabled:
        console.print(_step("stratify"))
        s_started = now_utc()
        generate_stratify_cmd(
            input=Path(cfg.stratify.input),
            output=Path(cfg.stratify.output),
            cap_per_bucket=cfg.stratify.cap_per_bucket,
            verbose=verbose,
        )
        if manifest_out:
            _stage(
                "stratify",
                s_started,
                file_entries([cfg.stratify.output], role="output", base_dir=manifest_anchor),
            )

    console.print(_step("export sft"))
    s_started = now_utc()
    export_sft_cmd(
        input=Path(cfg.sft.input),
        output_dir=Path(cfg.sft.output_dir),
        template=cfg.sft.template,
        system_prompt=cfg.sft.system_prompt,
        shard_size=cfg.sft.shard_size,
        compress=cfg.sft.compress,
        loss_policy=cfg.sft.loss_policy,
        verbose=verbose,
    )
    if manifest_out:
        _stage(
            "export_sft",
            s_started,
            discover_files(cfg.sft.output_dir, role="shard", base_dir=manifest_anchor),
        )

    console.print(_step("export dpo"))
    s_started = now_utc()
    export_dpo_cmd(
        input=Path(cfg.dpo.input),
        output_dir=Path(cfg.dpo.output_dir),
        strategy=cfg.dpo.strategy,
        system_prompt=cfg.dpo.system_prompt,
        shard_size=cfg.dpo.shard_size,
        compress=cfg.dpo.compress,
        verbose=verbose,
    )
    if manifest_out:
        _stage(
            "export_dpo",
            s_started,
            discover_files(cfg.dpo.output_dir, role="shard", base_dir=manifest_anchor),
        )

    if manifest_out:
        write_manifest(manifest_out, manifest)
        console.print(f"[green]manifest:[/green] {manifest.run_id} → {manifest_out}")

    console.rule("[green]done")


# ---------------------------------------------------------------------------
# Generate subcommands — seeds → trajectories → stratify
# ---------------------------------------------------------------------------


@generate_app.command("seeds")
def generate_seeds_cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Canonical Trajectory JSONL")],
    output: Annotated[Path, typer.Option("--output", "-o", help="seeds.jsonl output")],
    embedder: Annotated[
        str,
        typer.Option(
            "--embedder",
            help="'hash' (default, no deps) or 'sentence-transformers'.",
        ),
    ] = "hash",
    embedder_model: Annotated[
        str,
        typer.Option("--embedder-model"),
    ] = "sentence-transformers/all-MiniLM-L6-v2",
    cluster_method: Annotated[
        str,
        typer.Option("--cluster-method", help="'greedy' (default) or 'kmeans'."),
    ] = "greedy",
    n_clusters: Annotated[
        int | None,
        typer.Option("--n-clusters", help="KMeans cluster count; auto-sized when unset."),
    ] = None,
    similarity_threshold: Annotated[
        float,
        typer.Option(
            "--similarity-threshold",
            help="Greedy clustering: min cosine to merge into an existing cluster.",
        ),
    ] = 0.72,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Cluster user questions across canonical logs and emit one seed per cluster."""
    _setup_logging(verbose)
    from training_pipeline.generate.seeds import SeedExtractor

    extractor = SeedExtractor(
        embedder=embedder,  # type: ignore[arg-type]
        embedder_model=embedder_model,
        cluster_method=cluster_method,  # type: ignore[arg-type]
        n_clusters=n_clusters,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    n = extractor.extract_to_jsonl(input, output)
    console.print(f"[green]generate seeds:[/green] {n} seeds → {output}")


@generate_app.command("trajectories")
def generate_trajectories_cmd(
    seeds: Annotated[Path, typer.Option("--seeds", "-s", help="seeds.jsonl input")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    tool_registry: Annotated[
        Path,
        typer.Option("--tool-registry", help="YAML registry of tools and arg schemas."),
    ],
    fixtures_dir: Annotated[
        Path | None,
        typer.Option("--fixtures-dir", help="Directory of deterministic tool fixtures."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option("--backend", help="'stub', 'transformers', or 'vllm'."),
    ] = "stub",
    model_id: Annotated[
        str,
        typer.Option("--model", help="HF model id used by transformers/vllm backends."),
    ] = "Qwen/Qwen2.5-7B-Instruct",
    max_steps: Annotated[int, typer.Option("--max-steps")] = 5,
    drop_on_invalid_args: Annotated[
        bool,
        typer.Option(
            "--drop-on-invalid-args/--keep-invalid-args",
            help="Drop trajectories whose tool args fail the registry schema.",
        ),
    ] = True,
    system_prompt: Annotated[str | None, typer.Option("--system-prompt")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Drive an LLM + mock tools to produce synthetic trajectories from seeds."""
    _setup_logging(verbose)
    from training_pipeline.generate.generator import (
        StubLLMBackend,
        TrajectoryGenerator,
        TransformersLLMBackend,
        VLLMBackend,
    )
    from training_pipeline.generate.mock_tools import MockToolRegistry

    backend_obj: Any
    if backend == "stub":
        backend_obj = StubLLMBackend()
    elif backend == "transformers":
        backend_obj = TransformersLLMBackend(model_id=model_id)
    elif backend == "vllm":
        backend_obj = VLLMBackend(model_id=model_id)
    else:
        console.print(f"[red]generate trajectories:[/red] unknown backend {backend!r}")
        raise typer.Exit(code=2)

    mock = MockToolRegistry.from_config(
        tool_registry,
        fixtures_dir=fixtures_dir,
        seed=seed,
    )
    gen = TrajectoryGenerator(
        backend=backend_obj,
        mock_tools=mock,
        max_steps=max_steps,
        drop_on_invalid_args=drop_on_invalid_args,
        system_prompt=system_prompt,
        seed=seed,
    )
    written, dropped = gen.generate_to_jsonl(seeds, output)
    console.print(
        f"[green]generate trajectories:[/green] {written} written, {dropped} dropped → {output}"
    )


@generate_app.command("stratify")
def generate_stratify_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    cap_per_bucket: Annotated[
        int | None,
        typer.Option(
            "--cap-per-bucket",
            help="Max trajectories per (difficulty x edge-case) bucket. None to keep all.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Cap-per-bucket sampler over (difficulty x edge-case) buckets.

    Use this *before* ``tp split`` to avoid exporting a dataset that's 90 %
    "easy/single_tool" because that's what the long tail of logs produces.
    """
    _setup_logging(verbose)
    from training_pipeline.generate.difficulty import annotate, stratify

    counts: dict[str, int] = {}

    def _gen() -> Iterator[Trajectory]:
        for traj in _iter_trajectories(input):
            yield annotate(traj)

    items = list(_gen())
    sampled = stratify(items, cap_per_bucket=cap_per_bucket)
    for traj in sampled:
        d = traj.tags.get("difficulty", {})
        key = f"{d.get('tier', '?')}::{','.join(d.get('edge_cases', []) or ['none'])}"
        counts[key] = counts.get(key, 0) + 1
    write_jsonl(output, sampled)

    table = Table(title=f"Stratified output ({len(sampled)}/{len(items)})")
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        table.add_row(k, str(v))
    console.print(table)


manifest_app = typer.Typer(name="manifest", help="Inspect and verify run manifests.")
app.add_typer(manifest_app, name="manifest")


@manifest_app.command("verify")
def manifest_verify_cmd(
    manifest: Annotated[Path, typer.Argument(help="Path to manifest.json")],
    base_dir: Annotated[
        Path | None,
        typer.Option(
            "--base-dir",
            help="Directory whose contents the manifest's relative paths resolve against. "
            "Defaults to the manifest's parent directory.",
        ),
    ] = None,
) -> None:
    """Confirm every file in the manifest still hashes to the recorded value.

    Use this before publishing or training: a single edited shard or a partial
    re-export will surface here as a hash mismatch.
    """
    m = load_manifest(manifest)
    base = base_dir or manifest.parent
    errors = verify_manifest(m, base_dir=base)
    if errors:
        for e in errors:
            console.print(f"[red]✗[/red] {e}")
        raise typer.Exit(code=2)
    console.print(
        f"[green]ok:[/green] {sum(len(s.files) for s in m.stages)} files match {m.run_id}"
    )


@manifest_app.command("show")
def manifest_show_cmd(
    manifest: Annotated[Path, typer.Argument(help="Path to manifest.json")],
) -> None:
    """Print a summary of a manifest: run id, code version, file count per stage."""
    m = load_manifest(manifest)
    table = Table(title=f"Manifest {m.run_id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("pipeline_version", m.pipeline_version)
    table.add_row("created_at", m.created_at.isoformat())
    table.add_row("config_hash", m.config_hash or "—")
    table.add_row("stages", ", ".join(s.name for s in m.stages))
    table.add_row("files", str(sum(len(s.files) for s in m.stages)))
    console.print(table)


@app.command("hash-config")
def hash_config_cmd(
    config: Annotated[Path, typer.Option("--config", "-c")],
) -> None:
    """Print the deterministic hash of a pipeline config (used for run ids)."""
    cfg = load_pipeline_config(config)
    console.print(hash_obj(cfg.model_dump(mode="json")))


@app.command("validate-template")
def validate_template_cmd(
    input: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="SFT JSONL file or directory of shards.",
        ),
    ],
    template: Annotated[
        str,
        typer.Option(
            "--template",
            help="Chat template name (chatml, llama3, qwen, gemma, mistral, plain) or 'hf'.",
        ),
    ] = "chatml",
    tokenizer: Annotated[
        str | None,
        typer.Option(
            "--tokenizer",
            help=(
                "Tokenizer spec. None or 'whitespace' uses a cheap fallback. "
                "Use 'hf:<model_id>' for an HF AutoTokenizer."
            ),
        ),
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option("--max-tokens", help="Max context window the trainer will allow."),
    ] = 8192,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write the full report JSON here."),
    ] = None,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the SFT export through render+tokenize and report bad rows.

    This catches bugs that schema validation can't: empty renders, template
    errors on tool-call messages, and trajectories that overflow the model's
    context window. Use ``--tokenizer hf:<model_id>`` for the trainer's exact
    tokenizer; without it, a whitespace fallback gives a conservative estimate.
    """
    _setup_logging(verbose)
    rep = dryrun_jsonl(
        input,
        template=template,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        fail_fast=fail_fast,
    )
    if report:
        import orjson

        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_bytes(orjson.dumps(rep.as_dict(), option=orjson.OPT_INDENT_2))

    table = Table(title=f"Template dry-run ({rep.template} / {rep.tokenizer})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("records", str(rep.n_records))
    table.add_row("failed", str(rep.n_failed))
    table.add_row("over-budget", str(rep.n_overflow))
    table.add_row("max tokens seen", str(rep.max_tokens_seen))
    table.add_row("budget", str(max_tokens))
    console.print(table)
    if not rep.passed:
        console.print(
            f"[red]validate-template:[/red] {rep.n_failed} render failures, "
            f"{rep.n_overflow} over budget"
        )
        raise typer.Exit(code=2)
    console.print("[green]validate-template:[/green] all rows render and fit")


@app.command()
def eval(
    student: Annotated[Path, typer.Option("--student")],
    teacher: Annotated[Path, typer.Option("--teacher")],
    eval_set: Annotated[
        Path,
        typer.Option("--eval-set", help="JSONL of eval prompts and ground-truth tool calls"),
    ],
    report: Annotated[Path | None, typer.Option("--report")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Compare teacher and student outputs on a held-out eval set."""
    _setup_logging(verbose)
    from training_pipeline.eval.compare import compare_outputs

    summary = compare_outputs(student=student, teacher=teacher, eval_set=eval_set)
    if report:
        import orjson

        report.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    table = Table(title="Eval summary")
    table.add_column("Metric")
    table.add_column("Teacher")
    table.add_column("Student")
    for k, vals in summary["metrics"].items():
        table.add_row(k, f"{vals['teacher']:.3f}", f"{vals['student']:.3f}")
    console.print(table)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
