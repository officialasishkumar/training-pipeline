"""CLI entry point.

Each subcommand is a thin shell over the corresponding library function so
the same code paths are tested by ``pytest`` and exercised in production.

::

    tp ingest    --input PATH --output PATH [--source NAME]
    tp redact    --input PATH --output PATH [--rules FILE] [--audit FILE --audit-rate 0.05]
    tp tag       --input PATH --output PATH
    tp validate  --input PATH [--tool-registry FILE] [--issues FILE]
    tp split     --input PATH --output-dir DIR [--fractions 0.8 0.1 0.1]
    tp export sft --input PATH --output-dir DIR [--template chatml] [--shard-size 5000]
    tp export dpo --input PATH --output-dir DIR [--strategy feedback]
    tp run       --config FILE   # full pipeline
    tp eval      --student FILE --teacher FILE [--report FILE]
    tp version
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from training_pipeline import __version__
from training_pipeline.config import load_pipeline_config
from training_pipeline.export.dpo import DPOPairStrategy, export_dpo_jsonl
from training_pipeline.export.sft import iter_sft_records
from training_pipeline.export.shards import ShardWriter, write_dataset_card
from training_pipeline.ingest.normalizer import NormalizationError, normalize_records
from training_pipeline.ingest.parsers import iter_records, write_jsonl
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

app = typer.Typer(
    name="tp",
    help="training-pipeline: logs to SFT/DPO datasets for agentic LLMs.",
    no_args_is_help=True,
    add_completion=False,
)
export_app = typer.Typer(name="export", help="Build SFT or DPO datasets.")
app.add_typer(export_app, name="export")

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
        Optional[str],
        typer.Option("--source", "-s", help="Force a source adapter (default: auto)"),
    ] = None,
    quarantine: Annotated[
        Optional[Path],
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
        Optional[Path], typer.Option("--rules", help="YAML rule file (extends built-ins)")
    ] = None,
    audit: Annotated[Optional[Path], typer.Option("--audit", help="Audit JSONL output")] = None,
    audit_rate: Annotated[float, typer.Option("--audit-rate")] = 0.05,
    audit_seed: Annotated[int, typer.Option("--audit-seed")] = 0,
    audit_cap: Annotated[int, typer.Option("--audit-cap")] = 1000,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Detect PII and write redacted Trajectory JSONL with placeholders."""
    _setup_logging(verbose)
    rule_set = load_rules(rules) if rules else BUILTIN_RULES
    redactor = Redactor(rules=rule_set)
    sampler = AuditSampler(rate=audit_rate, seed=audit_seed, cap=audit_cap)
    totals: dict[str, int] = {}

    def _gen() -> Iterator[Trajectory]:
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
            yield res.trajectory

    write_jsonl(output, _gen())
    if audit:
        write_jsonl(audit, sampler.consume())

    table = Table(title="PII redaction summary")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    for k, v in sorted(totals.items()):
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"[green]redact:[/green] → {output}")


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
        Optional[Path],
        typer.Option("--tool-registry", help="YAML registry of valid tools and arg schemas"),
    ] = None,
    issues_output: Annotated[
        Optional[Path],
        typer.Option("--issues-output", help="Write all issues to this JSONL"),
    ] = None,
    output: Annotated[
        Optional[Path],
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
        Optional[list[str]],
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
    system_prompt: Annotated[Optional[str], typer.Option("--system-prompt")] = None,
    shard_size: Annotated[int, typer.Option("--shard-size")] = 5000,
    compress: Annotated[bool, typer.Option("--compress")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Build SFT JSONL shards (chat-template aligned) with a dataset card."""
    _setup_logging(verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with ShardWriter(
        output_dir,
        shard_size=shard_size,
        prefix="sft",
        compress=compress,
    ) as writer:
        for record in iter_sft_records(_iter_trajectories(input), system_prompt=system_prompt):
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
    )
    console.print(f"[green]export sft:[/green] {n} records → {output_dir}")


@export_app.command("dpo")
def export_dpo_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    strategy: Annotated[str, typer.Option("--strategy")] = "feedback",
    system_prompt: Annotated[Optional[str], typer.Option("--system-prompt")] = None,
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
        _from_failure_recovery,  # noqa: SLF001 - intentional reuse
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
                console.print(
                    f"[red]export dpo:[/red] strategy {strategy!r} not supported via CLI"
                )
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
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full pipeline from a YAML config."""
    _setup_logging(verbose)
    cfg = load_pipeline_config(config)
    console.rule(f"[bold]{cfg.name}")
    console.print("[1/6] ingest")
    ingest(
        input=Path(cfg.ingest.input),
        output=Path(cfg.ingest.output),
        source=cfg.ingest.source,
        quarantine=Path(cfg.ingest.quarantine) if cfg.ingest.quarantine else None,
        verbose=verbose,
    )
    console.print("[2/6] redact")
    redact(
        input=Path(cfg.pii.input),
        output=Path(cfg.pii.output),
        rules=Path(cfg.pii.rules_file) if cfg.pii.rules_file else None,
        audit=Path(cfg.pii.audit_output) if cfg.pii.audit_output else None,
        audit_rate=cfg.pii.audit_rate,
        audit_seed=cfg.pii.audit_seed,
        audit_cap=cfg.pii.audit_cap,
        verbose=verbose,
    )
    console.print("[3/6] tag")
    tag(
        input=Path(cfg.tag.input),
        output=Path(cfg.tag.output),
        verbose=verbose,
    )
    console.print("[4/6] validate")
    validate(
        input=Path(cfg.validation.input),
        tool_registry=Path(cfg.validation.tool_registry) if cfg.validation.tool_registry else None,
        issues_output=Path(cfg.validation.issues_output) if cfg.validation.issues_output else None,
        output=Path(cfg.validation.output)
        if cfg.validation.output and cfg.validation.drop_on_error
        else None,
        verbose=verbose,
    )
    console.print("[5/6] export sft")
    export_sft_cmd(
        input=Path(cfg.sft.input),
        output_dir=Path(cfg.sft.output_dir),
        template=cfg.sft.template,
        system_prompt=cfg.sft.system_prompt,
        shard_size=cfg.sft.shard_size,
        compress=cfg.sft.compress,
        verbose=verbose,
    )
    console.print("[6/6] export dpo")
    export_dpo_cmd(
        input=Path(cfg.dpo.input),
        output_dir=Path(cfg.dpo.output_dir),
        strategy=cfg.dpo.strategy,
        system_prompt=cfg.dpo.system_prompt,
        shard_size=cfg.dpo.shard_size,
        compress=cfg.dpo.compress,
        verbose=verbose,
    )
    console.rule("[green]done")


@app.command()
def eval(
    student: Annotated[Path, typer.Option("--student")],
    teacher: Annotated[Path, typer.Option("--teacher")],
    eval_set: Annotated[
        Path,
        typer.Option("--eval-set", help="JSONL of eval prompts and ground-truth tool calls"),
    ],
    report: Annotated[Optional[Path], typer.Option("--report")] = None,
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
