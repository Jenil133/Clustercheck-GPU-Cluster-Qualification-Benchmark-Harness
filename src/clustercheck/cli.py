"""ClusterCheck command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .command import CommandRunner
from .config import ConfigError, load_config
from .engine import (
    SYNTHETIC_NOTICE,
    aggregate,
    attach_fabric_result,
    create_baseline,
    finalize_run,
    initialize_run,
    new_run_id,
    run_node,
    update_submission,
)
from .storage import validate_identifier


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="clustercheck", description="GPU cluster qualification and admission gates")
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run-node", help="run node-local checks in a pre-created run")
    _common(run)
    run.add_argument("--mode", choices=("real", "simulation"), required=True)
    run.add_argument("--node", default=None)
    run.add_argument("--run-id", required=True)

    fabric = commands.add_parser("run-fabric", help="record one allocation-wide NCCL sweep")
    fabric.add_argument("--config", type=Path, required=True)
    fabric.add_argument("--run-dir", type=Path, required=True)
    fabric.add_argument("--mode", choices=("real", "simulation"), default="real",
                        help="must match the provenance established by initialize-run")

    init = commands.add_parser("initialize-run", help="exclusively reserve a run and write its manifest")
    _common(init)
    init.add_argument("--mode", choices=("real", "simulation"), required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--state", choices=("running", "submitting"), default="running")

    submission = commands.add_parser("submission-state", help="persist a Slurm submission state transition")
    submission.add_argument("--run-dir", type=Path, required=True)
    submission.add_argument("--state", choices=("submitting", "submitted", "failed"), required=True)
    submission.add_argument("--worker-node")
    submission.add_argument("--worker-job-id")
    submission.add_argument("--fabric-job-id")
    submission.add_argument("--admission-job-id")
    submission.add_argument("--error")

    final = commands.add_parser("finalize-run", help="verify exact evidence coverage and complete a run")
    final.add_argument("--config", type=Path, required=True)
    final.add_argument("--run-dir", type=Path, required=True)

    simulate = commands.add_parser("simulate", help="run deterministic, explicitly synthetic fleet fixtures")
    _common(simulate)
    simulate.add_argument("--run-id", default=None)
    simulate.add_argument("--baseline", type=Path, default=None, help="existing complete synthetic baseline")
    simulate.add_argument("--baseline-fixture-output", type=Path, default=None, help="explicit output root for a separate all-passing synthetic baseline fixture run")

    baseline = commands.add_parser("baseline", help="create a baseline from a complete all-passing run")
    baseline.add_argument("--run-dir", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--id", default=None)

    evaluate = commands.add_parser("evaluate", help="apply baseline admission gates once")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--baseline", type=Path, required=True)
    evaluate.add_argument("--apply-quarantine", action="store_true")

    submit = commands.add_parser("submit", help="transactionally launch qualification through Slurm")
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--output", type=Path, default=Path("runs"))
    submit.add_argument("--baseline", type=Path, required=True)
    submit.add_argument("--run-id", default=None)
    submit.add_argument("--apply-quarantine", action="store_true")
    submit.add_argument("--assets-dir", type=Path, default=None)
    return root


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, default=Path("runs"))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "initialize-run":
            _print(initialize_run(load_config(args.config), args.mode, args.run_id, args.output.resolve(), args.state))
            return 0
        if args.command == "submission-state":
            _print(update_submission(args.run_dir.resolve(), args.state, worker_node=args.worker_node, worker_job_id=args.worker_job_id, fabric_job_id=args.fabric_job_id, admission_job_id=args.admission_job_id, error=args.error))
            return 0
        if args.command == "run-node":
            config = load_config(args.config)
            node = args.node or os.environ.get("SLURMD_NODENAME") or os.uname().nodename
            record = run_node(config, args.mode, node, args.run_id, args.output.resolve())
            _print(record.to_dict())
            return 0 if record.overall_status == "pass" else 2
        if args.command == "run-fabric":
            _print(attach_fabric_result(load_config(args.config), args.run_dir.resolve(), args.mode))
            return 0
        if args.command == "finalize-run":
            _print(finalize_run(load_config(args.config), args.run_dir.resolve()))
            return 0
        if args.command == "simulate":
            return _simulate(args)
        if args.command == "baseline":
            _print(create_baseline(args.run_dir.resolve(), args.output.resolve(), args.id))
            return 0
        if args.command == "evaluate":
            report = aggregate(load_config(args.config), args.run_dir.resolve(), args.baseline.resolve(), args.apply_quarantine)
            _print(report.to_dict())
            return 0 if not report.rejected else 2
        if args.command == "submit":
            return _submit(args)
    except (ConfigError, FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"clustercheck: error: {exc}", file=sys.stderr)
        return 1
    return 1


def _simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_id = validate_identifier(args.run_id or new_run_id(), "run ID")
    print(SYNTHETIC_NOTICE, file=sys.stderr)
    output = args.output.resolve()
    initialize_run(config, "simulation", run_id, output)
    for node in config.cluster["nodes"]:
        run_node(config, "simulation", str(node), run_id, output)
    run_dir = output / run_id
    attach_fabric_result(config, run_dir, "simulation")
    finalize_run(config, run_dir)
    if args.baseline:
        baseline_path = args.baseline.resolve()
    elif args.baseline_fixture_output is not None:
        fixture_output = args.baseline_fixture_output.resolve()
        fixture_id = validate_identifier(f"{run_id}-baseline-fixture", "fixture run ID")
        initialize_run(config, "simulation", fixture_id, fixture_output, synthetic_fixture=True)
        for node in config.cluster["nodes"]:
            run_node(config, "simulation", str(node), fixture_id, fixture_output, synthetic_fixture=True)
        fixture_dir = fixture_output / fixture_id
        attach_fabric_result(config, fixture_dir, "simulation", synthetic_fixture=True)
        finalize_run(config, fixture_dir)
        baseline_path = run_dir / "synthetic-baseline.json"
        create_baseline(fixture_dir, baseline_path, "SYNTHETIC-DEMO-BASELINE")
    else:
        raise ValueError("synthetic baseline creation needs --baseline or explicit --baseline-fixture-output")
    report = aggregate(config, run_dir, baseline_path)
    _print(report.to_dict())
    return 0 if not report.rejected else 2


def _submit(args: argparse.Namespace) -> int:
    load_config(args.config)
    if not args.baseline.resolve().is_file():
        raise FileNotFoundError(f"baseline does not exist: {args.baseline.name}")
    run_id = validate_identifier(args.run_id or new_run_id(), "run ID")
    if args.assets_dir:
        assets = args.assets_dir
    else:
        packaged = Path(__file__).resolve().parent / "slurm"
        source = Path(__file__).resolve().parents[2] / "slurm"
        assets = packaged if packaged.is_dir() else source
    script = assets.resolve() / "submit_qualification.sh"
    if not script.is_file():
        raise FileNotFoundError("installed Slurm submission assets are unavailable")
    command = [str(script), str(args.config.resolve()), str(args.output.resolve()), str(args.baseline.resolve()), run_id]
    if args.apply_quarantine:
        command.append("--apply-quarantine")
    # The submission script re-enters this CLI, so guarantee the active
    # installation's console script is reachable regardless of shell activation.
    # Do not resolve: a venv's python is a symlink to the base interpreter, and
    # only the unresolved parent holds this installation's console scripts.
    bin_dirs = [str(Path(sys.argv[0]).parent), str(Path(sys.executable).parent)]
    prefix = os.pathsep.join(d for d in dict.fromkeys(bin_dirs) if d and Path(d).is_dir())
    path = os.environ.get("PATH", "")
    child_path = f"{prefix}{os.pathsep}{path}" if prefix and path else (prefix or path)
    result = CommandRunner(300).run(command, 300, extra_env={"PATH": child_path})
    if result.return_code != 0 or result.timed_out:
        raise RuntimeError(result.stderr.strip() or "Slurm submission failed")
    print(result.stdout, end="")
    return 0


def _print(value: dict[str, object]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
