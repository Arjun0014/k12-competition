from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from trace_ace.v04_forensic_audit import run_v04_forensic_audit


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the zero-training v0.4 packaged-runtime forensic audit."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_v04_forensic_audit(
        Path(args.project_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(report["gate_a"], indent=2, sort_keys=True))
    print(f"Phase A report: {report['run_id']}")


if __name__ == "__main__":
    main()
