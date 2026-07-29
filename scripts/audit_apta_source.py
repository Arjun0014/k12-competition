from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_ace.apta_source_audit import audit_apta_source, write_report


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the target-free APTA source and linkage audit."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--source-root",
        default="Datasets/APTA",
        help="Authorized APTA download root; must remain under Datasets/APTA.",
    )
    parser.add_argument("--report", help="Optional deterministic JSON output path.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    authorized_root = (project_root / "Datasets" / "APTA").resolve()
    source_root = Path(args.source_root)
    if not source_root.is_absolute():
        source_root = (project_root / source_root).resolve()
    if not _inside(source_root, authorized_root):
        raise PermissionError(
            f"APTA audit source must remain under {authorized_root}; got {source_root}"
        )

    report = audit_apta_source(source_root)
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = (project_root / report_path).resolve()
        report_hash = write_report(report, report_path)
        print(
            json.dumps(
                {
                    "report": report_path.as_posix(),
                    "sha256": report_hash,
                    "discovery_ready": report["discovery_ready"],
                    "gates": report["gates"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
