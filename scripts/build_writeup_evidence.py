import argparse
import json

from trace_ace.writeup_evidence import build_writeup_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic Trace the Ace write-up evidence index."
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    manifest = build_writeup_evidence(args.project_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
