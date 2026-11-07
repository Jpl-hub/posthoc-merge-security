"""Reproduce NBP tables and figures from the recorded evaluation counts."""

import argparse
from pathlib import Path

from _impl.report import tables
from cfg import require_root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/nbp_tables")
    parser.add_argument("--figure", action="store_true", help="Also render result figures")
    parser.add_argument("--defense-comparison", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require_root()
    tables(args.output)
    if args.figure:
        from _impl.figures import figures
        figures(Path(args.output) / "fig", Path(args.output) / "sources.json")

if __name__ == "__main__":
    main()
