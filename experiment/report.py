"""Rebuild the paper tables without model loading or new experiments."""

import argparse

from _impl.report import paper_tables
from cfg import require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/paper_tables")
    a = p.parse_args()
    require_root()
    paper_tables(None, a.output)


if __name__ == "__main__":
    main()
