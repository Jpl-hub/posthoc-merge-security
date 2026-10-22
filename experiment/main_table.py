"""Evaluate frozen publications with matched TA/TIES/DARE controls."""

import argparse

from _impl.evaluate import run, verify
from cfg import require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/evaluate.json")
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--verify", metavar="RESULT_DIRECTORY")
    a = p.parse_args()
    require_root()
    verify(a.verify) if a.verify else run(a.protocol, a.case, a.pilot)


if __name__ == "__main__":
    main()
