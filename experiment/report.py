"""Rebuild the paper tables without model loading or new experiments."""

import argparse

from _impl.report import attack_comparison, defense_comparison, paper_tables
from cfg import require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="/tmp/posthoc-paper-tables")
    p.add_argument("--attack-directory", help="Collect one fixed attack campaign")
    p.add_argument("--defense-comparison", action="store_true", help="Current defenses and PaRaMS, using completed results")
    a = p.parse_args()
    require_root()
    if a.defense_comparison:
        defense_comparison(a.output)
    elif a.attack_directory:
        attack_comparison(a.attack_directory)
    else:
        paper_tables(None, a.output)


if __name__ == "__main__":
    main()
