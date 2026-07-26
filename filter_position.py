#!/usr/bin/env python3
"""
filter_position.py
==================
Single driver for "Where Is the Filter? Joint Observational
Underdetermination from the Fermi Silence, the Doomsday Tail, and the
Anthropic Shadow" (v2).

Yavuz Selim Kilinc - Independent Researcher, Manisa, Turkiye
MIT License

Reproduces every table, figure and numerical claim in the manuscript.

    python filter_position.py            # checks, then all experiments
    python filter_position.py --checks   # verification suite only
    python filter_position.py --figures  # tables and figures only

Requirements: numpy >= 1.20, matplotlib >= 3.4.  No other dependencies.
Runtime: about 1 minute on a laptop (dominated by the 20,000-member
coupling-class sweep in Table 4).
"""

from __future__ import annotations

import argparse
import sys

import filter_checks
import filter_experiments


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checks", action="store_true",
                    help="run the verification suite only")
    ap.add_argument("--figures", action="store_true",
                    help="run the tables and figures only")
    args = ap.parse_args(argv)

    if args.figures:
        filter_experiments.run_all()
        return 0

    ok = filter_checks.run_all()
    if args.checks:
        return 0 if ok else 1
    if not ok:
        print("\nVerification failed; not producing figures.", file=sys.stderr)
        return 1
    print()
    filter_experiments.run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
