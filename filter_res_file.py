#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


def parse_valid_elements(file: Path) -> set[str]:
    return set(file.read_text().split())


def extract_species_from_path(path: str) -> list[str]:
    """
    Extract compound and its species from a string like '01_10{P,R1,R2}/$func/'
    → ['01_10P', '01_10R1', '01_10R2']
    """
    match = re.match(r"(?P<base>[\d_]+)\{(?P<species>[^}]+)\}", path)
    if not match:
        return []

    base = match.group("base")
    species = [s.strip() for s in match.group("species").split(",")]
    return [f"{base}{s}" for s in species]


def filter_res_file(res_lines: Sequence[str], valid_species: set[str]) -> list[str]:
    result = []
    for line in res_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            result.append(line)
            continue

        tokens = stripped.split()
        if len(tokens) < 2:
            result.append(line)
            continue

        path = tokens[1]
        # Only filter if the path contains a {...} pattern
        if "{" not in path or "}" not in path:
            result.append(line)
            continue

        species = extract_species_from_path(path)
        if all(s in valid_species for s in species):
            result.append(line)

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Filter a .res file to include only reactions with all valid species."
    )
    parser.add_argument(
        "--res", type=Path, required=True, help="Path to input .res file"
    )
    parser.add_argument(
        "--valid",
        type=Path,
        required=True,
        help="Path to file containing list of valid species",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output filtered .res file"
    )
    args = parser.parse_args()

    valid_species = parse_valid_elements(args.valid)
    res_lines = args.res.read_text().splitlines()
    filtered_lines = filter_res_file(res_lines, valid_species)

    args.output.write_text("\n".join(filtered_lines) + "\n")


if __name__ == "__main__":
    main()
