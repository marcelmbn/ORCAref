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
    Extract compound and its species from a string like:
    - '01_10{P,R1,R2}/$func/' → ['01_10P', '01_10R1', '01_10R2']
    - '{DMML_REACT,DMML_INT1}/$f' → ['DMML_REACT', 'DMML_INT1']
    """
    match = re.match(r"(?:(?P<base>[\w\d_]+))?\{(?P<species>[^}]+)\}", path)
    if not match:
        raise ValueError(
            f"Invalid format for species extraction: {path}. Expected format: 'base{{species}}'."
        )

    base = match.group("base") or ""
    species = [s.strip() for s in match.group("species").split(",")]

    if base:
        return [f"{base}{s}" for s in species]
    return species


def filter_res_file(res_lines: Sequence[str], valid_species: set[str]) -> list[str]:
    result = []
    for line in res_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            result.append(line)
            continue

        tokens = stripped.split()
        if not (
            "x" in tokens
            and (tokens[0].startswith("$tmer") or tokens[0].startswith("tmer"))
        ):
            # Here, we assume that the line is not a valid tmer entry
            # but just prepending lines in bash so we just append the line to the result
            result.append(line)
            continue

        # Here, we assume that the line is a valid tmer entry
        # Collect all species-related tokens between $tmer and the 'x' token
        species_tokens: list[str] = []
        for token in tokens[1:]:
            if token == "x":
                break
            species_tokens.append(token)

        species: list[str] = []
        for token in species_tokens:
            relevant_paths = token.split("/")
            token_to_search: str = ""
            for relevant_path in relevant_paths:
                relevant_path = relevant_path.strip()
                if relevant_path.startswith("$") or not relevant_path:
                    # Skip empty tokens or those starting with $
                    continue
                token_to_search = relevant_path
            if not token_to_search:
                continue
            if "{" in token_to_search and "}" in token_to_search:
                species.extend(extract_species_from_path(token_to_search))
            else:
                species.append(token_to_search)
        # make the species a set of unique species
        unique_species = set(species)
        if all(s in valid_species for s in unique_species):
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
