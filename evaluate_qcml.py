"""
Python script that reads energy files from TURBOMOLE
for a given method and returns a Pandas data frame
"""

from pathlib import Path
import argparse

import pandas as pd


def get_args() -> argparse.Namespace:
    """
    Get the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect stereoisomers for a given list of molecules."
    )
    parser.add_argument(
        "--verbosity", "-v", type=int, default=1, help="Verbosity level."
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=False,
        default="molecules.list",
        help="Keyword for the file that contains the list of molecules.",
    )
    return parser.parse_args()


def parse_energy_file(file_path: Path) -> float:
    """
    Parse the energy file and return the energy value.

    `energy`:

    $energy
    1      -760.02681299010885     -107.28077147427206  99.9 99.9 99.9
    $end
    """

    with open(file_path, encoding="utf8") as file:
        inside_energy_block = False
        scf_lines = []
        for line in file:
            line = line.strip()
            if line.startswith("$energy"):
                inside_energy_block = True
                continue
            if line.startswith("$end"):
                break
            if inside_energy_block and line and line[0].isdigit():
                scf_lines.append(line)
        if not scf_lines:
            raise ValueError("No SCF energy entries found between $energy and $end.")
        # Get the last SCF line and extract the total energy
        last_scf = scf_lines[-1].split()
        energy = float(last_scf[1])
    return energy


def statistics(energies: pd.DataFrame) -> None:
    """
    Print the statistics of the energies.
    """
    print(energies.describe())
    print(energies.corr())


def main() -> int:
    """
    Main function to execute the script.
    """
    args = get_args()
    input_file = Path(args.keyword).resolve()
    with open(input_file, encoding="utf-8") as file:
        mol_names = file.readlines()
    energies = pd.DataFrame()

    # create a dataframe with three columns: molecule, gxtb energy, wb97m-v energy
    for mol_name in mol_names:
        mol_name = mol_name.strip()
        # get the path to the energy file
        prefix = "/tmp1/grimme/BENCH/qcml_100k"
        gxtb_energy_file = Path(f"{prefix}/{mol_name}/energy").resolve()
        wb97m_energy_file = Path(f"{prefix}/{mol_name}/TZ/energy").resolve()

        # check if the files exist
        if not gxtb_energy_file.exists():
            print(f"File {gxtb_energy_file} does not exist.")
            continue
        if not wb97m_energy_file.exists():
            print(f"File {wb97m_energy_file} does not exist.")
            continue

        # parse the energy files
        gxtb_energy = parse_energy_file(gxtb_energy_file)
        wb97m_energy = parse_energy_file(wb97m_energy_file)

        # add the energies to the dataframe
        energies = pd.concat(
            [
                energies,
                pd.DataFrame(
                    {
                        "Molecule": [mol_name],
                        "GXTBEnergy": [gxtb_energy],
                        "WB97MEnergy": [wb97m_energy],
                    }
                ),
            ],
            ignore_index=True,
        )
    # print the statistics
    if args.verbosity > 0:
        print(energies)
        statistics(energies)

    # save the dataframe to a csv file
    energies.to_csv("energies.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
