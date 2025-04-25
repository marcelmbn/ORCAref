"""
Python script that reads energy files from TURBOMOLE
for a given method and returns a Pandas data frame
"""

from pathlib import Path
import argparse

import pandas as pd
import numpy as np


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
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="If set, the script will also parse the gradient files.",
        default=False,
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


def parse_gradient_file(file_path: Path) -> np.ndarray:
    """
    Parse a gradient file and return the gradient vector.

    the first half of the table (with symbols) corresponds to
    atomic coordinates
    the second half of the table (without symbols) corresponds to
    the gradient matrix

    `gradient`:
    $grad
      cycle =      1    SCF energy =  -760.02834405724   |dE/dxyz| =  0.224080
       -6.73465660645383      1.79728257060385      1.87734309810402      C
       -5.06166419571071      0.13036085644471     -0.91428879786277      Si
       -1.61406563590556     -0.92477970090561     -1.14963343310282      N
        1.30422381292015      0.40135094498180     -3.14590463265674      Sb
        1.91111984222510      3.76917282115182     -1.46406282623872      N
        4.06605255360352      3.50671411334025      0.23685086622626      C
        4.57973876308739      1.81157185647953      1.85103798054786      C
       -6.54093587183869      0.72132036212644      3.66103686894196      H
       -5.53521803649860      3.47392519092409      1.86945708202343      H
       -8.76743958851412      2.01686902796660      1.50523832113874      H
       -5.91376978620331      1.95603092513465     -2.69001827028434      H
       -6.31886454165582     -2.28589106197495     -1.52281997057599      H
       -1.57888138665053     -2.76105853260832     -0.38650937441116      H
       -1.43681077401058      1.27241482765711     -4.42352686887278      H
        2.78683155555397      5.19589035512176     -2.18278229999704      H
        5.34032768494116      5.17756890945212      0.47889506145465      H
        5.91302593505857      2.15378855027481      3.26804467523979      H
        3.24981671782730      0.35156506896309      2.61292469962901      H
      -0.2896928742757D-01   0.3717884698441D-01   0.2965718234312D-02
      -0.4697400768994D-01   0.3302368801693D-01  -0.1575557491075D-01
       0.7362481956079D-01   0.2943655972558D-01  -0.4630895719705D-01
       0.1830437502122D-01   0.7608560251053D-02   0.4201285256992D-02
      -0.5078256585117D-01   0.1747147135423D-01  -0.5127233174941D-01
       0.1571772449248D-01  -0.6621648694060D-01   0.2023014270283D-01
       0.3552306083445D-01   0.7901608470699D-01  -0.7152566068574D-01
      -0.3507859749637D-03  -0.1149084588027D-01   0.6208406042181D-02
       0.1485764097045D-01  -0.9893347229687D-02  -0.1255632105313D-01
      -0.1092523367182D-01  -0.7922851955300D-02   0.3251230820448D-02
      -0.5378819408482D-02  -0.1564693236560D-01   0.2893740268064D-01
       0.4742328133034D-02   0.3610531431756D-02  -0.2026809321857D-02
      -0.7581532152775D-02  -0.3916951078130D-01   0.1129467524758D-01
      -0.3753144134559D-02  -0.2086403940211D-01   0.3042113018334D-01
       0.1754454441046D-01  -0.3627640724062D-01   0.5342252859464D-01
       0.1656956602953D-01   0.1277524550763D-01   0.2139548808676D-01
      -0.2844441707263D-01  -0.5142159663289D-02  -0.1060421030464D-01
      -0.1372426606849D-01  -0.7498406519820D-02   0.2772185737285D-01
    $end
    """
    with open(file_path, encoding="utf8") as file:
        # read the file line by line
        lines = file.readlines()
        inside_number_block = False
        number_lines: list[int] = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            if line.startswith("$grad") or line.startswith("cycle"):
                inside_number_block = True
                continue
            if line.startswith("$end"):
                break
            if not inside_number_block:
                continue
            number_lines.append(i)
        # if len (number_lines) % 2 != 0:
        if len(number_lines) % 2 != 0:
            raise ValueError("The gradient file does not have the correct format.")
        # extract a numpy array from the last half of the table
        gradient = np.array(
            [
                list(map(float, lines[i].split()))
                for i in number_lines[len(number_lines) // 2 :]
            ]
        )
        return gradient


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
    if args.gradient:
        gradients = pd.DataFrame()

    for mol_name in mol_names:
        mol_name = mol_name.strip()
        # get the path to the energy file
        prefix = "/tmp1/grimme/BENCH/qcml_100k"
        gxtb_energy_file = Path(f"{prefix}/{mol_name}/energy").resolve()
        wb97m_energy_file = Path(f"{prefix}/{mol_name}/TZ/energy").resolve()
        if args.gradient:
            gxtb_gradient_file = Path(f"{prefix}/{mol_name}/gradient").resolve()
            wb97m_gradient_file = Path(f"{prefix}/{mol_name}/TZ/gradient").resolve()

        # check if the files exist
        if not gxtb_energy_file.exists():
            print(f"File {gxtb_energy_file} does not exist.")
            continue
        if not wb97m_energy_file.exists():
            print(f"File {wb97m_energy_file} does not exist.")
            continue
        if args.gradient and not gxtb_gradient_file.exists(): # pylint: disable=E0606
            print(f"File {gxtb_gradient_file} does not exist.")
            continue
        if args.gradient and not wb97m_gradient_file.exists(): # pylint: disable=E0606
            print(f"File {wb97m_gradient_file} does not exist.")
            continue

        # parse the energy files
        gxtb_energy = parse_energy_file(gxtb_energy_file)
        wb97m_energy = parse_energy_file(wb97m_energy_file)
        if args.gradient:
            gxtb_gradient = parse_gradient_file(gxtb_gradient_file)
            wb97m_gradient = parse_gradient_file(wb97m_gradient_file)
            print("gxtb gradient", gxtb_gradient)
            print("wb97m gradient", wb97m_gradient)
        exit()

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
