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
        "--methods",
        type=str,
        required=False,
        default="gfn2-xtb, r2scan-3c_GRAD",
        help="Comma separated list of methods to use.",
    )
    return parser.parse_args()


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
                list(map(lambda x: float(x.replace("D", "E")), lines[i].split()))
                for i in number_lines[len(number_lines) // 2 :]
            ]
        )
        return gradient


def parse_orca_gradient(file: Path) -> np.ndarray:
    """
    Parse the gradient from the ORCA output file:
    ------------------
    CARTESIAN GRADIENT
    ------------------

       1   O   :    0.006350291   -0.008440957   -0.003357480
       2   C   :   -0.008161812    0.003870253   -0.001779037
       3   C   :    0.006660876    0.003069053    0.001391212
       4   H   :   -0.000383006   -0.000387866    0.003712008
       5   H   :   -0.000792639    0.000124760   -0.002592300
       6   H   :   -0.002657347    0.000481362   -0.000513059
       7   H   :   -0.000557464   -0.000925923    0.000833975
       8   H   :   -0.001010191   -0.000685303   -0.001139491
       9   H   :    0.000551291    0.002894622    0.003444172

    Args:
        file (Path): Path to the ORCA output file.
    """

    gradient: list[list[float]] = []
    with open(file, "r", encoding="utf8") as f:
        # read all lines
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "CARTESIAN GRADIENT" in line:
                # start reading the gradient
                for j in range(i + 3, len(lines)):
                    if lines[j].strip() == "":
                        break
                    gradient.append([float(x) for x in lines[j].split()[3:]])

    return np.array(gradient)


def main() -> int:
    """
    Main function to execute the script.
    """
    args = get_args()
    input_file = Path(args.keyword).resolve()
    with open(input_file, encoding="utf-8") as file:
        mol_names = file.readlines()
    gradients = pd.DataFrame()
    methods = [method.strip() for method in args.methods.split(",")]

    for mol_name in mol_names:
        mol_name = mol_name.strip()
        # dictionary to store the gradient norms
        gradient_norms: dict[str, float] = {}
        for method in methods:
            # get the path to the energy file
            if method == "gfn2-xtb":
                gradient_file = Path(f"{mol_name}/{method}/gradient").resolve()
            if method == "r2scan-3c_GRAD":
                gradient_file = Path(f"{mol_name}/{method}/orca.out").resolve()

            if not gradient_file.exists():  # pylint: disable=E0606
                print(f"File {gradient_file} does not exist.")
                gradient_norms[method] = np.nan
                continue

            # parse gradient
            if method == "gfn2-xtb":
                gradient = parse_gradient_file(gradient_file)
            elif method == "r2scan-3c_GRAD":
                gradient = parse_orca_gradient(gradient_file)
            # evaluate norm of gradient vector
            gradient_norm = np.linalg.norm(gradient)  # pylint: disable=E0606
            gradient_norms[method] = gradient_norm

        # create a dataframe from the gradient norms
        gradients = pd.concat(
            [
                gradients,
                pd.DataFrame(
                    {
                        "Molecule": [mol_name],
                        "gfn2-xtb": [gradient_norms["gfn2-xtb"]],
                        "r2scan-3c": [gradient_norms["r2scan-3c_GRAD"]],
                    }
                ),
            ],
            ignore_index=True,
        )
    gradients.to_csv("gradients.csv", index=False)
    if args.verbosity > 0:
        print(gradients)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
