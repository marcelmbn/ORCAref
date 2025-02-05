"""
Python script to conduct the following job.
1. From a given structure file, generate the ORCA input file by execution of the 'qvSZP' binary.
2. Execute ORCA with the generated input file.
3. Parse relevant information from the ORCA output file.
4. Generate an emulated TM output file with the parsed information.
"""

from pathlib import Path
import shutil as sh
import argparse as ap
import subprocess as sp

AU2AA = 0.529177210544
AA2AU = 1.0 / AU2AA
PSE = {
    0: "X",
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}
PSE_NUMBERS: dict[str, int] = {k.lower(): v for v, k in PSE.items()}
PSE_SYMBOLS: dict[int, str] = {v: k.lower() for v, k in PSE.items()}


class Population:
    """
    Class to store the population information.

    Contains
    - atom type: atom number (int) and atom symbol (str)
    - Mulliken charge (float)
    - s, p, d, f, g populations (float)
    """

    def __init__(self, atom: int):
        self.atom = atom
        self.ati: int | None = None
        # Mulliken charges and populations
        self.mull_q: float | None = None
        self.mull_p_s: float | None = None
        self.mull_p_p: float | None = None
        self.mull_p_d: float | None = None
        self.mull_p_f: float | None = None
        self.mull_p_g: float | None = None
        # Mulliken spin populations
        self.mull_spinpop: float | None = None
        self.mull_spinpop_s: float | None = None
        self.mull_spinpop_p: float | None = None
        self.mull_spinpop_d: float | None = None
        self.mull_spinpop_f: float | None = None
        self.mull_spinpop_g: float | None = None

    def __str__(self):
        returnstr = f"Atom: {self.atom}\n"
        if self.ati is not None:
            returnstr += f"Atom type: {PSE_SYMBOLS[self.ati]}\n"
        if self.mull_q is not None:
            returnstr += f"Mulliken charge: {self.mull_q:8.5f}\n"
        if self.mull_p_s is not None:
            returnstr += f"Mulliken population 's': {self.mull_p_s:8.5f}\n"
        if self.mull_p_p is not None:
            returnstr += f"Mulliken population 'p': {self.mull_p_p:8.5f}\n"
        if self.mull_p_d is not None:
            returnstr += f"Mulliken population 'd': {self.mull_p_d:8.5f}\n"
        if self.mull_p_f is not None:
            returnstr += f"Mulliken population 'f': {self.mull_p_f:8.5f}\n"
        if self.mull_p_g is not None:
            returnstr += f"Mulliken population 'g': {self.mull_p_g:8.5f}\n"
        # if self.mull_spinpop is not None: print spin populations in a similar fashion
        if self.mull_spinpop is not None:
            # first print a separation line
            returnstr += "-" * 80 + "\n"
            returnstr += f"Mulliken spin population: {self.mull_spinpop:8.5f}\n"
        if self.mull_spinpop_s is not None:
            returnstr += f"Mulliken spin population 's': {self.mull_spinpop_s:8.5f}\n"
        if self.mull_spinpop_p is not None:
            returnstr += f"Mulliken spin population 'p': {self.mull_spinpop_p:8.5f}\n"
        if self.mull_spinpop_d is not None:
            returnstr += f"Mulliken spin population 'd': {self.mull_spinpop_d:8.5f}\n"
        if self.mull_spinpop_f is not None:
            returnstr += f"Mulliken spin population 'f': {self.mull_spinpop_f:8.5f}\n"
        if self.mull_spinpop_g is not None:
            returnstr += f"Mulliken spin population 'g': {self.mull_spinpop_g:8.5f}\n"
        return returnstr


def write_tm_mulliken(
    pops: list[Population], filename: str | Path, openshell: bool = False
) -> None:
    """
    Write the Mulliken population information in the TM output format.
    ```
        atomic populations from total density:

     atom      charge    n(s)      n(p)      n(d)      n(f)      n(g)
         1ce     1.78562   4.00840  12.04756  10.91160   1.24681
         2f     -0.59626   1.97543   5.62083
         3f     -0.59624   1.97543   5.62081
         4f     -0.59311   1.97544   5.61767
    ```
    if open shell:

    ```
     Unpaired electrons from D(alpha)-D(beta)

     atom      total     n(s)      n(p)      n(d)      n(f)      n(g)
         1ce     1.02700   0.00016  -0.00016   0.02946   0.99754
         2f     -0.00800  -0.00015  -0.00785
         3f     -0.00800  -0.00015  -0.00786
         4f     -0.01099  -0.00015  -0.01084
    ```
    """
    tm_mull = "    atomic populations from total density:\n\n"
    tm_mull += "atom      charge    n(s)      n(p)      n(d)      n(f)      n(g)\n"
    for pop in pops:
        if pop.ati is None:
            raise ValueError("No atom type found.")
        tm_mull += f"    {pop.atom:>3}{PSE_SYMBOLS[pop.ati]:<2}     "
        tm_mull += f"{pop.mull_q:8.5f}   "
        tm_mull += f"{pop.mull_p_s:8.5f}  " if pop.mull_p_s is not None else " " * 12
        tm_mull += f"{pop.mull_p_p:8.5f}  " if pop.mull_p_p is not None else " " * 12
        tm_mull += f"{pop.mull_p_d:8.5f}  " if pop.mull_p_d is not None else " " * 12
        tm_mull += f"{pop.mull_p_f:8.5f}  " if pop.mull_p_f is not None else " " * 12
        tm_mull += f"{pop.mull_p_g:8.5f}  " if pop.mull_p_g is not None else " " * 12
        tm_mull += "\n"

    # if spin populations are present, print them as well
    if openshell:
        tm_mull += "\n    Unpaired electrons from D(alpha)-D(beta)\n\n"
        tm_mull += "atom      total     n(s)      n(p)      n(d)      n(f)      n(g)\n"
        for pop in pops:
            if pop.ati is None:
                raise ValueError("No atom type found.")
            tm_mull += f"    {pop.atom:>3}{PSE_SYMBOLS[pop.ati]:<2}     "
            tm_mull += f"{pop.mull_spinpop:8.5f}   "
            tm_mull += (
                f"{pop.mull_spinpop_s:8.5f}  "
                if pop.mull_spinpop_s is not None
                else " " * 12
            )
            tm_mull += (
                f"{pop.mull_spinpop_p:8.5f}  "
                if pop.mull_spinpop_p is not None
                else " " * 12
            )
            tm_mull += (
                f"{pop.mull_spinpop_d:8.5f}  "
                if pop.mull_spinpop_d is not None
                else " " * 12
            )
            tm_mull += (
                f"{pop.mull_spinpop_f:8.5f}  "
                if pop.mull_spinpop_f is not None
                else " " * 12
            )
            tm_mull += (
                f"{pop.mull_spinpop_g:8.5f}  "
                if pop.mull_spinpop_g is not None
                else " " * 12
            )
            tm_mull += "\n"

    with open(filename, "w", encoding="utf8") as tm_out:
        tm_out.write(tm_mull)


def parse_orca_mulliken_charges(
    populations: list[Population], orca_output: str, openshell: bool = False
) -> list[Population]:
    """
    Parse the ORCA output file and extract the Mulliken population analysis.
    """
    # First parse the Mulliken atomic charges
    # split the content by lines
    lines = orca_output.split("\n")
    # iterate over the lines
    for i, line in enumerate(lines):
        # check if the line contains the Mulliken atomic charges
        if "MULLIKEN ATOMIC CHARGES" in line:
            # iterate over the lines after the line with the header
            for j in range(i + 2, len(lines)):
                # check if the line is empty
                if lines[j].strip() == "":
                    break
                if "Sum of atomic charges" in lines[j]:
                    break
                # split the line by ":"
                atom, charge = lines[j].split(":")
                atom_number: str | int
                atom_number, atom_symbol = atom.split()
                atom_number = int(atom_number)

                if openshell:
                    charge, spin_pop = charge.split()
                # append the atom and charge to the list
                populations[atom_number].mull_q = float(charge)
                populations[atom_number].ati = PSE_NUMBERS[atom_symbol.lower()]
                if openshell:
                    populations[atom_number].mull_spinpop = float(spin_pop)
    return populations


def parse_mulliken_reduced_orbital_charges(
    populations: list[Population], orca_output: str, nat: int, openshell: bool = False
) -> list[Population]:
    """
    Parse the ORCA output file and extract the Mulliken reduced orbital charges.

    The part looks as follows:
    ```
    --------------------------------
    MULLIKEN REDUCED ORBITAL CHARGES
    --------------------------------
      0 O s       :     1.733068  s :     1.733068
          pz      :     1.625117  p :     4.757872
          px      :     1.582040
          py      :     1.550715
          dz2     :     0.008794  d :     0.052880
          dxz     :     0.007340
          dyz     :     0.015704
          dx2y2   :     0.010845
          dxy     :     0.010195

      1 C s       :     0.962153  s :     0.962153
          pz      :     1.074845  p :     2.777703
          px      :     0.782766
          py      :     0.920092
          dz2     :     0.020886  d :     0.232749
          dxz     :     0.036019
          dyz     :     0.070005
          dx2y2   :     0.072115
          dxy     :     0.033725

      2 C s       :     1.020250  s :     1.020250
          pz      :     1.065344  p :     3.120061
          ...
    ```
    """
    # First parse the Mulliken atomic charges
    # split the content by lines
    lines = orca_output.split("\n")
    # iterate over the lines
    for i, line in enumerate(lines):
        # check if the line contains the Mulliken reduced orbital charges
        if "MULLIKEN REDUCED ORBITAL CHARGES" in line:
            # iterate over the lines after the line with the header
            atom_number: str | int | None
            last_atom = False
            for j in range(i + 2, len(lines)):
                # if SPIN in line: break
                if "SPIN" in lines[j]:
                    break
                # check if the line is empty
                if "s :" in lines[j]:
                    # split the line by ":"
                    atom_information, _, mull_s = lines[j].split(":")
                    atom_num_sym_angmom = atom_information.strip().split()
                    # check if it contains two or three elements
                    if len(atom_num_sym_angmom) == 2:
                        atom_number, atom_symbol = atom_num_sym_angmom
                    elif len(atom_num_sym_angmom) == 3:
                        atom_number, atom_symbol, _ = atom_num_sym_angmom
                    atom_number = int(atom_number)  # type: ignore
                    if atom_number >= nat - 1:
                        last_atom = True
                    # append the atom and charge to the list
                    populations[atom_number].mull_p_s = float(mull_s)
                if "p :" in lines[j]:
                    if atom_number is None or not isinstance(atom_number, int):
                        raise ValueError("No atom number found.")
                    # split the line by ":"
                    _, _, mull_p = lines[j].split(":")
                    populations[atom_number].mull_p_p = float(mull_p.strip())
                if "d :" in lines[j]:
                    if atom_number is None or not isinstance(atom_number, int):
                        raise ValueError("No atom number found.")
                    # split the line by ":"
                    _, _, mull_d = lines[j].split(":")
                    populations[atom_number].mull_p_d = float(mull_d.strip())
                if "f :" in lines[j]:
                    if atom_number is None or not isinstance(atom_number, int):
                        raise ValueError("No atom number found.")
                    # split the line by ":"
                    _, _, mull_f = lines[j].split(":")
                    populations[atom_number].mull_p_f = float(mull_f.strip())
                if "g :" in lines[j]:
                    if atom_number is None or not isinstance(atom_number, int):
                        raise ValueError("No atom number found.")
                    # split the line by ":"
                    _, _, mull_g = lines[j].split(":")
                    populations[atom_number].mull_p_g = float(mull_g.strip())
                if lines[j].strip() == "":
                    atom_number = None
                    if last_atom:
                        break
    if openshell:
        # iterate over the lines
        for i, line in enumerate(lines):
            # check if the line contains the Mulliken reduced orbital charges
            if "MULLIKEN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS" in line:
                # iterate over the lines after the line with the header
                start_spin_pops = False
                last_atom = False
                for j in range(i + 2, len(lines)):
                    if "SPIN" in lines[j]:
                        start_spin_pops = True
                    if not start_spin_pops:
                        continue
                    # check if the line is empty
                    if "s :" in lines[j]:
                        # split the line by ":"
                        atom_information, _, mull_s = lines[j].split(":")
                        atom_num_sym_angmom = atom_information.strip().split()
                        # check if it contains two or three elements
                        if len(atom_num_sym_angmom) == 2:
                            atom_number, atom_symbol = atom_num_sym_angmom
                        elif len(atom_num_sym_angmom) == 3:
                            atom_number, atom_symbol, _ = atom_num_sym_angmom
                        atom_number = int(atom_number)  # type: ignore
                        if atom_number >= nat - 1:
                            last_atom = True
                        # append the atom and charge to the list
                        populations[atom_number].mull_spinpop_s = float(mull_s)
                    if "p :" in lines[j]:
                        if atom_number is None or not isinstance(atom_number, int):
                            raise ValueError("No atom number found.")
                        # split the line by ":"
                        _, _, mull_p = lines[j].split(":")
                        populations[atom_number].mull_spinpop_p = float(mull_p.strip())
                    if "d :" in lines[j]:
                        if atom_number is None or not isinstance(atom_number, int):
                            raise ValueError("No atom number found.")
                        # split the line by ":"
                        _, _, mull_d = lines[j].split(":")
                        populations[atom_number].mull_spinpop_d = float(mull_d.strip())
                    if "f :" in lines[j]:
                        if atom_number is None or not isinstance(atom_number, int):
                            raise ValueError("No atom number found.")
                        # split the line by ":"
                        _, _, mull_f = lines[j].split(":")
                        populations[atom_number].mull_spinpop_f = float(mull_f.strip())
                    if "g :" in lines[j]:
                        if atom_number is None or not isinstance(atom_number, int):
                            raise ValueError("No atom number found.")
                        # split the line by ":"
                        _, _, mull_g = lines[j].split(":")
                        populations[atom_number].mull_spinpop_g = float(mull_g.strip())
                    if lines[j].strip() == "":
                        atom_number = None
                        if last_atom:
                            break

    return populations


def generate_orca_input(qvSZP_args: str, addtional_arguments: list[str]) -> None:
    """
    Generate the ORCA input file from the given structure file.
    """
    binary = "qvSZP"
    arguments = [arg for arg in qvSZP_args.split() if arg != ""]
    if addtional_arguments:
        arguments.extend(addtional_arguments)
    print(f"Running command-line call: {binary} {' '.join(arguments)}")
    # run command via subprocess, write the output to qvSZP.out and the error to qvSZP.err
    with open("qvSZP.out", "w") as out, open("qvSZP.err", "w") as err:
        sp.run([binary] + arguments, stdout=out, stderr=err, check=True)


def execute_orca(orca_input: Path) -> tuple[Path, Path]:
    """
    Execute ORCA with the generated input file.
    """

    # check full path of the ORCA binary (which is in the PATH)
    orca_path = sp.run(
        ["which", "orca"], capture_output=True, text=True, check=True
    ).stdout.strip()

    print(f"ORCA path: {orca_path}")

    # check if the ORCA input file exists
    if not orca_input.is_file():
        raise FileNotFoundError(f"ORCA input file '{orca_input}' not found.")

    # execute ORCA with the input file
    orca_output_file = Path("orca.out").resolve()
    orca_error_file = Path("orca.err").resolve()
    print(f"Running ORCA with input file '{orca_input}'")
    try:
        with open(orca_output_file, "w") as out, open(orca_error_file, "w") as err:
            sp.run([orca_path, orca_input], stdout=out, stderr=err, check=True)
    except sp.CalledProcessError as e:
        raise ValueError(f"ORCA did not terminate normally: {e}") from e

    return orca_output_file, orca_error_file


def get_args() -> ap.Namespace:
    """
    Get the arguments from the command line.
    """
    parser = ap.ArgumentParser(
        description="Conduct the job of converting ORCA output to TM output."
    )
    # define an argument that can take a large string that is given to the qvSZP binary
    parser.add_argument(
        "--qvSZP",
        type=str,
        required=True,
        help="Arguments to be passed to the qvSZP binary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="Print the content of the ORCA output and error files.",
    )
    parser.add_argument(
        "--multifwfnfile",
        "-mwfnf",
        type=str,
        required=False,
        help="Path to the Multiwfn output file.",
    )
    return parser.parse_args()


def convert_orca_output(orca_output_file: Path, openshell: bool) -> None:
    """
    Parse the ORCA output file and extract relevant information.

    The relevant part of the ORCA output looks as follows:

    ```
                    ********************************
                    * MULLIKEN POPULATION ANALYSIS *
                    ********************************

    -----------------------
    MULLIKEN ATOMIC CHARGES
    -----------------------
       0 O :   -0.543820
       1 C :    0.027395
       2 C :   -0.250227
       3 H :    0.086523
       4 H :    0.107410
       5 H :    0.092568
       6 H :    0.088546
       7 H :    0.084070
       8 H :    0.307536
    Sum of atomic charges:   -0.0000000

    --------------------------------
    MULLIKEN REDUCED ORBITAL CHARGES
    --------------------------------
      0 O s       :     1.733068  s :     1.733068
          pz      :     1.625117  p :     4.757872
          px      :     1.582040
          py      :     1.550715
          dz2     :     0.008794  d :     0.052880
          dxz     :     0.007340
          dyz     :     0.015704
          dx2y2   :     0.010845
          dxy     :     0.010195

      1 C s       :     0.962153  s :     0.962153
          pz      :     1.074845  p :     2.777703
          px      :     0.782766
          py      :     0.920092
          dz2     :     0.020886  d :     0.232749
          dxz     :     0.036019
          dyz     :     0.070005
          dx2y2   :     0.072115
          dxy     :     0.033725

      2 C s       :     1.020250  s :     1.020250
          pz      :     1.065344  p :     3.120061
          px      :     1.013388
          py      :     1.041329
          dz2     :     0.017538  d :     0.109916
          dxz     :     0.011347
          dyz     :     0.042558
          dx2y2   :     0.009990
          dxy     :     0.028483

      3 H s       :     0.877221  s :     0.877221
          pz      :     0.022061  p :     0.036257
          px      :     0.001834
          py      :     0.012361

      4 H s       :     0.857156  s :     0.857156
          pz      :     0.022479  p :     0.035434
          px      :     0.002433
          py      :     0.010521

      5 H s       :     0.855228  s :     0.855228
          pz      :     0.016160  p :     0.052204
          px      :     0.019462
          py      :     0.016581

      6 H s       :     0.859122  s :     0.859122
          pz      :     0.021534  p :     0.052331
          px      :     0.011950
          py      :     0.018847

      7 H s       :     0.863864  s :     0.863864
          pz      :     0.021134  p :     0.052066
          px      :     0.012640
          py      :     0.018292

      8 H s       :     0.595094  s :     0.595094
          pz      :     0.040669  p :     0.097371
          px      :     0.025609
          py      :     0.031092

    ```

    The output I want to have is the following (here for a different example):
    ```
        atomic populations from total density:

     atom      charge    n(s)      n(p)      n(d)      n(f)      n(g)
         1ce     1.78562   4.00840  12.04756  10.91160   1.24681
         2f     -0.59626   1.97543   5.62083
         3f     -0.59624   1.97543   5.62081
         4f     -0.59311   1.97544   5.61767


     Unpaired electrons from D(alpha)-D(beta)

     atom      total     n(s)      n(p)      n(d)      n(f)      n(g)
         1ce     1.02700   0.00016  -0.00016   0.02946   0.99754
         2f     -0.00800  -0.00015  -0.00785
         3f     -0.00800  -0.00015  -0.00786
         4f     -0.01099  -0.00015  -0.01084
    ```
    """
    # First open the file and read the whole content
    with open(orca_output_file, "r") as orca_out:
        orca_content = orca_out.read()

    # if not ORCA TERMINATED NORMALLY in orca_content raise an error
    if "ORCA TERMINATED NORMALLY" not in orca_content:
        raise ValueError("ORCA did not terminate normally.")

    # grep the number of atoms from this line:
    # ```Number of atoms                             ...      9```
    natoms = int(orca_content.split("Number of atoms")[1].split()[1])
    print(f"Found {natoms} atoms.")

    # set up a list of Population objects
    populations = [Population(i + 1) for i in range(natoms)]

    # mulliken_charges contains the Mulliken atomic charges with atom number as key and charge as value
    populations = parse_orca_mulliken_charges(populations, orca_content, openshell)
    populations = parse_mulliken_reduced_orbital_charges(
        populations, orca_content, natoms, openshell=openshell
    )
    for pop in populations:
        print(pop)

    # write the TM output file
    write_tm_mulliken(populations, "tm.out", openshell)


def parse_multiwfn_output(mwfnf: str | Path) -> list[float]:
    """
    Args:
        mwfnf: str | Path

    Return:
        atomic_charges:  list[float]

    ------
    multiwfn.chg looks as follows:
    C     0.000000    0.000000    0.316085  -0.0083631036
    N     0.000000    0.000000    1.476008  -0.1995613116
    At    0.000000    0.000000   -1.792093   0.2079244152
    """
    if isinstance(mwfnf, str):
        mwfnf = Path(mwfnf)
    mwfnf.resolve()
    if not mwfnf.is_file():
        raise FileNotFoundError(f"Multiwfn output file '{mwfnf}' not found.")
    with open(mwfnf, "r", encoding="utf8") as mwfn_out:
        mwfn_content = mwfn_out.read()
    # split the content by lines
    lines = mwfn_content.split("\n")
    # iterate over the lines
    atomic_charges = []
    for line in lines:
        # check if the line is empty
        if line.strip() == "":
            continue
        # split the line by whitespace
        charge = line.split()[4]
        atomic_charges.append(float(charge))
    return atomic_charges


def main():
    """
    Main function to conduct the job.
    """
    # call the argument parser
    args = get_args()

    add_args: list[str] = []
    # if mwfnf is given, parse the Multiwfn output file
    if args.multifwfnfile is not None:
        atomic_charges = parse_multiwfn_output(args.multifwfnfile)
        charge_args = ["--cm", "extonlyq"]
        add_args.extend(charge_args)
        print(f"Atomic charges from 'multiwfn.chg' or similar:\n{atomic_charges}")
        # write charges to a file "ext.charges" (just one float per line)
        with open("ext.charges", "w", encoding="utf8") as ext_out:
            for charge in atomic_charges:
                ext_out.write(f"{charge}\n")

    # check if "--struc" is contained in args.qvSZP
    # if yes, take the value after "--struc" as the structure file
    if "--struc" in args.qvSZP:
        struc_file = Path(args.qvSZP.split("--struc")[1].split()[0].strip()).resolve()
    else:
        struc_file = Path("coord").resolve()
    if not struc_file.is_file():
        print(f"Structure file '{struc_file}' not found. Trying 'coord' file.")
        raise FileNotFoundError(f"Structure file '{struc_file}' not found.")

    tmp_file = "struc.bak.xyz"
    # convert coord file to xyz file, if necessary
    if struc_file.name == "coord":
        sp.run(
            [
                "mctc-convert",
                "-i",
                "coord",
                "-o",
                "xyz",
                "coord",
                tmp_file,
                "--normalize",
            ],
            check=True,
        )
    # elif args.structure is of type xyz (ends with .xyz), then copy it to struc.bak.xyz
    elif struc_file.suffix == ".xyz":
        sh.copy(struc_file, tmp_file)
    else:
        raise ValueError("Unknown structure file type.")

    ati: list[int] = []
    symbols: list[str] = []
    xyz: list[list[float]] = []
    with open(tmp_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            if line.strip() == "":
                continue
            symbol = line.split()[0].lower()
            symbols.append(symbol)
            xyz.append([float(x) * AA2AU for x in line.split()[1:]])
            ati.append(PSE_NUMBERS[symbol])

    # check if .CHRG file is present. If yes, read the total charge from it.
    chargeset = False
    if Path(".CHRG").is_file():
        with open(".CHRG", "r") as chrg_file:
            charge = int(chrg_file.read().strip())
        print(f"Total charge from '.CHRG' file: {charge}")
        chargeset = True
    # same with .UHF file
    uhfset = False
    if Path(".UHF").is_file():
        with open(".UHF", "r") as uhf_file:
            uhf = int(uhf_file.read().strip())
        print(f"Total spin from '.UHF' file: {uhf}")
        uhfset = True
    else:
        # sum up the atomic numbers (sum of ati array)
        nel = sum(ati)
        if chargeset:
            # subtract the total charge from the sum of atomic numbers
            nel -= charge
        # check if odd
        if nel % 2 == 1:
            uhf = 1
            add_args.extend(["--uhf", "1"])
            print("Odd number of electrons detected. Setting UHF to 1.")

    # generate the ORCA input file
    generate_orca_input(args.qvSZP, add_args)

    # check if "--outname" is contained in args.qvSZP
    # if yes, take the value after "--outname" as the ORCA input file name
    # if not, set it to "wb97xd4-qvszp.inp"
    orca_input = Path("wb97xd4-qvszp.inp")
    if "--outname" in args.qvSZP:
        orca_input = Path(args.qvSZP.split("--outname")[1].split()[0] + ".inp")

    # parse the multiplicities from the ORCA input file
    # line in ORCA input: `* xyz    1  2`
    with open(orca_input, "r") as orca_in:
        for line in orca_in:
            if "* xyz" in line:
                multiplicity = int(line.split()[3])
                break
    if multiplicity is None:
        raise ValueError("No multiplicity found.")
    if multiplicity > 1:
        openshell = True
    else:
        openshell = False

    # execute ORCA with the generated input file
    orca_output_file, orca_error_file = execute_orca(orca_input)

    # raise appropriate error if orca_output does not contain "ORCA TERMINATED NORMALLY"
    if not orca_output_file.is_file():
        raise FileNotFoundError(f"ORCA output file '{orca_output_file}' not found.")

    # open both files and print the content
    if args.verbose:
        with (
            open(orca_output_file, "r") as orca_out,
            open(orca_error_file, "r") as orca_err,
        ):
            # if not ORCA TERMINATED NORMALLY in orca_content raise an error
            if "ORCA TERMINATED NORMALLY" not in orca_out:
                raise ValueError("ORCA did not terminate normally.")
            print("ORCA output file:")
            print(orca_out.read())
            print("ORCA error file:")
            print(orca_err.read())

    convert_orca_output(orca_output_file, openshell)


if __name__ == "__main__":
    main()
