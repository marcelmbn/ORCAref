"""
Python script to conduct a triple zeta basis set calculation using ORCA.
"""

from pathlib import Path
import argparse as ap
import subprocess as sp
import shutil as sh


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


def get_args() -> ap.Namespace:
    """
    Get the arguments from the command line.
    """
    parser = ap.ArgumentParser(
        description="Conduct the job of converting ORCA output to TM output."
    )
    # define an argument that can take a large string that is given to the qvSZP binary
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="Print the content of the ORCA output and error files.",
    )
    parser.add_argument(
        "--mpi",
        type=int,
        required=False,
        default=1,
        help="Number of MPI processes to use.",
    )
    parser.add_argument(
        "--orca_hirshfeld",
        action="store_true",
        required=False,
        help="Use ORCA internal Hirshfeld charges.",
    )
    # positional argument for the structure file
    parser.add_argument(
        "structure",
        type=str,
        help="Structure file that contains the coordinates of the molecule.",
    )
    return parser.parse_args()


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
        with (
            open(orca_output_file, "w", encoding="utf8") as out,
            open(orca_error_file, "w", encoding="utf8") as err,
        ):
            sp.run([orca_path, orca_input], stdout=out, stderr=err, check=True)
    except sp.CalledProcessError as e:
        raise ValueError(f"ORCA did not terminate normally: {e}") from e

    return orca_output_file, orca_error_file


def write_orca_input(
    mpi: int, chrg: int, uhf: int, heavy_ati: list[int] | None
) -> Path:
    """
    Write the input file for ORCA.
    """
    orca_input = Path("tz.inp").resolve()
    with open(orca_input, "w", encoding="utf8") as f:
        f.write("! wB97M-V def2-TZVPPD\n")
        f.write("! NoTRAH\n")
        f.write("! StrongSCF DefGrid3\n")
        f.write("! EnGrad\n")
        f.write("! PrintBasis\n")
        if heavy_ati:
            f.write("! AutoAux\n")
        f.write(f"%pal\n   nprocs {mpi}\nend\n")
        if heavy_ati:
            f.write("%basis\n")
            for heavy_atom in heavy_ati:
                f.write(f'   NewGTO  {PSE[heavy_atom]} "def-TZVP" end\n')
                f.write(f'   NewECP  {PSE[heavy_atom]} "def-ECP" end\n')
            f.write("end\n")
        f.write("%scf\n   maxiter 500\nend\n")
        f.write("%elprop\n\tOrigin 0.0,0.0,0.0\nend\n")
        f.write(
            "%output\n   Print[P_Hirshfeld] 1\n   "
            + "Print[ P_Mayer ] 0\n   Print[ P_Loewdin ] 0\nend\n"
        )
        f.write(f"*xyzfile {chrg} {uhf + 1} struc.xyz\n")
    return orca_input


def get_chrg_uhf(nel_orig: int) -> tuple[int, int]:
    """
    Get the charge and multiplicity from the .CHRG and .UHF files.

    Returns:
        tuple[int, int]: Charge and multiplicity.
    """

    # read the charge from the .CHRG file if it exists
    if Path(".CHRG").is_file():
        with open(".CHRG", "r", encoding="utf8") as f:
            chrg = int(f.readline().strip())
            print("Charge from '.CHRG' file:", chrg)
    else:
        chrg = 0

    # get the number of electrons
    nel = nel_orig - chrg

    # read the multiplicity from the .UHF file if it exists
    if Path(".UHF").is_file():
        with open(".UHF", "r", encoding="utf8") as f:
            uhf = int(f.readline().strip())
            print(f"UHF from '.UHF' file: {uhf}")
    else:
        if nel % 2 == 0:
            uhf = 0
        else:
            print("Odd number of electrons detected. Setting UHF to 1.")
            uhf = 1

    return chrg, uhf


def parse_hirshfeld(file: Path) -> list[float]:
    """
    Parse the Hirshfeld charges from the ORCA output file.

    Relevant ORCA output lines:
    ...

    ------------------
    HIRSHFELD ANALYSIS
    ------------------

    Total integrated alpha density =     56.999694904
    Total integrated beta density  =     55.999695584

      ATOM     CHARGE      SPIN
       0 O   -2.528834    0.057977
       1 C   -0.253541    0.005754
       2 C   -0.032957    0.010477
       3 H    0.090152    0.000918
       4 H    0.158754    0.002076
       5 H    0.135321    0.001610
       6 H    0.103791    0.010221
       7 H    0.052870    0.004086
       8 Th   4.275053    0.906880

      TOTAL   2.000610    0.999999

    -------
    TIMINGS
    -------
    ...

    Args:
        file (Path): Path to the ORCA output file.

    Returns:
        list[int]: List of Hirshfeld charges.
    """
    charges: list[float] = []
    with open(file, "r", encoding="utf8") as f:
        # read all lines
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "HIRSHFELD ANALYSIS" in line:
                # start reading the Hirshfeld charges
                for j in range(i + 7, len(lines)):
                    if lines[j].strip() == "":
                        continue
                    if "TOTAL" in lines[j]:
                        break
                    charges.append(float(lines[j].split()[2]))
    return charges


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
                        atom_number, _ = atom_num_sym_angmom
                    elif len(atom_num_sym_angmom) == 3:
                        atom_number, _, _ = atom_num_sym_angmom
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
    with open(orca_output_file, "r", encoding="utf8") as orca_out:
        orca_content = orca_out.read()

    # grep the number of atoms from this line:
    # ```Number of atoms                             ...      9```
    natoms = int(orca_content.split("Number of atoms")[1].split()[1])
    print(f"Found {natoms} atoms.")

    # set up a list of Population objects
    populations = [Population(i + 1) for i in range(natoms)]

    # mulliken_charges contains the Mulliken atomic charges
    # with atom number as key and charge as value
    populations = parse_orca_mulliken_charges(populations, orca_content, openshell)
    populations = parse_mulliken_reduced_orbital_charges(
        populations, orca_content, natoms, openshell=openshell
    )
    for pop in populations:
        print(pop)

    # write the TM output file
    write_tm_mulliken(populations, "scf.out", openshell)


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


def write_multiwfn_charges(
    charges: list[float], ati: list[int], xyz: list[list[float]]
) -> None:
    """
    Write the Hirshfeld charges to a multiwfn.chg file.
    C     0.000000    0.000000    0.316085  -0.0083631036
    N     0.000000    0.000000    1.476008  -0.1995613116
    At    0.000000    0.000000   -1.792093   0.2079244152

    Args:
        charges (list[int]): List of Hirshfeld charges.
    """
    if len(charges) != len(ati):
        raise ValueError("Length of charges and atomic numbers do not match.")
    with open("multiwfn.chg", "w", encoding="utf8") as f:
        for i, charge in enumerate(charges):
            f.write(
                f"{PSE[ati[i]]:>2} {xyz[i][0]:>14.8f} {xyz[i][1]:>14.8f} {xyz[i][2]:>14.8f} {charge:>14.8f}\n"
            )


def get_multiwfn_hirshfeld() -> tuple[Path, Path, Path]:
    """
    Execute MultiWFN for Hirshfeld Charges.
    """

    # check full path of the ORCA binary (which is in the PATH)
    orca_2mkl_path = sp.run(
        ["which", "orca_2mkl"], capture_output=True, text=True, check=True
    ).stdout.strip()

    print(f"ORCA_2MKL path: {orca_2mkl_path}")

    # convert ORCA gbw file to molden format
    multiwfn_charge_file = Path("multiwfn.chg").resolve()
    multiwfn_output_file = Path("multiwfn.out").resolve()
    multiwfn_error_file = Path("multiwfn.err").resolve()
    print("Running ORCA_2MKL")
    try:
        with (
            open(multiwfn_output_file, "w", encoding="utf8") as out,
            open(multiwfn_error_file, "w", encoding="utf8") as err,
        ):
            sp.run(
                [orca_2mkl_path, "tz", "-molden"], stdout=out, stderr=err, check=True
            )
    except sp.CalledProcessError as e:
        raise ValueError(f"ORCA_2MKL did not terminate normally: {e}") from e

    # check full path of the Multiwfn binary (which is in the PATH)
    multiwfn_path = sp.run(
        ["which", "Multiwfn"], capture_output=True, text=True, check=True
    ).stdout.strip()

    print(f"Multiwfn path: {multiwfn_path}")

    # Create 'inp' file with specified inputs
    with open("inp", "w", encoding="utf8") as multiwfn_input_file:
        multiwfn_input_file.write("7\n1\n1\ny\n0\n9\n1\nn\n0\nq\n")

    # run Multiwfn for Hirshfeld charges
    try:
        with (
            open("inp", "r", encoding="utf8") as inp,
            open(multiwfn_output_file, "w", encoding="utf8") as out,
            open(multiwfn_error_file, "w", encoding="utf8") as err,
        ):
            sp.run(
                [multiwfn_path, "tz.molden.input", "-nt", "8"],
                stdin=inp,
                stdout=out,
                stderr=err,
                check=True,
            )
    except sp.CalledProcessError as e:
        raise ValueError(f"Multiwfn did not terminate normally: {e}") from e

    # Move the generated Hirshfeld charges file to multiwfn.chg
    try:
        sh.move("tz.chg", "multiwfn.chg")
    except Exception as exc:
        raise ValueError(
            "Multiwfn did not generate the Hirshfeld charges file."
        ) from exc

    return multiwfn_charge_file, multiwfn_output_file, multiwfn_error_file


def write_control_file(energy: float, dipole: list[float]) -> None:
    """
    Write the control file for the TM output:
    $subenergy  Etot         E1                  Ej Ex                 Ec                 En
    -39.46273594172    -67.82138813233     23.07494182535 0.000000000000     0.000000000000     9.469652916458
    $charge from ridft
              1.000 (not to be modified here)
    $dipole from ridft
      x     0.00010248934969    y     0.00002281403674    z 0.00000000000132    a.u.

    Args:
        energy (float): Total energy.
        dipole (list[float]): Dipole moment.
    """
    with open("control", "w", encoding="utf8") as f:
        f.write(
            "$subenergy  Etot         E1                  Ej Ex                 Ec                 En\n"
        )
        f.write(
            f"{energy:>20.11f}    {0.0:>20.11f}     {0.0:>20.11f} {0.0:>20.11f}     {0.0:>20.11f}     {0.0:>20.11f}\n"
        )
        f.write("$dipole from ridft\n")
        f.write(
            f"  x     {dipole[0]:>20.11f}    y     {dipole[1]:>20.11f}    z {dipole[2]:>20.11f}    a.u.\n"
        )
        f.write("$end\n")


def parse_gradient(file: Path) -> list[list[float]]:
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
    return gradient


def write_tm_gradient(
    gradient: list[list[float]],
    xyz: list[list[float]],
    symbols: list[str],
    energy: float,
) -> None:
    """
    Write the gradient to a TM gradient file:
    $grad
      cycle =      1    SCF energy =   -11.39433809450   |dE/dxyz| =  0.000206
       -2.28377308081965      0.47888412430551      0.00986135744064      O
       -0.09355866925134     -1.03299913689694      0.03099565440194      C
        2.33481276207902      0.51910811867711      0.01027690268576      C
       -0.10639904570851     -2.29577204625936      1.68367916569539      H
       -0.21650182663134     -2.19216842346259     -1.67256392109945      H
        3.96453468532290     -0.72683368189313     -0.07332130313004      H
        2.36267325147101      1.76430214597466     -1.62411387508703      H
        2.47151759889187      1.66716238545536      1.71352887904645      H
       -2.21780747886488      1.61895040795194      1.42909166965606      H
       7.5962740085231E-05  -4.0857934540112E-05  -4.8744761475391E-05
      -6.0983845417856E-05   7.9830766457179E-05  -3.4925661034367E-05
      -1.4463699542727E-05   1.2449070639496E-05  -2.5758177939448E-05
      -9.0543927133971E-07  -3.0757631178324E-05   2.9978764766058E-05
       9.9487072236105E-06  -3.3117742741925E-06  -1.7761861926360E-06
       1.8388736755787E-05  -5.6520712811802E-05   2.2432871280090E-05
       1.8193409276013E-05   6.5970552417126E-06  -5.2966267867581E-05
      -2.8770522205566E-05   3.7645234986818E-05   2.2991740734472E-05
      -1.7370086903159E-05  -5.0740745207732E-06   8.8767677728882E-05
    $end

    Args:
        gradient (list[list[float]]): Gradient.
        xyz (list[list[float]]): XYZ coordinates.
    """
    # calculate  |dE/dxyz|
    sumofsqaures = sum((sum(x**2 for x in grad) for grad in gradient))
    norm_grad = sumofsqaures**0.5
    with open("gradient", "w", encoding="utf8") as f:
        f.write("$grad\n")
        f.write(
            f"  cycle =      1    SCF energy =   {energy}   |dE/dxyz| =  {norm_grad}\n"
        )
        for i, coord in enumerate(xyz):
            f.write(
                f"{coord[0]:>20.11f} {coord[1]:>20.11f} {coord[2]:>20.11f} {symbols[i]}\n"
            )
        for i, grad in enumerate(gradient):
            f.write(f"{grad[0]:>20.8e} {grad[1]:>20.8e} {grad[2]:>20.8e}\n")
        f.write("$end\n")


def main() -> int:
    """
    Main function to conduct the job.
    """
    # call the argument parser
    args = get_args()

    # convert coord file to xyz file, if necessary
    if args.structure == "coord":
        sp.run(
            [
                "mctc-convert",
                "-i",
                "coord",
                "-o",
                "xyz",
                "coord",
                "struc.xyz",
                "--normalize",
            ],
            check=True,
        )
    # elif args.structure is of type xyz (ends with .xyz), then copy it to struc.xyz
    elif args.structure.endswith(".xyz") and not args.structure == "struc.xyz":
        sh.copy(args.structure, "struc.xyz")
    elif args.structure == "struc.xyz":
        pass
    else:
        raise ValueError("Unknown structure file type.")

    # check if the structure contains elements with Z > 86
    heavy_atoms: list[int] | None = None
    ati: list[int] = []
    symbols: list[str] = []
    xyz: list[list[float]] = []
    with open("struc.xyz", "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            if line.strip() == "":
                continue
            symbol = line.split()[0].lower()
            symbols.append(symbol)
            xyz.append([float(x) * AA2AU for x in line.split()[1:]])
            ati.append(PSE_NUMBERS[symbol])
            if PSE_NUMBERS[symbol] > 86:
                if not heavy_atoms:
                    heavy_atoms = [PSE_NUMBERS[symbol]]
                else:
                    heavy_atoms.append(PSE_NUMBERS[symbol])

    # get number of electrons
    nel_raw = sum(ati)

    # get the charge and multiplicity from the .CHRG and .UHF files
    chrg, uhf = get_chrg_uhf(nel_raw)

    orca_input_file = write_orca_input(args.mpi, chrg, uhf, heavy_atoms)

    # execute ORCA with the generated input file
    orca_output_file, _ = execute_orca(orca_input_file)
    with open(orca_output_file, "r", encoding="utf8") as orca_out:
        if "ORCA TERMINATED NORMALLY" not in orca_out.read():
            raise ValueError("ORCA did not terminate normally.")

    if args.orca_hirshfeld:
        # parse the Hirshfeld charges from the ORCA output file
        charges = parse_hirshfeld(orca_output_file)
        print(f"Hirshfeld charges: {charges}")
        # write charges to multiwfn.chg
        write_multiwfn_charges(charges, ati, xyz)
    else:
        # Do Hirshfeld population analysis with Multiwfn
        multiwfn_charge_file, _, _ = get_multiwfn_hirshfeld()
        charges = []
        with open(multiwfn_charge_file, "r", encoding="utf8") as multiwfn_out:
            for line in multiwfn_out:
                charges.append(float(line.split()[4]))
        print(f"Hirshfeld charges: {charges}")

    # parse energy from ORCA output file
    energy: float | None = None
    dipole: list[float] | None = None
    with open(orca_output_file, "r", encoding="utf8") as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[4])
            # parse dipole moment from ORCA output file (from following lines)
            # Total Dipole Moment    :      0.517831297       0.043824204       0.411962740
            if "Total Dipole Moment" in line:
                dipole = [float(x) for x in line.split()[4:]]
                break
    if not energy:
        raise ValueError("Energy not found in ORCA output file.")
    if not dipole:
        raise ValueError("Dipole moment not found in ORCA output file.")
    print(f"Total energy: {energy}")
    print(f"Dipole moment: {dipole}")
    write_control_file(energy, dipole)

    # parse gradient from ORCA output file
    gradient = parse_gradient(orca_output_file)
    print(f"Gradient: {gradient}")
    write_tm_gradient(gradient, xyz, symbols, energy)
    convert_orca_output(orca_output_file, openshell=uhf > 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
