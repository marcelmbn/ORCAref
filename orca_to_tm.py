"""
Python script to conduct the following job.
1. From a given structure file, generate the ORCA input file by execution of the 'qvSZP' binary.
2. Execute ORCA with the generated input file.
3. Parse relevant information from the ORCA output file.
4. Generate an emulated TM output file with the parsed information.
"""

from pathlib import Path
import argparse as ap
import subprocess as sp


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
        self.mull_q: float | None = None
        self.mull_p_s: float | None = None
        self.mull_p_p: float | None = None
        self.mull_p_d: float | None = None
        self.mull_p_f: float | None = None
        self.mull_p_g: float | None = None

    def __str__(self):
        returnstr = f"Atom: {self.atom}\n"
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
        return returnstr


def parse_orca_mulliken_charges(
    populations: list[Population], orca_output: str
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
                # append the atom and charge to the list
                populations[atom_number].mull_q = float(charge)
    return populations


def parse_mulliken_reduced_orbital_charges(
    populations: list[Population], orca_output: str, nat: int
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
            for j in range(i + 2, len(lines)):
                # check if the line is empty
                last_atom = False
                if "s :" in lines[j]:
                    # split the line by ":"
                    atom_information, _, mull_s = lines[j].split(":")
                    atom_number, atom_symbol, _ = atom_information.split()
                    atom_number = int(atom_number)
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

    return populations


def generate_orca_input(qvSZP_args: str):
    """
    Generate the ORCA input file from the given structure file.
    """
    binary = "qvSZP"
    arguments = [arg for arg in qvSZP_args.split() if arg != ""]
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
    with open(orca_output_file, "w") as out, open(orca_error_file, "w") as err:
        sp.run([orca_path, orca_input], stdout=out, stderr=err, check=True)

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
    return parser.parse_args()


def convert_orca_output(orca_output_file: Path) -> str:
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

    # grep the number of atoms from this line:
    # ```Number of atoms                             ...      9```
    natoms = int(orca_content.split("Number of atoms")[1].split()[1])
    print(f"Found {natoms} atoms.")

    # set up a list of Population objects
    populations = [Population(i + 1) for i in range(natoms)]

    # mulliken_charges contains the Mulliken atomic charges with atom number as key and charge as value
    populations = parse_orca_mulliken_charges(populations, orca_content)
    populations = parse_mulliken_reduced_orbital_charges(
        populations, orca_content, natoms
    )
    for pop in populations:
        print(pop)

    return ""


def main():
    """
    Main function to conduct the job.
    """
    # call the argument parser
    args = get_args()

    # generate the ORCA input file
    generate_orca_input(args.qvSZP)

    # check if "--outname" is contained in args.qvSZP
    # if yes, take the value after "--outname" as the ORCA input file name
    # if not, set it to "wb97xd4-qvszp.inp"
    orca_input = Path("wb97xd4-qvszp.inp")
    if "--outname" in args.qvSZP:
        orca_input = Path(args.qvSZP.split("--outname")[1].split()[0] + ".inp")

    # execute ORCA with the generated input file
    orca_output_file, orca_error_file = execute_orca(orca_input)

    # open both files and print the content
    if args.verbose:
        with open(orca_output_file, "r") as orca_out, open(
            orca_error_file, "r"
        ) as orca_err:
            print("ORCA output file:")
            print(orca_out.read())
            print("ORCA error file:")
            print(orca_err.read())

    tm_output = convert_orca_output(orca_output_file)


if __name__ == "__main__":
    main()
