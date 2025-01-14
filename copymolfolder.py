"""
This script is used to copy a folder of molecules and perform ORCA calculations on them.
"""

from pathlib import Path
import argparse
import shutil as sh
import subprocess as sp
import hashlib
import csv

import copy
import numpy as np

BOHR2AA = (
    0.529177210544  # taken from https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
)
PSE: dict[int, str] = {
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


def get_lanthanides() -> list[int]:
    """
    Get the atomic numbers of lanthanides.
    """
    lanthanides = list(range(56, 71))
    return lanthanides


def get_actinides() -> list[int]:
    """
    Get the atomic numbers of actinides.
    """
    actinides = list(range(88, 103))
    return actinides


def check_if_neighbours(source_elem: str, target_elem: str) -> bool:
    """
    Check if the source and target elements are neighbours in the periodic table.
    """
    source_idx = PSE_NUMBERS[source_elem.lower()]
    target_idx = PSE_NUMBERS[target_elem.lower()]
    if abs(source_idx - target_idx) == 1:
        return True
    return False


def calculate_f_electrons(atlist: np.ndarray) -> int:
    """
    Calculate the number of unpaired electrons in a molecule.
    """
    f_electrons = 0
    for ati, occurrence in enumerate(atlist):
        if ati in get_lanthanides():
            f_electrons += (ati - 55) * occurrence
        elif ati in get_actinides():
            f_electrons += (ati - 87) * occurrence
    return f_electrons


def parse_orca_hirshfeld(orca_out: str) -> list[float]:
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
    lines = orca_out.split("\n")
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


def parse_orca_gradient(orca_out: str) -> list[list[float]]:
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
    lines = orca_out.split("\n")
    for i, line in enumerate(lines):
        if "CARTESIAN GRADIENT" in line:
            # start reading the gradient
            for j in range(i + 3, len(lines)):
                if lines[j].strip() == "":
                    break
                gradient.append([float(x) for x in lines[j].split()[3:]])
    return gradient


def parse_orca_energy(orca_out: str) -> float:
    """
    Parse the ORCA output to get the final energy.
    """
    for line in orca_out.split("\n"):
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split()[4])
            return energy
    raise ValueError("Energy not found in ORCA output.")


def parse_orca_dipole(orca_out: str) -> list[float]:
    """
    Parse the ORCA output to get the dipole moment.
    """
    dipole: list[float] | None = None
    for line in orca_out.split("\n"):
        # Total Dipole Moment    :      0.517831297       0.043824204       0.411962740
        if "Total Dipole Moment" in line:
            dipole = [float(x) for x in line.split()[4:]]
            break
    if dipole is None:
        raise ValueError("Dipole moment not found in ORCA output.")
    return dipole


class Molecule:
    """
    A class representing a molecule.
    """

    def __init__(self, name: str = ""):
        """
        Initialize a molecule with a name and an optional number of atoms.

        :param name: The name of the molecule.
        :param num_atoms: The initial number of atoms in the molecule (default is 0).
        """
        self._name = name

        self._num_atoms: int | None = None
        self._charge: int | None = None
        self._uhf: int | None = None
        self._atlist: np.ndarray = np.array([], dtype=int)
        self._xyz: np.ndarray = np.array([], dtype=float)
        self._ati: np.ndarray = np.array([], dtype=int)
        self._energy: float | None = None

        self.rng = np.random.default_rng()

    def __str__(self) -> str:
        """
        Return a user-friendly string representation of the molecule.
        """
        returnstr: str = ""
        first_line = True
        if self._name:
            if not first_line:
                returnstr += "\n"
            returnstr += f"Molecule: {self.name}"
            first_line = False
        if self._num_atoms is not None:
            if not first_line:
                returnstr += "\n"
            returnstr += f"# atoms: {self.num_atoms}"
            first_line = False
        if self._charge is not None:
            if not first_line:
                returnstr += "\n"
            returnstr += f"total charge: {self.charge}"
            first_line = False
        if self._uhf is not None:
            if not first_line:
                returnstr += "\n"
            returnstr += f"# unpaired electrons: {self.uhf}"
            first_line = False
        if self._atlist.size:
            if not first_line:
                returnstr += "\n"
            returnstr += f"atomic numbers: {self.atlist}\n"
            returnstr += f"sum formula: {self.sum_formula()}"
            first_line = False
        if self._xyz.size:
            if not first_line:
                returnstr += "\n"
            returnstr += f"atomic coordinates:\n{self.xyz}"
            first_line = False
        if self._ati.size:
            if not first_line:
                returnstr += "\n"
            returnstr += f"atomic number per index: {self._ati}"
            first_line = False
        return returnstr

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the molecule.
        """
        # print in this order: name, num_atoms, charge, uhf, xyz
        returnstr = (
            f"Molecule(name={self._name}, "
            + f"num_atoms={self._num_atoms}, "
            + f"charge={self._charge}, "
            + f"uhf={self._uhf}, "
            + f"ati={self._ati}, "
            + f"atlist={self._atlist}, "
            + f"xyz={self._xyz}), "
            + f"sum_formula: {self.sum_formula()}"
        )
        return returnstr

    @staticmethod
    def read_mol_from_file(file: str | Path) -> "Molecule":
        """
        Read the XYZ coordinates and the charge of the molecule from a file.
        Thereby, generate a completely new molecule object from scratch.

        Can be called like this:
            from molecule import Molecule
            # Call the static method using the class name
            xyz_file = "example_molecule.xyz"
            molecule_instance = Molecule.read_mol_from_file(xyz_file)
            # Now you can use the molecule_instance as needed
            print(molecule_instance.name)

        The layout of the file is as follows:
        ```
        num_atoms
        'Generated by mindlessgen-v{__version__}'
        <symbol 1> <x1> <y1> <z1>
        <symbol 2> <x2> <y2> <z2>
        ...
        ```

        :param file: The XYZ file to read from.
        :return: A new instance of Molecule with the read data.
        """
        molecule = Molecule()
        if isinstance(file, str):
            file_path = Path(file).resolve()
        elif isinstance(file, Path):
            file_path = file.resolve()
        else:
            raise TypeError("String or Path expected.")
        molecule.read_xyz_from_file(file_path)
        if file_path.with_suffix(".CHRG").exists():
            molecule.read_charge_from_file(file_path.with_suffix(".CHRG"))
        else:
            molecule.charge = 0
        if file_path.with_suffix(".UHF").exists():
            molecule.read_uhf_from_file(file_path.with_suffix(".UHF"))
        else:
            molecule.uhf = 0
        molecule.name = file_path.stem
        return molecule

    @staticmethod
    def read_mol_from_coord(file: str | Path) -> "Molecule":
        """
        Read the XYZ coordinates and the charge of the molecule from a 'coord' file.
        Thereby, generate a completely new molecule object from scratch.

        Can be called like this:
            from molecule import Molecule
            # Call the static method using the class name
            coord_file = "coord"
            molecule_instance = Molecule.read_mol_from_coord(coord_file)
            # Now you can use the molecule_instance as needed
            print(molecule_instance.name)

        :param file: The 'coord' file to read from.
        :return: A new instance of Molecule with the read data.
        """
        molecule = Molecule()
        if isinstance(file, str):
            file_path = Path(file).resolve()
        elif isinstance(file, Path):
            file_path = file.resolve()
        else:
            raise TypeError("String or Path expected.")
        molecule.read_xyz_from_coord(file_path)
        uhf_path = file_path.parent / ".UHF"
        if uhf_path.exists():
            molecule.read_uhf_from_file(uhf_path)
        else:
            molecule.uhf = 0
        chrg_path = file_path.parent / ".CHRG"
        if chrg_path.exists():
            molecule.read_charge_from_file(chrg_path)
        else:
            molecule.charge = 0
        return molecule

    @property
    def name(self) -> str:
        """
        Get the name of the molecule.

        :return: The name of the molecule.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the name of the molecule.

        :param value: The name to set.
        :raise TypeError: If the value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("String expected.")

        self._name = value

    @property
    def num_atoms(self) -> int:
        """
        Get the number of atoms in the molecule.

        :return: The number of atoms in the molecule.
        """
        if self._num_atoms is not None:
            return self._num_atoms
        else:
            if self._atlist.size:
                self.num_atoms = np.sum(self._atlist)
            elif self._xyz.size:
                self.num_atoms = len(self._xyz)
            elif self._ati.size:
                self.num_atoms = len(self._ati)
            if self._num_atoms:
                return self._num_atoms
            else:
                raise ValueError("Number of atoms not present and could not be set.")

    @num_atoms.setter
    # can be either int or numpy.int64
    def num_atoms(self, value: int | np.int64):
        """
        Set the number of atoms in the molecule.

        :param value: The number of atoms to set.
        :raise TypeError: If the value is not an integer.
        :raise ValueError: If the number of atoms is negative.
        """
        if not isinstance(value, int) and not isinstance(value, np.int64):
            raise TypeError("Integer expected.")
        if value < 0:
            raise ValueError("Number of atoms cannot be negative.")

        self._num_atoms = int(value)

    @property
    def charge(self) -> int:
        """
        Get the charge of the molecule.

        :return: The charge of the molecule.
        """
        if self._charge is not None:
            return self._charge
        else:
            raise ValueError("Charge not set.")

    @charge.setter
    def charge(self, value: int | float):
        """
        Set the charge of the molecule.

        :param value: The charge to set.
        :raise TypeError: If the value is not an integer.
        """
        try:
            value = int(value)
        except ValueError as e:
            raise TypeError("Integer expected.") from e

        self._charge = value

    @property
    def uhf(self) -> int:
        """
        Get the UHF of the molecule.

        :return: The UHF of the molecule.
        """
        if self._uhf is not None:
            return self._uhf
        else:
            raise ValueError("UHF not set.")

    @uhf.setter
    def uhf(self, value: int | float):
        """
        Set the UHF of the molecule.

        :param value: The UHF to set.
        :raise TypeError: If the value is not an integer.
        """
        try:
            value = int(value)
        except ValueError as e:
            raise TypeError("Integer expected.") from e

        if value < 0:
            raise ValueError("Number of unpaired electrons cannot be negative.")

        self._uhf = value

    @property
    def xyz(self) -> np.ndarray:
        """
        Get the XYZ coordinates of the molecule.

        :return: The XYZ coordinates of the molecule.
        """
        return self._xyz

    @xyz.setter
    def xyz(self, value: np.ndarray):
        """
        Set the XYZ coordinates of the molecule.

        :param value: The XYZ coordinates to set.
        :raise TypeError: If the value is not a numpy array.
        :raise ValueError: If the shape of the array is not (num_atoms, 3).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Numpy array expected.")
        if value.shape[1] != 3:
            raise ValueError("Shape of array must be (num_atoms, 3).")

        self._xyz = value

    @property
    def ati(self) -> np.ndarray:
        """
        Get the atomic number per index of the molecule.

        :return: The atomic number per index of the molecule.
        """
        return self._ati

    @ati.setter
    def ati(self, value: np.ndarray):
        """
        Set the atomic number per atom in the molecule.

        :param value: The atomic number per index to set.
        :raise TypeError: If the value is not a numpy array.
        :raise ValueError: If the shape of the array is not (num_atoms,).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Numpy array expected.")
        # check if array has the right shape
        if value.ndim != 1:
            raise ValueError("Array must have one dimension.")
        # if num_atoms is set, check if the array has the right length
        if self._num_atoms:
            if value.shape[0] != self._num_atoms:
                raise ValueError("Shape of array must be (num_atoms,).")

        self._ati = value

    @property
    def atlist(self) -> np.ndarray:
        """
        Get the initial array with the len 103 (number of elements in the periodic table)
        with the number of atoms of each element.

        :return: The array with the number of atoms of each element.
        """
        return self._atlist

    @atlist.setter
    def atlist(self, value: np.ndarray):
        """
        Set the array with the number of atoms of each element.

        :param value: The atomic numbers to set.
        :raise TypeError: If the value is not a numpy array.
        :raise ValueError: If the shape of the array is not (num_atoms,).
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("Numpy array expected.")
        # check if array has the right shape
        if value.ndim != 1:
            raise ValueError("Array must have one dimension.")

        self._atlist = value

    @property
    def energy(self) -> float | None:
        """
        Get the energy of the molecule.

        :return: The energy of the molecule.
        """
        return self._energy

    @energy.setter
    def energy(self, value: float):
        """
        Set the energy of the molecule.

        :param value: The energy to set.
        :raise TypeError: If the value is not a float.
        """
        if not isinstance(value, float):
            raise TypeError("Float expected.")

        self._energy = value

    def get_xyz_str(self) -> str:
        """
        Obtain a string with the full XYZ file information of the molecule.
        """
        xyz_str = f"{self.num_atoms}\n"
        try:
            commentline = f"Total charge: {self.charge} ; "
        except ValueError:
            commentline = ""
        try:
            commentline = commentline + f"Unpaired electrons: {self.uhf}\n"
        except ValueError:
            pass
        xyz_str += commentline
        for i in range(self.num_atoms):
            xyz_str += (
                f"{PSE[self.ati[i] + 1]:<5} "
                + f"{self.xyz[i, 0]:>12.7f} "
                + f"{self.xyz[i, 1]:>12.7f} "
                + f"{self.xyz[i, 2]:>12.7f}\n"
            )
        return xyz_str

    def get_coord_str(self) -> str:
        """
        Obtain a string with the full 'coord' file information of the molecule.
        """
        coord_str = "$coord\n"
        for i in range(self.num_atoms):
            coord_str += (
                f"{self.xyz[i, 0] / BOHR2AA:>20.14f} "
                + f"{self.xyz[i, 1] / BOHR2AA:>20.14f} "
                + f"{self.xyz[i, 2] / BOHR2AA:>20.14f} "
                + f"{PSE[self.ati[i] + 1]}\n"
            )
        coord_str += "$end\n"
        return coord_str

    def write_xyz_to_file(self, filename: str | Path | None = None):
        """
        Write the XYZ coordinates of the molecule to a file.

        The layout of the file is as follows:
        ```
        num_atoms
        'Generated by mindlessgen-v{__version__}'
        <symbol 1> <x1> <y1> <z1>
        <symbol 2> <x2> <y2> <z2>
        ...
        ```

        :param filename: The name of the file to write to.
        """
        # raise an error if the number of atoms is not set
        if self._num_atoms is None:
            raise ValueError("Number of atoms not set.")
        if not self._ati.size:
            raise ValueError("Atomic numbers not set.")
        if not self._xyz.size:
            raise ValueError("Atomic coordinates not set.")

        if filename:
            if not isinstance(filename, Path):
                filename = Path(filename).resolve()
        else:
            filename = Path("mlm_" + self.name + ".xyz").resolve()

        with open(filename, "w", encoding="utf8") as f:
            f.write(self.get_xyz_str())
        # if the charge is set, write it to a '.CHRG' file
        if self._charge is not None and self._charge != 0:
            with open(filename.with_suffix(".CHRG"), "w", encoding="utf8") as f:
                f.write(f"{self.charge}\n")
        # if the UHF is set, write it to a '.UHF' file
        if self._uhf is not None and self._uhf > 0:
            with open(filename.with_suffix(".UHF"), "w", encoding="utf8") as f:
                f.write(f"{self.uhf}\n")

    def write_coord_to_file(self, filename: str | Path | None = None):
        """
        Write the 'coord' file of the molecule to a file.

        The layout of the file is as follows:
        ```
        $coord
        <x1> <y1> <z1> <symbol 1>
        <x2> <y2> <z2> <symbol 2>
        ...
        $end
        ```

        :param filename: The name of the file to write to.
        """
        # raise an error if the number of atoms is not set
        if self._num_atoms is None:
            raise ValueError("Number of atoms not set.")
        if not self._ati.size:
            raise ValueError("Atomic numbers not set.")
        if not self._xyz.size:
            raise ValueError("Atomic coordinates not set.")

        if filename:
            if not isinstance(filename, Path):
                filename = Path(filename).resolve()
        else:
            filename = Path("mlm_" + self.name + ".coord").resolve()

        with open(filename, "w", encoding="utf8") as f:
            f.write(self.get_coord_str())
        # if the charge is set, write it to a '.CHRG' file
        if self._charge is not None and self._charge != 0:
            chrg_filename = filename.parent / ".CHRG"
            with open(chrg_filename, "w", encoding="utf8") as f:
                f.write(f"{self.charge}\n")
        # if the UHF is set, write it to a '.UHF' file
        if self._uhf is not None and self._uhf > 0:
            uhf_filename = filename.parent / ".UHF"
            with open(uhf_filename, "w", encoding="utf8") as f:
                f.write(f"{self.uhf}\n")

    def read_xyz_from_file(self, filename: str | Path) -> None:
        """
        Read the XYZ coordinates of the molecule from a file.

        The layout of the file is as follows:
        ```
        num_atoms
        'Generated by mindlessgen-v{__version__}'
        <symbol 1> <x1> <y1> <z1>
        <symbol 2> <x2> <y2> <z2>
        ...
        ```

        :param filename: The name of the file to read from.
        """
        with open(filename, encoding="utf8") as f:
            lines = f.readlines()
            # read the number of atoms
            self.num_atoms = int(lines[0])
            # read the atomic coordinates
            self.xyz = np.zeros((self.num_atoms, 3))
            self.ati = np.zeros(self.num_atoms, dtype=int)
            self.atlist = np.zeros(103, dtype=int)
            for i in range(self.num_atoms):
                line = lines[i + 2].split()
                self.ati[i] = PSE_NUMBERS[line[0].lower()] - 1
                self.xyz[i, 0] = float(line[1])
                self.xyz[i, 1] = float(line[2])
                self.xyz[i, 2] = float(line[3])
                self.atlist[self.ati[i]] += 1

    def read_xyz_from_coord(self, filename: str | Path) -> None:
        """
        Read the XYZ coordinates of the molecule from a 'coord' file.

        The layout of the file is as follows:
        ```
        $coord
         0.00000000000000      0.00000000000000      3.60590687077610     u
         0.00000000000000      0.00000000000000     -3.60590687077610     u
         0.00000000000000      2.74684941070244      0.00000000000000     F
         3.72662552762076      0.00000000000000      5.36334486193405     F
        -3.72662552762076      0.00000000000000      5.36334486193405     F
         3.72662552762076      0.00000000000000     -5.36334486193405     F
         0.00000000000000     -2.74684941070244      0.00000000000000     F
        -3.72662552762076      0.00000000000000     -5.36334486193405     F
        $end
        ... or ...
        $coord
            0.00000000000000      0.00000000000000      2.30704866281983  u
            0.00000000000000      0.00000000000000     -2.30704866281983  u
        $redundant
            number_of_atoms             2
            degrees_of_freedom          1
            internal_coordinates        1
            frozen_coordinates          0
        # definitions of redundant internals
        1 k  1.0000000000000 stre   1    2           val=   4.61410
                1 non zero eigenvalues  of BmBt
                1           2.000000000    1    0
                1
        $end

        :param filename: The name of the file to read from.
        """
        with open(filename, encoding="utf8") as f:
            lines = f.readlines()
            # number of atoms
            num_atoms = 0
            for line in lines:
                if line.startswith("$coord"):
                    continue
                if line.startswith("$end") or line.startswith("$redundant"):
                    break
                num_atoms += 1
            self.num_atoms = num_atoms
            # read the atomic coordinates
            self.xyz = np.zeros((self.num_atoms, 3))
            self.ati = np.zeros(self.num_atoms, dtype=int)
            self.atlist = np.zeros(103, dtype=int)
            for i in range(self.num_atoms):
                line_entries = lines[i + 1].split()
                self.xyz[i, 0] = float(line_entries[0]) * BOHR2AA
                self.xyz[i, 1] = float(line_entries[1]) * BOHR2AA
                self.xyz[i, 2] = float(line_entries[2]) * BOHR2AA
                self.ati[i] = PSE_NUMBERS[line_entries[3].lower()] - 1
                self.atlist[self.ati[i]] += 1

    def read_charge_from_file(self, filename: str | Path):
        """
        Read the charge of the molecule from a file.

        The layout of the file is as follows:
        ```
        charge
        ```

        :param filename: The name of the file to read from.
        """
        with open(filename, encoding="utf8") as f:
            self.charge = int(f.readline().strip().split()[0])

    def read_uhf_from_file(self, filename: str | Path):
        """
        Read the UHF of the molecule from a file.

        The layout of the file is as follows:
        ```
        uhf
        ```

        :param filename: The name of the file to read from.
        """
        with open(filename, encoding="utf8") as f:
            self.uhf = int(f.readline().strip().split()[0])

    def sum_formula(self) -> str:
        """
        Get the sum formula of the molecule.
        """
        if not self._atlist.size:
            raise ValueError("Atomic numbers not set.")
        sumformula = ""
        # begin with C, H, N, O (i.e., 6, 1, 7, 8)
        for i in [5, 0, 6, 7]:
            if self._atlist[i] > 0:
                sumformula += PSE[i + 1] + str(self.atlist[i])
        # Go through all entries of self._ati that are not zero
        for elem, count in enumerate(self.atlist):
            if elem not in [5, 0, 6, 7] and count > 0:
                sumformula += PSE[elem + 1] + str(self.atlist[elem])
        return sumformula

    def copy(self) -> "Molecule":
        """
        Create a deep copy of the molecule instance.

        :return: A new instance of Molecule that is a deep copy of the current instance.
        """
        # Create a new instance of Molecule
        new_molecule = Molecule(self._name)

        # Deep copy all attributes
        if self._num_atoms is not None:
            new_molecule.num_atoms = copy.deepcopy(self.num_atoms)
        if self._charge is not None:
            new_molecule.charge = copy.deepcopy(self.charge)
        if self._uhf is not None:
            new_molecule.uhf = copy.deepcopy(self.uhf)
        if self._atlist.size:
            new_molecule.atlist = copy.deepcopy(self.atlist)
        if self._xyz.size:
            new_molecule.xyz = copy.deepcopy(self.xyz)
        if self._ati.size:
            new_molecule.ati = copy.deepcopy(self.ati)

        return new_molecule

    def set_name_from_formula(self) -> None:
        """
        Get the name of the molecule from its sum formula.

        :Arguments: None

        :Returns: None
        """

        molname = self.sum_formula()
        # add a random hash to the name
        hashname = hashlib.sha256(self.rng.bytes(32)).hexdigest()[:6]
        self.name = f"{molname}_{hashname}"


class ORCA:
    """
    This class handles all interaction with the ORCA external dependency.
    """

    def __init__(self, path: str | Path, ncores: int = 1, maxcore: int = 4000) -> None:
        """
        Initialize the ORCA class.
        """
        if isinstance(path, str):
            self.path: Path = Path(path).resolve()
        elif isinstance(path, Path):
            self.path = path
        else:
            raise TypeError("orca_path should be a string or a Path object.")
        self.ncores = ncores
        self.maxcore = maxcore

    def optimize(
        self, molecule: Molecule, tmp_path: str | Path, verbosity: int = 1
    ) -> Molecule:
        """
        Optimize a molecule using ORCA.
        """
        # write the molecule to a temporary file
        molfile = "mol.xyz"
        if isinstance(tmp_path, str):
            tmp_path = Path(tmp_path).resolve()
        elif isinstance(tmp_path, Path):
            tmp_path = tmp_path.resolve()
        else:
            raise TypeError("tmp_path should be a string or a Path object.")
        tmp_path.mkdir(exist_ok=True)
        molecule.write_xyz_to_file(tmp_path / molfile)

        inputname = "orca_opt.inp"
        orca_input = self._gen_input(molecule, molfile, True)
        if verbosity > 1:
            print("ORCA input file:\n##################")
            print(orca_input)
            print("##################")
        with open(tmp_path / inputname, "w", encoding="utf8") as f:
            f.write(orca_input)

        # run orca
        arguments = [
            inputname,
        ]

        orca_log_out, orca_log_err, return_code = self._run(
            temp_path=tmp_path, arguments=arguments
        )
        if verbosity > 2:
            print(orca_log_out)
        if return_code != 0:
            raise RuntimeError(
                f"ORCA failed with return code {return_code}:\n{orca_log_err}"
            )

        # read the optimized molecule from the output file
        xyzfile = Path(tmp_path / inputname).resolve().with_suffix(".xyz")
        optimized_molecule = molecule.copy()
        optimized_molecule.read_xyz_from_file(xyzfile)
        return optimized_molecule

    def singlepoint(
        self, molecule: Molecule, tmp_path: str | Path, verbosity: int = 1
    ) -> str:
        """
        Perform a single point calculation using ORCA.
        """
        # write the molecule to a temporary file
        molfile = "mol.xyz"
        if isinstance(tmp_path, str):
            tmp_path = Path(tmp_path).resolve()
        elif isinstance(tmp_path, Path):
            tmp_path = tmp_path.resolve()
        else:
            raise TypeError("tmp_path should be a string or a Path object.")
        tmp_path.mkdir(exist_ok=True)
        molecule.write_xyz_to_file(tmp_path / molfile)

        # write the input file
        inputname = "orca.inp"
        orca_input = self._gen_input(molecule, molfile)
        if verbosity > 2:
            print("ORCA input file:\n##################")
            print(self._gen_input(molecule, molfile))
            print("##################")
        with open(tmp_path / inputname, "w", encoding="utf8") as f:
            f.write(orca_input)

        # run orca
        arguments = [
            inputname,
        ]
        orca_log_out, orca_log_err, return_code = self._run(
            temp_path=tmp_path, arguments=arguments
        )
        if verbosity > 2:
            print(orca_log_out)
        if return_code != 0:
            raise RuntimeError(
                f"ORCA failed with return code {return_code}:\n{orca_log_err}"
            )

        return orca_log_out

    def check_gap(
        self, molecule: Molecule, threshold: float, verbosity: int = 1
    ) -> bool:
        """
        Check if the HL gap is larger than a given threshold.
        """
        raise NotImplementedError("check_gap not implemented for ORCA.")

    def _run(self, temp_path: Path, arguments: list[str]) -> tuple[str, str, int]:
        """
        Run ORCA with the given arguments.

        Arguments:
        arguments (list[str]): The arguments to pass to orca.

        Returns:
        tuple[str, str, int]: The output of the ORCA calculation (stdout and stderr)
                              and the return code
        """
        try:
            orca_out = sp.run(
                [str(self.path)] + arguments,
                cwd=temp_path,
                capture_output=True,
                check=True,
            )
            # get the output of the ORCA calculation (of both stdout and stderr)
            orca_log_out = orca_out.stdout.decode("utf8", errors="replace")
            orca_log_err = orca_out.stderr.decode("utf8", errors="replace")
            # check if the output contains "ORCA TERMINATED NORMALLY"
            if "ORCA TERMINATED NORMALLY" not in orca_log_out:
                raise sp.CalledProcessError(
                    1,
                    str(self.path),
                    orca_log_out.encode("utf8"),
                    orca_log_err.encode("utf8"),
                )
            return orca_log_out, orca_log_err, 0
        except sp.CalledProcessError as e:
            orca_log_out = e.stdout.decode("utf8", errors="replace")
            orca_log_err = e.stderr.decode("utf8", errors="replace")
            return orca_log_out, orca_log_err, e.returncode
        finally:
            # write orca_log_out and orca_log_err to files
            with open(temp_path / "orca_log.out", "w", encoding="utf8") as f:
                f.write(orca_log_out)
            with open(temp_path / "orca_error.out", "w", encoding="utf8") as f:
                f.write(orca_log_err)

            # now clean up the temporary directory
            files_to_delete = [
                "*.tmp",
                # "*.gbw",
                "*.densities",
                "*.ges",
                "*.hostnames",
                "*.bas*",
                "*_atom*.inp",
                "*_atom*.out",
                "*.bibtex",
                "*.property.txt",
                "*_property.txt",
                "*.err",
                "*.engrad",
                "*.opt",
                "*_trj.xyz",
                "*.goat.*.out",
                "*.goat.*.xyz",
                "*.goat.*.carthess",
                "*.goat.*.scfgrad.inp",
                "*.finalensemble*.xyz",
                "*.globalminimum.xyz",
                "*.densitiesinfo",
            ]
            for pattern in files_to_delete:
                for file in temp_path.glob(pattern):
                    file.unlink()

    def _gen_input(
        self,
        molecule: Molecule,
        xyzfile: str,
        optimization: bool = False,
    ) -> str:
        """
        Generate a default input file for ORCA.
        """
        orca_input = "! wB97M-V def2-TZVPPD EnGrad\n"
        orca_input += "! StrongSCF DEFGRID3\n"
        orca_input += "! NoTRAH\n"
        orca_input += "! PrintBasis\n"
        if optimization:
            orca_input += "! OPT\n"
        if any(atom >= 86 for atom in molecule.ati):
            orca_input += "! AutoAux\n"
        orca_input += f"%pal nprocs {self.ncores} end\n"
        orca_input += f"%maxcore {self.maxcore}\n"
        # "! AutoAux" keyword for super-heavy elements as def2/J ends at Rn
        if any(atom >= 86 for atom in molecule.ati):
            orca_input += "%basis\n"
            # take heavy atoms from the molecule.atlist
            # -> if the any entry of index 86 or higher is not zero
            heavy_atoms = np.where(molecule.atlist[85:] > 0)[0] + 86
            for heavy_atom in heavy_atoms:
                orca_input += f'\tNewGTO  {PSE[heavy_atom]} "def-TZVP" end\n'
                orca_input += f'\tNewECP  {PSE[heavy_atom]} "def-ECP" end\n'
            orca_input += "end\n"
        orca_input += "%scf\n\tMaxIter 500\nend\n"
        orca_input += "%elprop\n\tOrigin 0.0,0.0,0.0\nend\n"
        orca_input += "%output\n"
        orca_input += "\tPrint[ P_Internal ]   0  # internal coordinates\n"
        orca_input += "\tPrint[ P_OrbEn ]      1  # orbital energies\n"
        orca_input += "\tPrint[ P_Mayer ]      0  # Mayer population analysis\n"
        orca_input += "\tPrint[ P_Loewdin ]    0  # Loewdin population analysis\n"
        orca_input += "\tPrint[ P_Hirshfeld ]  1  # Hirshfeld density analysis\n"
        orca_input += "end\n"

        orca_input += f"* xyzfile {molecule.charge} {molecule.uhf + 1} {xyzfile}\n"
        return orca_input


# TODO: 1. Convert this to a @staticmethod of Class ORCA
#       2. Rename to `get_method` or similar to enable an abstract interface
#       3. Add the renamed method to the ABC `QMMethod`
#       4. In `main.py`: Remove the passing of the path finder functions as arguments
#          and remove the boiler plate code to make it more general.
def get_orca_path(binary_name: str | Path | None = None) -> Path:
    """
    Get the path to the orca binary based on different possible names
    that are searched for in the PATH.
    """
    default_orca_names: list[str | Path] = ["orca", "orca_dev"]
    # put binary name at the beginning of the lixt to prioritize it
    if binary_name is not None:
        binary_names = [binary_name] + default_orca_names
    else:
        binary_names = default_orca_names
    # Get ORCA path from 'which orca' command
    for binpath in binary_names:
        which_orca = sh.which(binpath)
        if which_orca:
            orca_path = Path(which_orca).resolve()
            return orca_path
    raise ImportError("'orca' binary could not be found.")


def convert_actinide(inp_mol: Molecule, source: str, target: str) -> Molecule:
    """
    Convert the actinide element in the molecule to the target element.
    """
    # exchange all atoms of the "source" element with the "target" element
    inp_mol.ati = np.where(
        inp_mol.ati == PSE_NUMBERS[source.lower()] - 1,
        PSE_NUMBERS[target.lower()] - 1,
        inp_mol.ati,
    )
    # take the existing atlist, and add the integer entry of the source element to the integer entry of the target element
    inp_mol.atlist[PSE_NUMBERS[target.lower()] - 1] += inp_mol.atlist[
        PSE_NUMBERS[source.lower()] - 1
    ]
    inp_mol.atlist[PSE_NUMBERS[source.lower()] - 1] = 0
    # in the molecule.name, replace any occurence of the source element symbol with the target element symbol
    inp_mol.name = inp_mol.name.replace(source.lower(), target.lower())
    inp_mol.name = inp_mol.name.replace(source.upper(), target.upper())
    return inp_mol


def calculate_spin_state(orca: ORCA, mol: Molecule, calc_dir: Path, verbosity: int):
    """
    Calculate the spin state of the molecule.
    """
    try:
        orca_output = orca.singlepoint(mol, calc_dir, verbosity=verbosity)
        total_energy = parse_orca_energy(orca_output)
        if verbosity > 1:
            print(
                f" \tORCA calculation in '{calc_dir.name}' successful. Energy: {total_energy}"
            )
    except RuntimeError as e:
        print(f"Error in ORCA calculation: {e}")
        total_energy = 0.0
        if verbosity > 1:
            print(
                f" \tORCA calculation '{calc_dir.name}' failed. Setting energy to zero."
            )
    return total_energy


def process_molecule_directory(
    arguments: argparse.Namespace,
):
    """
    Process a directory of molecules.
    """
    orca = ORCA(path=get_orca_path(), ncores=arguments.mpi, maxcore=arguments.maxcore)
    target_dir = arguments.output / arguments.target.capitalize()
    target_dir.mkdir(parents=True, exist_ok=True)

    source_dir: Path = arguments.source.resolve()
    if arguments.original_element:
        source_element = arguments.original_element
        if source_element.lower() not in PSE_NUMBERS:
            raise ValueError(f"Element {source_element} not in the periodic table.")
    else:
        source_element = source_dir.name

    # list of successful molecules
    successful_molecules: list[Molecule] = []

    # check if the source and target elements are neighbours in the periodic table
    if not check_if_neighbours(source_element, arguments.target):
        raise ValueError(
            f"Source ({source_element}) and target ({arguments.target}) elements are not neighbours in the periodic table."
        )

    for molecule_dir in source_dir.iterdir():
        # if molecule_dir not a directory
        if not molecule_dir.is_dir() or molecule_dir.name.startswith("."):
            continue
        if arguments.verbosity > 0:
            print(f"Processing molecule: {molecule_dir.name}\n")
        coord_file = molecule_dir / "coord"
        if not coord_file.exists():
            continue

        molecule = Molecule.read_mol_from_coord(coord_file)
        molecule.name = molecule_dir.name
        molecule = convert_actinide(molecule, source_element, arguments.target)
        ### CAUTION: UHF has not been adjusted yet!
        if arguments.verbosity > 1:
            print("#### Molecule before ORCA calculation ####")
            print(molecule)
            print("##########################################\n")

        # create new directory in target directory for the molecule
        mol_dir = target_dir / molecule.name
        mol_dir.mkdir(parents=True, exist_ok=True)

        # deep copy of original UHF value
        original_uhf = copy.deepcopy(molecule.uhf)

        # Perform ORCA calculations
        try:
            if molecule.uhf > 0:
                if arguments.verbosity > 0:
                    print(
                        "Molecule has unpaired electrons. "
                        + "Performing closed-shell and open-shell calculations..."
                    )
                tmp_molecule = molecule.copy()
                closed_shell_uhf = tmp_molecule.uhf - (
                    1 * molecule.atlist[PSE_NUMBERS[arguments.target.lower()] - 1]
                )
                tmp_molecule.uhf = closed_shell_uhf
                if tmp_molecule.uhf < 0:
                    print("We cannot assign a negative number of unpaired electrons.")
                    closed_shell_energy = 0.0
                else:
                    closed_shell_dir = mol_dir / "closed_shell"
                    closed_shell_energy = calculate_spin_state(
                        orca,
                        tmp_molecule,
                        closed_shell_dir,
                        verbosity=arguments.verbosity,
                    )
            else:
                closed_shell_energy = None
                if arguments.verbosity > 0:
                    print(
                        "Molecule has no unpaired electrons. Performing only the open-shell calculation..."
                    )
            tmp_molecule = molecule.copy()
            open_shell_uhf = tmp_molecule.uhf + (
                1 * molecule.atlist[PSE_NUMBERS[arguments.target.lower()] - 1]
            )  # number of electrons varies by the number of elements exchanged
            tmp_molecule.uhf = open_shell_uhf
            if tmp_molecule.uhf > calculate_f_electrons(tmp_molecule.atlist):
                print(
                    f"\tNumber of unpaired electrons ({tmp_molecule.uhf}) in the molecule is larger than "
                    + f"the calculated number of f and d electrons ({calculate_f_electrons(tmp_molecule.atlist)})."
                )
                open_shell_energy = 0.0
            else:
                open_shell_dir = mol_dir / "open_shell"
                open_shell_energy = calculate_spin_state(
                    orca, tmp_molecule, open_shell_dir, verbosity=arguments.verbosity
                )

            # In this step, final energy and UHF are assigned
            if molecule.uhf > 0:
                # if both energies are zero, skip the molecule
                if closed_shell_energy == 0.0 and open_shell_energy == 0.0:
                    raise ValueError("Both closed-shell and open-shell energies failed")

                molecule.uhf = (
                    closed_shell_uhf
                    if closed_shell_energy < open_shell_energy  # type: ignore
                    else open_shell_uhf
                )
                molecule.energy = min(closed_shell_energy, open_shell_energy)  # type: ignore

                if arguments.verbosity > 0:
                    print(f"Molecule: {molecule.name}")
                    print(f"\tClosed-shell energy: {closed_shell_energy}")
                    print(f"\tOpen-shell energy: {open_shell_energy}")
                    print(f"\tPreferred UHF: {molecule.uhf}, Energy: {molecule.energy}")
            else:
                if open_shell_energy == 0.0:
                    raise ValueError("Only available open-shell calculation failed.")
                molecule.energy = open_shell_energy
                molecule.uhf = open_shell_uhf
                if arguments.verbosity > 0:
                    print(f"Molecule: {molecule.name}")
                    print(f"\tOpen-shell energy: {molecule.energy}")
        except ValueError as e:
            print(f"No energy evaluation succesful: {e}")
            print(f"Skipping molecule {molecule.name}...")
            continue

        # write the molecule to the target directory
        molecule.write_xyz_to_file(mol_dir / "struc.xyz")
        molecule.write_coord_to_file(mol_dir / "coord")
        # copy GBW file from preferred calculation to the target directory
        if original_uhf > 0 and closed_shell_energy < open_shell_energy:  # type: ignore
            lowest_dir = closed_shell_dir
        else:
            lowest_dir = open_shell_dir
        postprocess_molecule(molecule, lowest_dir, "orca_log.out", arguments.verbosity)
        successful_molecules.append(molecule)

    if arguments.verbosity > 0:
        print("Successfully processed molecules:")
        for mol in successful_molecules:
            print(f"\t{mol.name}")
        print(f"Processed {len(successful_molecules)} molecules.")
    # write list of successful molecules to a CSV file
    with open(
        target_dir / "fitmolecules.csv", "w", newline="", encoding="utf8"
    ) as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Molecule", "UHF", "Energy"])
        for mol in successful_molecules:
            csvwriter.writerow([mol.name, mol.uhf, mol.energy])
    print(
        f"List of successful molecules written to '{target_dir.resolve()}/fitmolecules.csv'"
    )


def write_tm_control_file(
    energy: float, dipole: list[float], filename: str | Path = "control"
):
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
    with open(filename, "w", encoding="utf8") as f:
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


def write_tm_gradient(
    gradient: list[list[float]],
    xyz: list[list[float]],
    symbols: list[str],
    energy: float,
    filename: str | Path = "gradient",
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
    with open(filename, "w", encoding="utf8") as f:
        f.write("$grad\n")
        f.write(
            f"  cycle =      1    SCF energy =   {energy}   |dE/dxyz| =  {norm_grad}\n"
        )
        for i, coord in enumerate(xyz):
            f.write(
                f"{coord[0] / BOHR2AA:>20.11f} "
                + f"{coord[1] / BOHR2AA:>20.11f} "
                + f"{coord[2] / BOHR2AA:>20.11f} "
                + f"{symbols[i]}\n"
            )
        for i, grad in enumerate(gradient):
            f.write(f"{grad[0]:>20.8e} {grad[1]:>20.8e} {grad[2]:>20.8e}\n")
        f.write("$end\n")


def write_multiwfn_charges(
    charges: list[float], ati: list[int], xyz: list[list[float]], file: Path
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
    with open(file, "w", encoding="utf8") as f:
        for i, charge in enumerate(charges):
            f.write(
                f"{PSE[ati[i] + 1]:>2} "
                + f"{xyz[i][0] / BOHR2AA:>14.8f} "
                + f"{xyz[i][1] / BOHR2AA:>14.8f} "
                + f"{xyz[i][2] / BOHR2AA:>14.8f} "
                + f"{charge:>14.8f}\n"
            )


def postprocess_molecule(mol: Molecule, calc_dir: Path, orca_file: str, verbosity: int):
    """
    Postprocess the molecule after the ORCA calculation.
    """
    # parse the Hirshfeld charges from the ORCA output file
    with open(calc_dir / orca_file, "r", encoding="utf8") as f:
        orca_output = f.read()
    charges = parse_orca_hirshfeld(orca_output)
    if verbosity > 1:
        print(f"\tHirshfeld charges: {charges}")
    ati = list(mol.ati)
    xyz = list(mol.xyz)
    write_multiwfn_charges(charges, ati, xyz, calc_dir.parent / "multiwfn.chg")
    dipole = parse_orca_dipole(orca_output)
    if verbosity > 1:
        print(f"\tDipole moment: {dipole}")
    write_tm_control_file(mol.energy, dipole, calc_dir.parent / "control")  # type: ignore
    # parse gradient from ORCA output file
    gradient = parse_orca_gradient(orca_output)
    if verbosity > 1:
        print(f"\tGradient: {gradient}")
    # convert ati to a list of PSE symbols
    symbols = [PSE[elem + 1] for elem in ati]
    write_tm_gradient(gradient, xyz, symbols, mol.energy, calc_dir.parent / "gradient")  # type: ignore


# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Process molecule directories.")
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        type=Path,
        help="Source directory path. Should be equivalent to the element symbol.",
    )
    parser.add_argument(
        "--target",
        "-t",
        required=True,
        type=str,
        default=None,
        help="Target element symbol.",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="Output directory path."
    )
    parser.add_argument(
        "--original-element",
        "-oe",
        type=str,
        default=None,
        required=False,
        help="Original element symbol.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        default=1,
        required=False,
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--mpi",
        "-m",
        type=int,
        default=1,
        required=False,
        help="Number of MPI processes to use.",
    )
    parser.add_argument(
        "--maxcore",
        "-mc",
        type=int,
        default=4000,
        required=False,
        help="Memory limit per core in MB. Options: <int>",
    )

    args = parser.parse_args()
    process_molecule_directory(args)


if __name__ == "__main__":
    main()
