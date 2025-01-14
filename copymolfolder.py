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

    def __init__(self, path: str | Path, ncores: int = 1) -> None:
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
        orca_input = "! wB97M-V def2-TZVPPD\n"
        orca_input += "! StrongSCF DEFGRID3\n"
        orca_input += "! NoTRAH \n"
        orca_input += "! PrintBasis\n"
        if optimization:
            orca_input += "! OPT\n"
        if any(atom >= 86 for atom in molecule.ati):
            orca_input += "! AutoAux\n"
        orca_input += f"%pal nprocs {self.ncores} end\n"
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
        orca_input += "%output\n"
        orca_input += "\tPrint[ P_Internal ]   0  # internal coordinates\n"
        orca_input += "\tPrint[ P_OrbEn ]      1  # orbital energies\n"
        orca_input += "\tPrint[ P_Basis ]      1  # basis set information\n"
        orca_input += "\tPrint[ P_Mayer ]      0  # Mayer population analysis\n"
        orca_input += "\tPrint[ P_Loewdin ]    0  # Loewdin population analysis\n"
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


def parse_orca_energy(output: str) -> float:
    """
    Parse the ORCA output to get the final energy.
    """
    for line in output.split("\n"):
        if "FINAL SINGLE POINT ENERGY" in line:
            energy = float(line.split()[4])
            return energy
    raise ValueError("Energy not found in ORCA output.")


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


def check_if_neighbours(source_elem: str, target_elem: str) -> bool:
    """
    Check if the source and target elements are neighbours in the periodic table.
    """
    source_idx = PSE_NUMBERS[source_elem.lower()]
    target_idx = PSE_NUMBERS[target_elem.lower()]
    if abs(source_idx - target_idx) == 1:
        return True
    return False


def calculate_spin_state(orca: ORCA, mol: Molecule, calc_dir: Path, verbosity: int):
    """
    Calculate the spin state of the molecule.
    """
    try:
        orca_output = orca.singlepoint(mol, calc_dir, verbosity=verbosity)
        total_energy = parse_orca_energy(orca_output)
        if verbosity > 1:
            print(f" \tORCA calculation successful. Energy: {total_energy}")
    except RuntimeError as e:
        print(f"Error in ORCA calculation: {e}")
        total_energy = 0.0
        if verbosity > 1:
            print(" \tORCA alculation failed. Setting energy to zero.")
    return total_energy


def process_molecule_directory(
    arguments: argparse.Namespace,
):
    """
    Process a directory of molecules.
    """
    orca = ORCA(path=get_orca_path(), ncores=arguments.mpi)
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

        # Perform ORCA calculations
        if molecule.uhf > 0:
            if arguments.verbosity > 0:
                print(
                    "Molecule has unpaired electrons. "
                    + "Performing closed-shell and open-shell calculations..."
                )
            try:
                tmp_molecule = molecule.copy()
                tmp_molecule.uhf -= (
                    1 * molecule.atlist[PSE_NUMBERS[arguments.target.lower()] - 1]
                )  # number of electrons varies by the number of elements exchanged
                if tmp_molecule.uhf < 0:
                    print("We cannot assign a negative number of unpaired electrons.")
                    closed_shell_energy = 0.0
                else:
                    calc_dir = mol_dir / "closed_shell"
                    closed_shell_energy = calculate_spin_state(
                        orca, tmp_molecule, calc_dir, verbosity=arguments.verbosity
                    )
                tmp_molecule = molecule.copy()
                tmp_molecule.uhf += (
                    1 * molecule.atlist[PSE_NUMBERS[arguments.target.lower()] - 1]
                )  # number of electrons varies by the number of elements exchanged
                if tmp_molecule.uhf > calculate_f_electrons(tmp_molecule.atlist):
                    print(
                        f"\tNumber of unpaired electrons ({tmp_molecule.uhf}) in the molecule is larger than "
                        + f"the calculated number of f and d electrons ({calculate_f_electrons(tmp_molecule.atlist)})."
                    )
                    open_shell_energy = 0.0
                else:
                    calc_dir = mol_dir / "open_shell"
                    open_shell_energy = calculate_spin_state(
                        orca, tmp_molecule, calc_dir, verbosity=arguments.verbosity
                    )
                # if both energies are zero, skip the molecule
                if closed_shell_energy == 0.0 and open_shell_energy == 0.0:
                    raise ValueError("Both closed-shell and open-shell energies failed")

                preferred_uhf = (
                    molecule.uhf - 1
                    if closed_shell_energy < open_shell_energy
                    else molecule.uhf + 1
                )
                molecule.energy = min(closed_shell_energy, open_shell_energy)

                if arguments.verbosity > 0:
                    print(f"Molecule: {molecule.name}")
                    print(f"  Closed-shell energy: {closed_shell_energy}")
                    print(f"  Open-shell energy: {open_shell_energy}")
                    print(
                        f"  Preferred UHF: {preferred_uhf}, Energy: {molecule.energy}"
                    )
            except ValueError as e:
                print(f"No energy evaluation succesful: {e}")
                print(f"Skipping molecule {molecule.name}...")
                continue
        else:
            if arguments.verbosity > 0:
                print(
                    "Molecule has no unpaired electrons. Performing only the open-shell calculation..."
                )
            molecule.uhf += (
                1 * molecule.atlist[PSE_NUMBERS[arguments.target.lower()] - 1]
            )  # number of electrons varies by the number of elements exchanged
            calc_dir = mol_dir / "open_shell"
            molecule.energy = calculate_spin_state(
                orca, molecule, calc_dir, verbosity=arguments.verbosity
            )
            if molecule.energy == 0.0:
                print("No energy evaluation succesful. Skipping molecule...")
                continue
        # write the molecule to the target directory
        molecule.uhf = preferred_uhf
        molecule.write_xyz_to_file(mol_dir / "struc.xyz")
        molecule.write_coord_to_file(mol_dir / "coord")
        # copy GBW file from preferred calculation to the target directory
        if preferred_uhf > 0:
            gbw_file = mol_dir / "open_shell" / "orca.gbw"
        else:
            gbw_file = mol_dir / "closed_shell" / "orca.gbw"
        sh.copy(gbw_file, mol_dir / "orca.gbw")
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

    args = parser.parse_args()
    process_molecule_directory(args)


if __name__ == "__main__":
    main()
