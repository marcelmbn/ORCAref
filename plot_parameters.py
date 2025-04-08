#!/usr/bin/env python3
"""
Plot parameter values from a semiempirical parameter file over the period and the group
of a specified element.

Usage example:
    python plot_parameters.py --par 1 --element 5
This will plot parameter 1 (1-indexed) for all elements in the period of atomic number 5
and for all elements in the group of atomic number 5, all in one plot.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt


class PeriodicTable:
    """
    Provides a complete periodic table mapping (for Z=1 to 103) using IUPAC conventions.
    The dictionary keys are atomic numbers and the values are tuples: (symbol, period, group).
    Note: For the lanthanides and actinides (Z=57–71 and Z=89–103) the group is set to 3.
    """

    def __init__(self) -> None:
        self.elements: dict[int, tuple[str, int, int]] = {
            1: ("H", 1, 1),
            2: ("He", 1, 18),
            3: ("Li", 2, 1),
            4: ("Be", 2, 2),
            5: ("B", 2, 13),
            6: ("C", 2, 14),
            7: ("N", 2, 15),
            8: ("O", 2, 16),
            9: ("F", 2, 17),
            10: ("Ne", 2, 18),
            11: ("Na", 3, 1),
            12: ("Mg", 3, 2),
            13: ("Al", 3, 13),
            14: ("Si", 3, 14),
            15: ("P", 3, 15),
            16: ("S", 3, 16),
            17: ("Cl", 3, 17),
            18: ("Ar", 3, 18),
            19: ("K", 4, 1),
            20: ("Ca", 4, 2),
            21: ("Sc", 4, 3),
            22: ("Ti", 4, 4),
            23: ("V", 4, 5),
            24: ("Cr", 4, 6),
            25: ("Mn", 4, 7),
            26: ("Fe", 4, 8),
            27: ("Co", 4, 9),
            28: ("Ni", 4, 10),
            29: ("Cu", 4, 11),
            30: ("Zn", 4, 12),
            31: ("Ga", 4, 13),
            32: ("Ge", 4, 14),
            33: ("As", 4, 15),
            34: ("Se", 4, 16),
            35: ("Br", 4, 17),
            36: ("Kr", 4, 18),
            37: ("Rb", 5, 1),
            38: ("Sr", 5, 2),
            39: ("Y", 5, 3),
            40: ("Zr", 5, 4),
            41: ("Nb", 5, 5),
            42: ("Mo", 5, 6),
            43: ("Tc", 5, 7),
            44: ("Ru", 5, 8),
            45: ("Rh", 5, 9),
            46: ("Pd", 5, 10),
            47: ("Ag", 5, 11),
            48: ("Cd", 5, 12),
            49: ("In", 5, 13),
            50: ("Sn", 5, 14),
            51: ("Sb", 5, 15),
            52: ("Te", 5, 16),
            53: ("I", 5, 17),
            54: ("Xe", 5, 18),
            55: ("Cs", 6, 1),
            56: ("Ba", 6, 2),
            57: ("La", 6, 3),
            58: ("Ce", 6, 19),
            59: ("Pr", 6, 20),
            60: ("Nd", 6, 21),
            61: ("Pm", 6, 22),
            62: ("Sm", 6, 23),
            63: ("Eu", 6, 24),
            64: ("Gd", 6, 25),
            65: ("Tb", 6, 26),
            66: ("Dy", 6, 27),
            67: ("Ho", 6, 28),
            68: ("Er", 6, 29),
            69: ("Tm", 6, 30),
            70: ("Yb", 6, 31),
            71: ("Lu", 6, 32),
            72: ("Hf", 6, 4),
            73: ("Ta", 6, 5),
            74: ("W", 6, 6),
            75: ("Re", 6, 7),
            76: ("Os", 6, 8),
            77: ("Ir", 6, 9),
            78: ("Pt", 6, 10),
            79: ("Au", 6, 11),
            80: ("Hg", 6, 12),
            81: ("Tl", 6, 13),
            82: ("Pb", 6, 14),
            83: ("Bi", 6, 15),
            84: ("Po", 6, 16),
            85: ("At", 6, 17),
            86: ("Rn", 6, 18),
            87: ("Fr", 7, 1),
            88: ("Ra", 7, 2),
            89: ("Ac", 7, 3),
            90: ("Th", 7, 19),
            91: ("Pa", 7, 20),
            92: ("U", 7, 21),
            93: ("Np", 7, 22),
            94: ("Pu", 7, 23),
            95: ("Am", 7, 24),
            96: ("Cm", 7, 25),
            97: ("Bk", 7, 26),
            98: ("Cf", 7, 27),
            99: ("Es", 7, 28),
            100: ("Fm", 7, 29),
            101: ("Md", 7, 30),
            102: ("No", 7, 31),
            103: ("Lr", 7, 32),
        }

    def get_period(self, atomic_number: int) -> int:
        return self.elements[atomic_number][1]

    def get_group(self, atomic_number: int) -> int:
        return self.elements[atomic_number][2]

    def get_symbol(self, atomic_number: int) -> str:
        return self.elements[atomic_number][0]

    def get_elements_in_period(
        self, period: int
    ) -> list[tuple[int, tuple[str, int, int]]]:
        return [(Z, data) for Z, data in self.elements.items() if data[1] == period]

    def get_elements_in_group(
        self, group: int
    ) -> list[tuple[int, tuple[str, int, int]]]:
        return [(Z, data) for Z, data in self.elements.items() if data[2] == group]


class ParameterData:
    """
    Reads the parameter file and stores parameter values by atomic number.

    The file is assumed to have blocks of lines beginning with a header line that consists
    solely of the atomic number. The subsequent lines until the next header are expected to
    contain whitespace‐separated floats representing the parameters for that element.
    """

    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.data: dict[int, list[list[float]]] = {}
        self._parse_file()

    def _parse_file(self) -> None:
        # Now store each non-header line as a separate list of floats.
        current_atomic: int | None = None
        current_lines: list[list[float]] = []
        content = self.file_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Header line: only digits.
            if line.isdigit():
                if current_atomic is not None:
                    self.data[current_atomic] = current_lines
                current_atomic = int(line)
                current_lines = []
            else:
                parts = line.split()
                try:
                    values = [float(x) for x in parts]
                except ValueError:
                    continue
                current_lines.append(values)
        if current_atomic is not None:
            self.data[current_atomic] = current_lines

    def get_parameter_cell(self, atomic_number: int, row: int, col: int) -> float:
        """
        Returns the parameter from the specified row and column
        (both 1-indexed) for the given atomic number.
        For example, get_parameter_cell(atomic_number, 3, 2) returns the second value in the third line.
        Raises a ValueError if the requested cell is empty.
        """
        block = self.data.get(atomic_number)
        if block is None:
            raise ValueError(
                f"No parameter block found for atomic number {atomic_number}"
            )
        i = row - 1
        j = col - 1
        if i < 0 or i >= len(block):
            raise IndexError(f"Row {row} is out of range for element {atomic_number}")
        row_values = block[i]
        if j < 0 or j >= len(row_values):
            raise IndexError(
                f"Column {col} is out of range for element {atomic_number}, row {row}"
            )
        value = row_values[j]
        if value is None:
            raise ValueError(
                f"The requested cell (row {row}, column {col}) is empty for element {atomic_number}"
            )
        return value

    def get_parameter(self, atomic_number: int, par_index: int) -> float:
        """
        Returns the parameter (1-indexed) for the element with the given atomic number,
        flattening the parameter block (a list of lists) into a single list.
        """
        block = self.data.get(atomic_number)
        if block is None:
            raise ValueError(
                f"No parameter data found for atomic number {atomic_number}"
            )
        # Flatten the block (list of lists) into a single list of floats.
        flat_list = [value for row in block for value in row]
        if not (1 <= par_index <= len(flat_list)):
            raise IndexError(
                f"Parameter index {par_index} out of range for element {atomic_number}"
            )
        return flat_list[par_index - 1]


class ParameterPlotter:
    """
    Plots a given parameter (selected by its index) for elements in the same period and group.
    """

    def __init__(
        self, param_data: ParameterData, periodic_table: PeriodicTable
    ) -> None:
        self.param_data = param_data
        self.periodic_table = periodic_table

    def plot_parameter_cell(self, cell: tuple[int, int], element: int) -> None:
        row, col = cell
        # Check the reference element first; if its cell is missing, raise an error.
        ref_val = self.param_data.get_parameter_cell(element, row, col)
        period = self.periodic_table.get_period(element)
        group = self.periodic_table.get_group(element)
        symbol = self.periodic_table.get_symbol(element)

        # Gather values for all elements in the same period.
        period_elements = sorted(
            self.periodic_table.get_elements_in_period(period), key=lambda x: x[0]
        )
        period_x: list[str] = []
        period_y: list[float] = []
        for Z, _ in period_elements:
            try:
                y_val = self.param_data.get_parameter_cell(Z, row, col)
            except ValueError:
                continue
            period_x.append(f"{self.periodic_table.get_symbol(Z)}({Z})")
            period_y.append(y_val)

        # Gather values for all elements in the same group.
        # Gather values for all elements in the same group.
        group_elements = sorted(
            self.periodic_table.get_elements_in_group(group), key=lambda x: x[0]
        )
        group_x: list[str] = []
        group_y: list[float] = []
        for Z, _ in group_elements:
            try:
                y_val = self.param_data.get_parameter_cell(Z, row, col)
            except ValueError:
                continue
            group_x.append(f"{self.periodic_table.get_symbol(Z)}({Z})")
            group_y.append(y_val)

        # Plotting.
        plt.figure(figsize=(10, 6))
        plt.plot(period_x, period_y, marker="o", label=f"Period {period}")
        plt.plot(group_x, group_y, marker="s", label=f"Group {group}")
        plt.xlabel("Element (Symbol (Atomic Number))")
        plt.ylabel(f"Value from row {row}, column {col}")
        plt.title(
            f"Parameter (row {row}, col {col}) for elements in Period {period} and Group {group}\n"
            f"(Reference element: {symbol}({element}))"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_parameter(self, par_index: int, element: int) -> None:
        # Determine period and group for the provided element.
        period = self.periodic_table.get_period(element)
        group = self.periodic_table.get_group(element)
        symbol = self.periodic_table.get_symbol(element)

        # Gather elements in the same period.
        period_elements = sorted(
            self.periodic_table.get_elements_in_period(period), key=lambda x: x[0]
        )
        period_x: list[str] = []
        period_y: list[float] = []
        for Z, _ in period_elements:
            try:
                y_val = self.param_data.get_parameter(Z, par_index)
                period_x.append(f"{self.periodic_table.get_symbol(Z)}({Z})")
                period_y.append(y_val)
            except Exception:
                # Skip if parameter data is not available.
                continue

        # Gather elements in the same group.
        group_elements = sorted(
            self.periodic_table.get_elements_in_group(group), key=lambda x: x[0]
        )
        group_x: list[str] = []
        group_y: list[float] = []
        for Z, _ in group_elements:
            try:
                y_val = self.param_data.get_parameter(Z, par_index)
                group_x.append(f"{self.periodic_table.get_symbol(Z)}({Z})")
                group_y.append(y_val)
            except Exception:
                continue

        # Plotting the data.
        plt.figure(figsize=(10, 6))
        plt.plot(period_x, period_y, marker="o", label=f"Period {period}")
        plt.plot(group_x, group_y, marker="s", label=f"Group {group}")
        plt.xlabel("Element (Symbol (Atomic Number))")
        plt.ylabel(f"Parameter {par_index} Value")
        plt.title(
            f"Parameter {par_index} for elements in Period {period} and Group {group}\n"
            f"(Reference element: {symbol}({element}))"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a parameter cell (row,col) from a block of parameters for elements in the same period and group."
    )
    parser.add_argument(
        "--cell",
        type=str,
        required=True,
        help="Cell in the parameter block in 'row,col' format (e.g., '3,2' means 2nd parameter in the 3rd line)",
    )
    parser.add_argument(
        "--element",
        type=int,
        required=True,
        help="Atomic number of the element of interest",
    )
    parser.add_argument(
        "--param-file",
        type=Path,
        default=Path(".gxtb"),
        help="Path to the parameter file (default: .gxtb)",
    )
    args = parser.parse_args()

    try:
        row_str, col_str = args.cell.split(",")
        row = int(row_str)
        col = int(col_str)
    except Exception:
        parser.error(
            "Invalid --cell format. Expected format: 'row,col' (for example: 3,2)"
        )
    cell = (row, col)

    pt = PeriodicTable()
    param_data = ParameterData(args.param_file)
    plotter = ParameterPlotter(param_data, pt)
    plotter.plot_parameter_cell(cell, args.element)


if __name__ == "__main__":
    main()
