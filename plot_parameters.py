#!/usr/bin/env python3
"""
Plot parameter values from a semiempirical parameter file over the period and the group
of a specified element.

Usage example:
    python plot_parameters.py --cell "1,1" --element 5
This will plot the parameter cell (row 1, col 1) for all elements in the period of atomic number 5
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot parameter cell(s) (row,col) from a block of parameters "
        + "for elements in the same period and group."
    )
    parser.add_argument(
        "--cell",
        type=str,
        required=True,
        help="Cell(s) in the parameter block. Use semicolon-separated 'row,col' pairs (e.g. '1,1;2,2').",
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
    parser.add_argument(
        "--period", action="store_true", help="Plot only the period trend"
    )
    parser.add_argument(
        "--group", action="store_true", help="Plot only the group trend"
    )
    args = parser.parse_args()

    def parse_cell_input(cell_str):
        # Accepts semicolon-separated list of row,col pairs (e.g. "3,2;4,2;4,3")
        # or ranges using dashes (e.g. "3-4,2" means (3,2), (4,2))
        parts = cell_str.replace(" ", "").split(";")
        cells = []
        for part in parts:
            row_col = part.split(",")
            if len(row_col) != 2:
                raise ValueError(f"Invalid format for cell: '{part}'")
            row, col = map(int, row_col)
            cells.append((row, col))
        return cells

    cell_list = parse_cell_input(args.cell)
    if not cell_list:
        parser.error("No valid --cell parameters specified.")

    pt = PeriodicTable()
    param_data = ParameterData(args.param_file)

    # For each cell, gather period and group data, plot all lines in one plot.
    _, ax = plt.subplots(figsize=(10, 6))
    symbol = pt.get_symbol(args.element)
    period = pt.get_period(args.element)
    group = pt.get_group(args.element)
    ylabels = []
    plot_period = args.period or not (args.period or args.group)
    plot_group = args.group or not (args.period or args.group)
    for row, col in cell_list:
        # period line
        period_elements = sorted(pt.get_elements_in_period(period), key=lambda x: x[0])
        period_z = []
        period_y = []
        for z, _ in period_elements:
            try:
                y_val = param_data.get_parameter_cell(z, row, col)
                period_z.append(z)
                period_y.append(y_val)
            except (ValueError, IndexError):
                continue
        # group line
        group_elements = sorted(pt.get_elements_in_group(group), key=lambda x: x[0])
        group_z = []
        group_y = []
        for z, _ in group_elements:
            try:
                y_val = param_data.get_parameter_cell(z, row, col)
                group_z.append(z)
                group_y.append(y_val)
            except (ValueError, IndexError):
                continue
        if plot_period:
            ax.plot(
                period_z,
                period_y,
                marker="o",
                label=f"Cell ({row},{col}) - Period {period}",
            )
        if plot_group:
            ax.plot(
                group_z,
                group_y,
                marker="s",
                label=f"Cell ({row},{col}) - Group {group}",
            )
        ylabels.append(f"({row},{col})")
    # Set x-ticks using atomic number (Z) as tick labels
    all_z: set[int] = set()
    for row, col in cell_list:
        if plot_period:
            all_z.update(z for (z, _) in pt.get_elements_in_period(period))
        if plot_group:
            all_z.update(z for (z, _) in pt.get_elements_in_group(group))
    all_z_list = list(sorted(all_z))
    ax.set_xticks(all_z_list)
    ax.set_xticklabels([f"{pt.get_symbol(z)}: {z}" for z in all_z_list], rotation=90)
    ax.set_xlabel("Atomic Number (Z)")
    # Improved ylabel if multiple cells
    if len(cell_list) == 1:
        ax.set_ylabel(f"Parameter Value {ylabels[0]}")
    else:
        ax.set_ylabel(f"Parameter Value (cells: {', '.join(ylabels)})")
    ax.legend()
    ax.set_title(
        f"Parameter trends across Periods and Groups\n"
        f"Cells: {', '.join([f'({r},{c})' for (r, c) in cell_list])}, "
        + f"Reference element: {symbol}({args.element})"
    )
    # Add extra space at the bottom for rotated labels
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
