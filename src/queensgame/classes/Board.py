import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import yaml
import plothist
from .Cell import Cell
from .Queen import Queen
from queensgame.utils import hex_to_color_name, all_sublists, check_available_scans


class Board:
    def __init__(self, size):
        self.size = size
        self.colors = []
        self.cells = [[Cell(row, col) for col in range(size)] for row in range(size)]
        self.seed = None

        # Solving attributes
        self.excluded_cells = []
        self.color_history = []
        self.solution = []
        self.n_recursions = 0

    def __str__(self):
        return f"Board size {self.size}, colors: {self.colors}"

    def __copy__(self):
        new_board = Board(self.size)
        new_board.colors = self.colors.copy()
        new_board.cells = []
        new_board.seed = self.seed
        new_board.excluded_cells = self.excluded_cells.copy()
        new_board.color_history = self.color_history.copy()
        new_board.solution = self.solution.copy()
        for row in self.cells:
            new_row = []
            for cell in row:
                new_row.append(cell.__copy__())
            new_board.cells.append(new_row)
        return new_board

    def read_board(self, yaml_file):
        print(f"Reading board from {yaml_file}")

        with open(yaml_file, "r") as file:
            board_state = yaml.safe_load(file)

        for row in range(self.size):
            for col in range(self.size):
                color = hex_to_color_name(board_state[f"({row}, {col})"])
                self.get_cell(row, col).color = color
                if color not in self.colors:
                    self.colors.append(color)

    def generate_random_board(
        self, seed=None, palette="gist_ncar", no_single_cell=True, verbose=True
    ):
        if self.size < 4:
            raise ValueError("Board size must be at least 4")
        if self.solution:
            raise ValueError("Board already has a solution")

        palette = plothist.get_color_palette(palette, self.size)
        colors = []
        for c in range(len(palette)):
            colors.append(tuple(palette[c][:3]))
            if c not in self.colors:
                self.colors.append(colors[c])

        if len(colors) < self.size:
            raise ValueError("Not enough different colors in palette")

        if seed is None:
            seed_rng = np.random.default_rng()
            self.seed = seed_rng.integers(0, 2**32)
        else:
            self.seed = seed

        rng = np.random.default_rng(self.seed)

        rng.shuffle(colors)

        while not self.is_solved():
            for row in range(self.size):
                busy_cells = [cell.col for cell in self.cells[row] if cell.busy]
                try:
                    col_rng = rng.choice(
                        [i for i in range(0, self.size) if i not in busy_cells]
                    )
                    self.get_cell(row, col_rng).color = colors[row]
                    self.add_queen(row, int(col_rng))
                except ValueError:
                    for coords in self.get_queens_coords():
                        self.remove_queen(coords[0], coords[1])
                        self.get_cell(coords[0], coords[1]).color = "White"

        while self.has_colorless_cells():
            for row in range(self.size):
                for col in range(self.size):
                    if self.get_cell(row, col).color == "White":
                        adjacent_colors = ["White"]
                        for cell in self.get_adjacent_cells(row, col):
                            if cell.color != "White":
                                adjacent_colors.append(cell.color)
                        self.get_cell(row, col).color = adjacent_colors[
                            rng.integers(0, len(adjacent_colors))
                        ]

        for coords in self.get_queens_coords():
            self.remove_queen(coords[0], coords[1])

        if no_single_cell:
            while self.get_smallest_color(value=True) == 1:
                color = self.get_smallest_color()
                cell = self.get_cells_color(color)[0]
                for cell in self.get_adjacent_cells(cell.row, cell.col):
                    if rng.random() > 0.5:
                        cell.color = color
                        break

        if verbose:
            print(f"Board generated with seed {self.seed}")

    def get_cell(self, row, col):
        return self.cells[row][col]

    def get_adjacent_cells(
        self, row, col, with_diagonals=False, only_diagonals=False, only_free=False
    ):
        adjacent_cells = []
        if only_diagonals:
            with_diagonals = True
        if row > 0:
            if not only_diagonals:
                adjacent_cells.append(self.get_cell(row - 1, col))
            if col > 0 and with_diagonals:
                adjacent_cells.append(self.get_cell(row - 1, col - 1))
            if col < self.size - 1 and with_diagonals:
                adjacent_cells.append(self.get_cell(row - 1, col + 1))
        if row < self.size - 1:
            if not only_diagonals:
                adjacent_cells.append(self.get_cell(row + 1, col))
            if col > 0 and with_diagonals:
                adjacent_cells.append(self.get_cell(row + 1, col - 1))
            if col < self.size - 1 and with_diagonals:
                adjacent_cells.append(self.get_cell(row + 1, col + 1))
        if col > 0 and not only_diagonals:
            adjacent_cells.append(self.get_cell(row, col - 1))
        if col < self.size - 1 and not only_diagonals:
            adjacent_cells.append(self.get_cell(row, col + 1))
        if only_free:
            return [cell for cell in adjacent_cells if not cell.busy]
        return adjacent_cells

    def get_cells_color(self, color, only_free=False, remove_excluded_cells=False):
        cells = []
        for row in self.cells:
            for cell in row:
                if cell.color == color:
                    if remove_excluded_cells:
                        if cell.coords not in self.excluded_cells:
                            if only_free:
                                if not cell.busy:
                                    cells.append(cell)
                            else:
                                cells.append(cell)
                    else:
                        if only_free:
                            if not cell.busy:
                                cells.append(cell)
                        else:
                            cells.append(cell)
        return cells

    def get_queen_corners(self, queen):
        row, col = queen.row, queen.col
        corners = []
        if row > 0:
            if col > 0:
                corners.append((row - 1, col - 1))
            if col < self.size - 1:
                corners.append((row - 1, col + 1))
        if row < self.size - 1:
            if col > 0:
                corners.append((row + 1, col - 1))
            if col < self.size - 1:
                corners.append((row + 1, col + 1))
        return corners

    def add_queen(self, row, col):
        queen = Queen(row, col, self.get_cell(row, col).color)
        if not self.get_cell(row, col).busy:
            self.get_cell(row, col).add_queen(queen)
            for corner in self.get_queen_corners(queen):
                self.get_cell(corner[0], corner[1]).busy += 1
            for cell_ in self.cells[row]:
                cell_.busy += 1
            for row_ in self.cells:
                row_[col].busy += 1
            for cell_ in self.get_cells_color(self.get_cell(row, col).color):
                cell_.busy += 1
        else:
            raise ValueError(f"Cell ({row}, {col}) is not free")

    def remove_queen(self, row, col):
        if self.get_cell(row, col).queen is not None:
            for corner in self.get_queen_corners(self.get_cell(row, col).queen):
                self.get_cell(corner[0], corner[1]).reduce_busy()
            for cell_ in self.cells[row]:
                cell_.reduce_busy()
            for row_ in self.cells:
                row_[col].reduce_busy()
            for cell_ in self.get_cells_color(self.get_cell(row, col).color):
                cell_.reduce_busy()
            self.get_cell(row, col).remove_queen()
        else:
            raise ValueError(f"Cell ({row}, {col}) has no queen")

    def count_queens(self):
        count = 0
        for row in self.cells:
            for cell in row:
                if cell.queen:
                    count += 1
        return count

    def count_free_cells(self):
        count = 0
        for row in self.cells:
            for cell in row:
                if not cell.busy and cell.coords not in self.excluded_cells:
                    count += 1
        return count

    def count_free_per_color(self, order=None, remove_empty=False):
        free_per_color = {}
        for color in self.colors:
            free_per_color[color] = 0
        for row in self.cells:
            for cell in row:
                if not cell.busy and cell.coords not in self.excluded_cells:
                    free_per_color[cell.color] += 1
        if remove_empty:
            for color in self.colors:
                if free_per_color[color] == 0:
                    free_per_color.pop(color)
        if order == "asc":
            return {
                k: v
                for k, v in sorted(free_per_color.items(), key=lambda item: item[1])
            }
        elif order == "desc":
            return {
                k: v
                for k, v in sorted(
                    free_per_color.items(), key=lambda item: item[1], reverse=True
                )
            }
        return free_per_color

    def get_smallest_color(self, value=False):
        free_per_color = self.count_free_per_color(order="asc", remove_empty=True)
        if len(free_per_color) == 0:
            return None
        if value:
            return list(free_per_color.values())[0]
        return list(free_per_color.keys())[0]

    def get_queen(self, row, col):
        return self.get_cell(row, col).queen

    def get_queens_coords(self):
        queen_coords = []
        for row in self.cells:
            for cell in row:
                if cell.queen is not None:
                    queen_coords.append(cell.queen.coords)
        return queen_coords

    def get_color_queen(self, color):
        for row in self.cells:
            for cell in row:
                if cell.color == color and cell.queen is not None:
                    return cell.queen
        return None

    def is_solvable(self):
        if (
            len(self.count_free_per_color(remove_empty=True)) + self.count_queens()
            < self.size
        ):
            return False
        return True

    def is_solved(self):
        if self.count_queens() == self.size:
            return True
        return False

    def is_color_solvable(self, color):
        cells = self.get_cells_color(color, only_free=True, remove_excluded_cells=True)
        if len(cells) == 0:
            return False
        return True

    def has_colorless_cells(self):
        for row in self.cells:
            for cell in row:
                if cell.color == "White":
                    return True
        return False

    def add_to_color_history(self, color):
        if color not in self.color_history:
            self.color_history.append(color)

    def dummy_scan_lines(self):
        # Un-optimized version, works for small boards
        for scan_rows in all_sublists(range(self.size)):
            complete_colors = []
            for color in self.colors:
                if all(
                    [
                        cell.row in scan_rows
                        for cell in self.get_cells_color(color, only_free=True)
                    ]
                ):
                    complete_colors.append(color)

            if len(complete_colors) == len(scan_rows):
                for row in scan_rows:
                    for col in range(self.size):
                        if (
                            self.get_cell(row, col).color not in complete_colors
                            and not self.get_cell(row, col).busy
                        ):
                            self.get_cell(row, col).busy += 1

        for scan_cols in all_sublists(range(self.size)):
            complete_colors = []
            for color in self.colors:
                if all(
                    [
                        cell.col in scan_cols
                        for cell in self.get_cells_color(color, only_free=True)
                    ]
                ):
                    complete_colors.append(color)

            if len(complete_colors) == len(scan_cols):
                for col in scan_cols:
                    for row in range(self.size):
                        if (
                            self.get_cell(row, col).color not in complete_colors
                            and not self.get_cell(row, col).busy
                        ):
                            self.get_cell(row, col).busy += 1

    def scan_lines(self):
        row_scanned = []
        colors_scanned = []
        scan_color_rows = {}
        while True:
            scan_color_rows = {}
            for color in self.colors:
                for cell in self.get_cells_color(color, only_free=True):
                    if (
                        color in scan_color_rows
                        and cell.row not in scan_color_rows[color] + row_scanned
                    ):
                        scan_color_rows[color].append(cell.row)
                    elif color not in scan_color_rows and color not in colors_scanned:
                        scan_color_rows[color] = [cell.row]

            if check_available_scans(scan_color_rows) is None:
                break

            colors_scan, rows_scan = check_available_scans(scan_color_rows)

            if len(rows_scan) == 1:
                for col in range(self.size):
                    if (
                        self.get_cell(rows_scan[0], col).color != colors_scan
                        and not self.get_cell(rows_scan[0], col).busy
                    ):
                        self.get_cell(rows_scan[0], col).busy += 1
                row_scanned.append(rows_scan[0])
                colors_scanned.append(colors_scan)
                scan_color_rows.pop(colors_scan)
            else:
                for row in rows_scan:
                    row_scanned.append(row)
                    for col in range(self.size):
                        if (
                            self.get_cell(row, col).color not in colors_scan
                            and not self.get_cell(row, col).busy
                        ):
                            self.get_cell(row, col).busy += 1
                for color in colors_scan:
                    colors_scanned.append(color)
                    scan_color_rows.pop(color)

        col_scanned = []
        colors_scanned = []
        scan_color_cols = {}
        while True:
            scan_color_cols = {}
            for color in self.colors:
                for cell in self.get_cells_color(color, only_free=True):
                    if (
                        color in scan_color_cols
                        and cell.col not in scan_color_cols[color] + col_scanned
                    ):
                        scan_color_cols[color].append(cell.col)
                    elif color not in scan_color_cols and color not in colors_scanned:
                        scan_color_cols[color] = [cell.col]

            if check_available_scans(scan_color_cols) is None:
                break

            colors_scan, cols_scan = check_available_scans(scan_color_cols)

            if len(cols_scan) == 1:
                for row in range(self.size):
                    if (
                        self.get_cell(row, cols_scan[0]).color != colors_scan
                        and not self.get_cell(row, cols_scan[0]).busy
                    ):
                        self.get_cell(row, cols_scan[0]).busy += 1
                col_scanned.append(cols_scan[0])
                colors_scanned.append(colors_scan)
                scan_color_cols.pop(colors_scan)
            else:
                for col in cols_scan:
                    col_scanned.append(col)
                    for row in range(self.size):
                        if (
                            self.get_cell(row, col).color not in colors_scan
                            and not self.get_cell(row, col).busy
                        ):
                            self.get_cell(row, col).busy += 1
                for color in colors_scan:
                    colors_scanned.append(color)
                    scan_color_cols.pop(color)

    def scan_overlaps(self):
        for color in self.colors:
            cells_seen = {}
            for cell in self.get_cells_color(color, only_free=True):
                for adj_cell in self.get_adjacent_cells(
                    cell.row, cell.col, only_diagonals=True, only_free=True
                ):
                    if adj_cell.coords in cells_seen:
                        cells_seen[adj_cell.coords] += 1
                    else:
                        cells_seen[adj_cell.coords] = 1
                for row in range(self.size):
                    if (row, cell.col) in cells_seen:
                        cells_seen[(row, cell.col)] += 1
                    else:
                        cells_seen[(row, cell.col)] = 1
                for col in range(self.size):
                    if (cell.row, col) in cells_seen:
                        cells_seen[(cell.row, col)] += 1
                    else:
                        cells_seen[(cell.row, col)] = 1

            for cell in self.get_cells_color(color, only_free=True):
                cells_seen[cell.coords] = 0

            for coords, count in cells_seen.items():
                if count >= len(self.get_cells_color(color, only_free=True)):
                    self.get_cell(coords[0], coords[1]).busy += 1

    def solve(self, Ql=None, verbose=False):
        """
        Recursive backtracking algorithm to solve the board.
        """
        self.n_recursions += 1

        if verbose:
            self.print_board_cells()
            print(f"Excluded cells: {self.excluded_cells}")
            print(f"Color history: {self.color_history}")
            print(f"Last queen: {Ql}")

        if self.is_solved():
            self.solution = self.get_queens_coords()
            return self

        if self.is_solvable():
            if verbose:
                print("Try to add another queen")
            color = self.get_smallest_color()
            cell = self.get_cells_color(
                color, only_free=True, remove_excluded_cells=True
            )[0]
            if verbose:
                print(f"Color: {color}")
                print(f"Cell: {cell}")
            self.add_queen(cell.row, cell.col)
            self.add_to_color_history(cell.color)
            self.solve(self.get_queen(cell.row, cell.col), verbose=verbose)

        else:
            if verbose:
                print("Not solvable")
            if self.get_queen(Ql.row, Ql.col) is not None:
                self.excluded_cells.append(Ql.coords)
                self.remove_queen(Ql.row, Ql.col)

            if verbose:
                print(f"Excluded cells update: {self.excluded_cells}")

            if not self.is_color_solvable(Ql.color):
                if verbose:
                    print(f"No more free cells for color {Ql.color}")
                for case in self.get_cells_color(Ql.color):
                    if case.coords in self.excluded_cells:
                        self.excluded_cells.remove(case.coords)
                self.color_history.pop()
                Ql = self.get_color_queen(self.color_history[-1])
                self.excluded_cells.append(Ql.coords)
                self.remove_queen(Ql.row, Ql.col)
                self.solve(Ql, verbose=verbose)

            self.solve(None, verbose=verbose)

    def fSolve(self):
        """
        Much cleaner recursive backtracking algorithm to solve the board.
        """
        self.n_recursions += 1

        if self.is_solved():
            return self.get_queens_coords()

        color = self.get_smallest_color()

        if self.is_solvable():
            for cell in self.get_cells_color(
                color, only_free=True, remove_excluded_cells=True
            ):
                self.add_queen(cell.row, cell.col)

                if self.fSolve():
                    return True

                self.remove_queen(cell.row, cell.col)
        else:
            return False

    def print_board(self):
        for row in self.cells:
            for cell in row:
                print(cell)
            print()

    def print_board_colors(self):
        display = ""
        for row in self.cells:
            for cell in row:
                display += f"{cell.color[0]} "
            display += "\n"
        print(display)

    def print_board_cells(self):
        display = ""
        for row in self.cells:
            for cell in row:
                if cell.queen:
                    display += "Q "
                elif cell.coords in self.excluded_cells:
                    display += "E "
                elif not cell.busy:
                    display += "- "
                elif cell.busy:
                    display += "x "
            display += "\n"
        print(display)

    def print_queens(self):
        print("Queens:")
        for row in self.cells:
            for cell in row:
                if cell.queen:
                    print(f"\t{cell.queen}")

    def plot_board(self, output="board.png", with_cell_status=False):
        fig, ax = plt.subplots(figsize=(self.size, self.size))
        for row in self.cells:
            for cell in row:
                ax.add_patch(
                    patches.Rectangle(
                        (cell.col, cell.row),
                        1,
                        1,
                        facecolor=cell.color,
                        edgecolor="None",
                    )
                )
                ax.add_patch(
                    patches.Rectangle(
                        (cell.col, cell.row),
                        1,
                        1,
                        facecolor="None",
                        edgecolor="black",
                        alpha=0.1,
                    )
                )

                if cell.queen:
                    img = mpimg.imread(
                        Path(__file__).parent / ".." / "img" / "queen.png"
                    )
                    im = OffsetImage(img, zoom=0.2)
                    ab = AnnotationBbox(
                        im, (cell.col + 0.5, cell.row + 0.5), frameon=False, zorder=10
                    )
                    ax.add_artist(ab)

                elif with_cell_status:
                    if cell.busy:
                        ax.text(
                            cell.col + 0.5,
                            cell.row + 0.5,
                            f"{cell.busy}",
                            ha="center",
                            va="center",
                            color="black",
                            backgroundcolor="white",
                        )
                    if cell.coords in self.excluded_cells:
                        ax.text(
                            cell.col + 0.5,
                            cell.row + 0.5,
                            "Ex.",
                            ha="center",
                            va="center",
                            color="black",
                            backgroundcolor="white",
                        )

        # Plot hard lines between cells
        for r in range(self.size - 1):
            for c in range(self.size):
                if isinstance(self.get_cell(r, c).color, np.ndarray):
                    if not np.array_equal(
                        self.get_cell(r, c).color, self.get_cell(r + 1, c).color
                    ):
                        ax.plot([c, c + 1], [r, r], color="black", lw=1.2)
                else:
                    if self.get_cell(r, c).color != self.get_cell(r + 1, c).color:
                        ax.plot([c, c + 1], [r + 1, r + 1], color="black", lw=1.2)

        for c in range(self.size - 1):
            for r in range(self.size):
                if isinstance(self.get_cell(r, c).color, np.ndarray):
                    if not np.array_equal(
                        self.get_cell(r, c).color, self.get_cell(r, c + 1).color
                    ):
                        ax.plot([c, c], [r, r + 1], color="black", lw=1.2)
                else:
                    if self.get_cell(r, c).color != self.get_cell(r, c + 1).color:
                        ax.plot([c + 1, c + 1], [r, r + 1], color="black", lw=1.2)

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        # ax.set_aspect('equal')
        _ = ax.axis("off")

        fig.patch.set_facecolor("black")

        fig.gca().invert_yaxis()

        if with_cell_status:
            output = output.replace(".png", "_status.png")

        fig.savefig(output, bbox_inches="tight")
        print(f"Board saved as {output}")
