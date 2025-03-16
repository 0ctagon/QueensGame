import plothist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from pathlib import Path
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import pandas as pd
import yaml
import numpy as np
from queensgame.utils import hex_to_color_name


class DBoard:
    """
    Uses a pandas DataFrame to store the board state instead of a list of lists of Cell objects that was used in the Board class. Works better with higher board sizes.
    """

    def __init__(self, size):
        self.size = size
        self.cells = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [range(size), range(size)], names=["row", "col"]
            )
        )
        # self.cells["coords"] = self.cells.index
        self.cells["color"] = "White"
        self.cells["busy"] = 0
        self.cells["excluded"] = False
        self.cells["queen"] = False

        self.colors = []
        self.seed = None

        # Solving attributes
        self.color_history = []
        self.solution = []
        self.n_recursions = 0

    def __str__(self):
        return f"Board size {self.size}, colors: {self.colors}"

    def __copy__(self):
        new_board = DBoard(self.size)
        new_board.colors = self.colors.copy()
        new_board.cells = self.cells.copy()
        new_board.seed = self.seed

    def read_board(self, yaml_file):
        print(f"Reading board from {yaml_file}")

        with open(yaml_file, "r") as file:
            board_state = yaml.safe_load(file)

        for row in range(self.size):
            for col in range(self.size):
                color = hex_to_color_name(board_state[f"({row}, {col})"])
                self.cells.at[(row, col), "color"] = color

        self.colors = self.cells["color"].unique().tolist()

    def generate_random_board(
        self, seed=None, palette="gist_ncar", no_single_cell=True, verbose=True
    ):
        if self.size < 4:
            raise ValueError("Board size must be at least 4")

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
                busy_cells = (
                    self.cells[self.cells["busy"] > 0]
                    .query(f"row == {row}")
                    .index.get_level_values("col")
                    .tolist()
                )
                try:
                    col_rng = rng.choice(
                        [i for i in range(0, self.size) if i not in busy_cells]
                    )
                    self.cells.at[(row, col_rng), "color"] = colors[row]
                    self.add_queen(row, int(col_rng))
                except ValueError:
                    for coords in self.get_queens_coords():
                        self.remove_queen(coords[0], coords[1])
                        self.cells.at[coords, "color"] = "White"

        while self.has_colorless_cells():
            for cell in self.cells[self.cells["color"] == "White"].index:
                adjacent_colors = ["White"]
                for adj_cell in self.get_adj_cells_coords(cell[0], cell[1]):
                    if self.cells.at[adj_cell, "color"] != "White":
                        adjacent_colors.append(self.cells.at[adj_cell, "color"])
                self.cells.at[cell, "color"] = adjacent_colors[
                    rng.integers(0, len(adjacent_colors))
                ]

        for coords in self.get_queens_coords():
            self.remove_queen(coords[0], coords[1])

        if no_single_cell:
            while self.get_smallest_color(value=True) == 1:
                color = self.get_smallest_color()
                cell = self.get_cells_color(color).iloc[0]
                for adj_cell in self.get_adj_cells_coords(
                    cell.name[0], cell.name[1], only_free=True
                ):
                    if rng.random() > 0.5:
                        self.cells.at[adj_cell, "color"] = color

        if verbose:
            print(f"Board generated with seed {self.seed}")

    def get_adj_cells_coords(
        self, row, col, with_diagonals=False, only_diagonals=False, only_free=False
    ):
        adjacent_cells = []
        if only_diagonals:
            with_diagonals = True
        if row > 0:
            if not only_diagonals:
                adjacent_cells.append((row - 1, col))
            if col > 0 and with_diagonals:
                adjacent_cells.append((row - 1, col - 1))
            if col < self.size - 1 and with_diagonals:
                adjacent_cells.append((row - 1, col + 1))
        if row < self.size - 1:
            if not only_diagonals:
                adjacent_cells.append((row + 1, col))
            if col > 0 and with_diagonals:
                adjacent_cells.append((row + 1, col - 1))
            if col < self.size - 1 and with_diagonals:
                adjacent_cells.append((row + 1, col + 1))
        if col > 0 and not only_diagonals:
            adjacent_cells.append((row, col - 1))
        if col < self.size - 1 and not only_diagonals:
            adjacent_cells.append((row, col + 1))
        if only_free:
            return [
                coords
                for coords in adjacent_cells
                if self.cells.at[coords, "busy"] == 0
            ]
        return adjacent_cells

    def get_cells_color(self, color, only_free=False, remove_excluded_cells=False):
        cells = self.cells[self.cells["color"] == color]
        if remove_excluded_cells:
            cells = cells[~cells["excluded"]]
        if only_free:
            cells = cells[cells["busy"] == 0]
        return cells

    def add_queen(self, row, col):
        if self.cells.at[(row, col), "busy"] == 0:
            self.cells.at[(row, col), "busy"] += 1
            self.cells.at[(row, col), "queen"] = True
            for corner in self.get_adj_cells_coords(row, col, only_diagonals=True):
                self.cells.at[corner, "busy"] += 1
            for r in range(self.size):
                self.cells.at[(r, col), "busy"] += 1
            for c in range(self.size):
                self.cells.at[(row, c), "busy"] += 1
            for cell in self.get_cells_color(self.cells.at[(row, col), "color"]).index:
                self.cells.at[cell, "busy"] += 1
        else:
            raise ValueError(f"Cell ({row}, {col}) is not free")

    def decrease_busy(self, row, col):
        if self.cells.at[(row, col), "busy"] > 0:
            self.cells.at[(row, col), "busy"] -= 1

    def remove_queen(self, row, col):
        if self.cells.at[(row, col), "queen"]:
            self.cells.at[(row, col), "queen"] = False
            self.decrease_busy(row, col)
            for corner in self.get_adj_cells_coords(row, col, only_diagonals=True):
                self.decrease_busy(corner[0], corner[1])
            for r in range(self.size):
                self.decrease_busy(r, col)
            for c in range(self.size):
                self.decrease_busy(row, c)
            for cell in self.get_cells_color(self.cells.at[(row, col), "color"]).index:
                self.decrease_busy(cell[0], cell[1])
        else:
            raise ValueError(f"Cell ({row}, {col}) has no queen")

    def count_queens(self):
        return self.cells["queen"].sum()

    def count_free_cells(self):
        return self.cells[self.cells["busy"] == 0].shape[0]

    def count_free_per_color(self, order=None, remove_empty=False):
        free_per_color = (
            self.cells[(self.cells["busy"] == 0) & (~self.cells["excluded"])]
            .groupby("color")
            .size()
        )

        if remove_empty:
            free_per_color = free_per_color[free_per_color > 0]
        if order == "asc":
            return free_per_color.sort_values()
        elif order == "desc":
            return free_per_color.sort_values(ascending=False)
        return free_per_color

    def get_smallest_color(self, value=False):
        free_per_color = self.count_free_per_color(order="asc", remove_empty=True)
        if free_per_color.empty:
            return None
        if value:
            return free_per_color.iloc[0]
        return free_per_color.index[0]

    def get_queens_coords(self):
        return self.cells[self.cells["queen"]].index.to_list()

    def get_color_queen(self, color):
        return self.cells[(self.cells["color"] == color) & self.cells["queen"]]

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
        if self.cells[self.cells["color"] == "White"].shape[0] > 0:
            return True
        return False

    def add_to_color_history(self, color):
        if color not in self.color_history:
            self.color_history.append(color)

    # TODO: To be converted
    # def dummy_scan_lines(self):
    #     # Un-optimized version, works for small boards
    #     for scan_rows in all_sublists(range(self.size)):
    #         complete_colors = []
    #         for color in self.colors:
    #             if all(
    #                 [cell.row in scan_rows for cell in self.get_cells_color(color, only_free=True)]
    #             ):
    #                 complete_colors.append(color)

    #         if len(complete_colors) == len(scan_rows):
    #             for row in scan_rows:
    #                 for col in range(self.size):
    #                     if self.get_cell(row, col).color not in complete_colors and not self.get_cell(row, col).busy:
    #                         self.get_cell(row, col).busy += 1

    #     for scan_cols in all_sublists(range(self.size)):
    #         complete_colors = []
    #         for color in self.colors:
    #             if all(
    #                 [cell.col in scan_cols for cell in self.get_cells_color(color, only_free=True)]
    #             ):
    #                 complete_colors.append(color)

    #         if len(complete_colors) == len(scan_cols):
    #             for col in scan_cols:
    #                 for row in range(self.size):
    #                     if self.get_cell(row, col).color not in complete_colors and not self.get_cell(row, col).busy:
    #                         self.get_cell(row, col).busy += 1

    # def scan_lines(self):
    #     row_scanned = []
    #     colors_scanned = []
    #     scan_color_rows = {}
    #     while True:
    #         scan_color_rows = {}
    #         for color in self.colors:
    #             for cell in self.get_cells_color(color, only_free=True):
    #                 if color in scan_color_rows and cell.row not in scan_color_rows[color]+row_scanned:
    #                     scan_color_rows[color].append(cell.row)
    #                 elif color not in scan_color_rows and color not in colors_scanned:
    #                     scan_color_rows[color] = [cell.row]

    #         if check_available_scans(scan_color_rows) is None:
    #             break

    #         colors_scan, rows_scan = check_available_scans(scan_color_rows)

    #         if len(rows_scan) == 1:
    #             for col in range(self.size):
    #                 if self.get_cell(rows_scan[0], col).color != colors_scan and not self.get_cell(rows_scan[0], col).busy:
    #                     self.get_cell(rows_scan[0], col).busy += 1
    #             row_scanned.append(rows_scan[0])
    #             colors_scanned.append(colors_scan)
    #             scan_color_rows.pop(colors_scan)
    #         else:
    #             for row in rows_scan:
    #                 row_scanned.append(row)
    #                 for col in range(self.size):
    #                     if self.get_cell(row, col).color not in colors_scan and not self.get_cell(row, col).busy:
    #                         self.get_cell(row, col).busy += 1
    #             for color in colors_scan:
    #                 colors_scanned.append(color)
    #                 scan_color_rows.pop(color)

    #     col_scanned = []
    #     colors_scanned = []
    #     scan_color_cols = {}
    #     while True:
    #         scan_color_cols = {}
    #         for color in self.colors:
    #             for cell in self.get_cells_color(color, only_free=True):
    #                 if color in scan_color_cols and cell.col not in scan_color_cols[color]+col_scanned:
    #                     scan_color_cols[color].append(cell.col)
    #                 elif color not in scan_color_cols and color not in colors_scanned:
    #                     scan_color_cols[color] = [cell.col]

    #         if check_available_scans(scan_color_cols) is None:
    #             break

    #         colors_scan, cols_scan = check_available_scans(scan_color_cols)

    #         if len(cols_scan) == 1:
    #             for row in range(self.size):
    #                 if self.get_cell(row, cols_scan[0]).color != colors_scan and not self.get_cell(row, cols_scan[0]).busy:
    #                     self.get_cell(row, cols_scan[0]).busy += 1
    #             col_scanned.append(cols_scan[0])
    #             colors_scanned.append(colors_scan)
    #             scan_color_cols.pop(colors_scan)
    #         else:
    #             for col in cols_scan:
    #                 col_scanned.append(col)
    #                 for row in range(self.size):
    #                     if self.get_cell(row, col).color not in colors_scan and not self.get_cell(row, col).busy:
    #                         self.get_cell(row, col).busy += 1
    #             for color in colors_scan:
    #                 colors_scanned.append(color)
    #                 scan_color_cols.pop(color)

    # def scan_overlaps(self):
    #     for color in self.colors:
    #         cells_seen = {}
    #         for cell in self.get_cells_color(color, only_free=True):
    #             for adj_cell in self.get_adjacent_cells(cell.row, cell.col, only_diagonals=True, only_free=True):
    #                 if adj_cell.coords in cells_seen:
    #                     cells_seen[adj_cell.coords] += 1
    #                 else:
    #                     cells_seen[adj_cell.coords] = 1
    #             for row in range(self.size):
    #                 if (row, cell.col) in cells_seen:
    #                     cells_seen[(row, cell.col)] += 1
    #                 else:
    #                     cells_seen[(row, cell.col)] = 1
    #             for col in range(self.size):
    #                 if (cell.row, col) in cells_seen:
    #                     cells_seen[(cell.row, col)] += 1
    #                 else:
    #                     cells_seen[(cell.row, col)] = 1

    #         for cell in self.get_cells_color(color, only_free=True):
    #             cells_seen[cell.coords] = 0

    #         for coords, count in cells_seen.items():
    #             if count >= len(self.get_cells_color(color, only_free=True)):
    #                 self.get_cell(coords[0], coords[1]).busy += 1

    def solve(self, lQcoords=None, verbose=False):
        """
        Recursive backtracking algorithm to solve the board.
        """
        self.n_recursions += 1

        if verbose:
            print(f"Excluded cells: {self.cells[self.cells['excluded']]}")
            print(f"Color history: {self.color_history}")
            print(f"Last cell: {lQcoords}")

        if self.is_solved():
            return

        if self.is_solvable():
            if verbose:
                print("Try to add another queen")
            color = self.get_smallest_color()
            cell = self.get_cells_color(
                color, only_free=True, remove_excluded_cells=True
            ).iloc[0]
            if verbose:
                print(f"Color: {color}")
                print(f"Cell: {cell}")
            self.add_queen(*cell.name)
            self.add_to_color_history(cell.color)
            self.solve(cell.name, verbose=verbose)

        else:
            if verbose:
                print("Not solvable")
            if self.cells.at[lQcoords, "queen"]:
                self.cells.loc[lQcoords, "excluded"] = True
                self.remove_queen(*lQcoords)

            if verbose:
                print(f"Excluded cells update: {self.cells[self.cells['excluded']]}")

            if not self.is_color_solvable(self.cells.at[lQcoords, "color"]):
                if verbose:
                    print(
                        f"No more free cells for color {self.cells.at[lQcoords, 'color']}"
                    )
                color = self.cells.at[lQcoords, "color"]
                self.cells.loc[self.cells["color"] == color, "excluded"] = False
                self.color_history.pop()
                lCell = self.get_color_queen(self.color_history[-1])
                self.cells.loc[lCell.index, "excluded"] = True
                self.remove_queen(*lCell.index[0])
                self.solve(lCell.index[0], verbose=verbose)

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
            ).index:
                self.add_queen(*cell)

                if self.fSolve():
                    return True

                self.remove_queen(*cell)
        else:
            return False

    def print_queens(self):
        print("Queens:")
        print(self.cells[self.cells["queen"] == True])

    def plot_board(self, output="board.png", with_cell_status=False):
        fig, ax = plt.subplots(figsize=(self.size, self.size))
        for index, cell in self.cells.iterrows():
            row, col = index
            ax.add_patch(
                patches.Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=cell.color,
                    edgecolor="None",
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor="None",
                    edgecolor="black",
                    alpha=0.1,
                )
            )

            if cell.queen:
                img = mpimg.imread(Path(__file__).parent / ".." / "img" / "queen.png")
                im = OffsetImage(img, zoom=0.2)
                ab = AnnotationBbox(
                    im, (col + 0.5, row + 0.5), frameon=False, zorder=10
                )
                ax.add_artist(ab)

            elif with_cell_status:
                if cell.busy:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        f"{cell.busy}",
                        ha="center",
                        va="center",
                        color="black",
                        backgroundcolor="white",
                    )
                if cell.excluded:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        "Ex.",
                        ha="center",
                        va="center",
                        color="black",
                        backgroundcolor="white",
                    )

        # Plot hard lines between cells
        for r in range(self.size - 1):
            for c in range(self.size):
                if not np.array_equal(
                    self.cells.at[(r, c), "color"], self.cells.at[(r + 1, c), "color"]
                ):
                    ax.plot([c, c + 1], [r + 1, r + 1], color="black", lw=1.2)

        for c in range(self.size - 1):
            for r in range(self.size):
                if not np.array_equal(
                    self.cells.at[(r, c), "color"], self.cells.at[(r, c + 1), "color"]
                ):
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
