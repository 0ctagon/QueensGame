class Cell:
    def __init__(self, row, col, color="White"):
        self.row = row
        self.col = col
        self.color = color
        self.queen = None
        self.busy = 0
        self.coords = (row, col)

    def __str__(self):
        return (
            f"Cell ({self.row}, {self.col})"
            + (" | F " if not self.busy else " | X ")
            + f"| {self.color}"
            + (f" - QUEEN" if self.queen else "")
        )

    def __repr__(self):
        return f"({self.row}, {self.col})"

    def __copy__(self):
        new_cell = Cell(self.row, self.col, self.color)
        new_cell.queen = self.queen
        new_cell.busy = self.busy
        return new_cell

    def add_queen(self, queen):
        if self.busy:
            raise ValueError(f"Cell ({self.row}, {self.col}) is not free")

        self.queen = queen
        self.busy += 1

    def remove_queen(self):
        if self.queen is None:
            raise ValueError(f"Cell ({self.row}, {self.col}) has no queen")
        self.queen = None
        self.reduce_busy()

    def reduce_busy(self):
        if self.busy > 0:
            self.busy -= 1
