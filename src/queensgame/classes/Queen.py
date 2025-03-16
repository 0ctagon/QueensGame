class Queen:
    def __init__(self, row, col, _color=None):
        self.row = row
        self.col = col
        self.coords = (row, col)
        self.color = _color

    def __str__(self):
        return f"Queen on {self.coords}" + (f" | {self.color}" if self.color else "")
