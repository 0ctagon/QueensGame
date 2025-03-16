from queensgame import Board
from pathlib import Path
import os

size = 11

gameIDs = ["board1x7", "250309x7", "250310x7", "250309x11", "250402x11", "250310x11"]

for gameID in gameIDs:
    size = int(gameID.split("x")[-1])

    board = Board(size)
    board.read_board(f"{Path(__file__).parent / "data" / gameID}.yaml")

    board.solve()
    board.print_queens()

    os.makedirs(Path(__file__).parent / "plots", exist_ok=True)
    board.plot_board(output=f"{Path(__file__).parent / "plots" / f"{gameID}.png"}")
