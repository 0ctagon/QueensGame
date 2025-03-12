from classes import *
import sys

sys.setrecursionlimit(10**6)

size = 30

board = Board(size)

board.generate_random_board()

board.plot_board(output=f"plots/{board.seed}x{size}.png")

board.solve()

board.plot_board(output=f"plots/{board.seed}x{size}_solved.png")
board.plot_board(output=f"plots/{board.seed}x{size}_solved.png", with_cell_status=True)
