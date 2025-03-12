from classes import *

size = 11

# gameID = 'board1'
# gameID = '250309x7'
# gameID = '250310x7'
# gameID = "250309x11"
# gameID = '250402x11'
gameID = "250310x11"

output = f"plots/{gameID}.png"

board = Board(size)
board.read_board(f"examples/{gameID}.yaml")

# board.add_queen(0, 0)

board.plot_board(output=output)
board.print_board_colors()
board.solve()
# board.solve(verbose=True)

# board.print_board_cells()
board.print_queens()

output = output.replace(".png", "_solved.png")

board.plot_board(output=output)
board.plot_board(output=output, with_cell_status=True)
