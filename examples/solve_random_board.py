from queensgame import *

size = 100

board = DBoard(size)
board.generate_random_board()

board.plot_board(output=f"{board.seed}x{size}.png")

board.fSolve()

board.plot_board(output=f"{board.seed}x{size}_solved.png")
board.plot_board(output=f"{board.seed}x{size}_solved.png", with_cell_status=True)
