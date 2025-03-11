import pygame
import sys
import yaml
import argparse

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 700, 700
ROWS, COLS = 11,11
SQUARE_SIZE = WIDTH // COLS
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 50
BUTTON_X, BUTTON_Y = WIDTH + 20, (HEIGHT - BUTTON_HEIGHT) // 5
COLOR_SQUARE_SIZE = 30
COLOR_SQUARE_X = BUTTON_X
COLOR_SQUARE_Y = BUTTON_Y + BUTTON_HEIGHT + 20
INDICATOR_X, INDICATOR_Y = BUTTON_X, BUTTON_Y - BUTTON_HEIGHT
INDICATOR_WIDTH, INDICATOR_HEIGHT = BUTTON_WIDTH, 30
SAVE_BUTTON_X, SAVE_BUTTON_Y = BUTTON_X, BUTTON_Y - BUTTON_HEIGHT*2 -20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
VIOLET = (238, 130, 238)
BROWN = (165, 42, 42)
GREY = (128, 128, 128)
BEIGE = (245, 245, 220)
PINK = (255, 192, 203)

# Create the screen
screen = pygame.display.set_mode((WIDTH + BUTTON_WIDTH + 40, HEIGHT))
pygame.display.set_caption(f"Create a {ROWS}x{COLS} board")

# Function to draw the board
def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(screen, WHITE, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)

# Function to draw the reset button
def draw_button():
    pygame.draw.rect(screen, GRAY, (BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT))
    font = pygame.font.Font(None, 36)
    text = font.render("Reset", True, BLACK)
    text_rect = text.get_rect(center=(BUTTON_X + BUTTON_WIDTH // 2, BUTTON_Y + BUTTON_HEIGHT // 2))
    screen.blit(text, text_rect)

# Function to draw the color selection squares
def draw_color_squares(colors):
    for i, color in enumerate(colors):
        pygame.draw.rect(screen, color, (COLOR_SQUARE_X, COLOR_SQUARE_Y + i * (COLOR_SQUARE_SIZE + 10), COLOR_SQUARE_SIZE, COLOR_SQUARE_SIZE))
        pygame.draw.rect(screen, BLACK, (COLOR_SQUARE_X, COLOR_SQUARE_Y + i * (COLOR_SQUARE_SIZE + 10), COLOR_SQUARE_SIZE, COLOR_SQUARE_SIZE), 1)

def draw_color_indicator(selected_color):
    pygame.draw.rect(screen, selected_color, (INDICATOR_X, INDICATOR_Y, INDICATOR_WIDTH, INDICATOR_HEIGHT))
    pygame.draw.rect(screen, BLACK, (INDICATOR_X, INDICATOR_Y, INDICATOR_WIDTH, INDICATOR_HEIGHT), 1)

def draw_save_button():
    pygame.draw.rect(screen, GRAY, (SAVE_BUTTON_X, SAVE_BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT))
    font = pygame.font.Font(None, 36)
    text = font.render("Save", True, BLACK)
    text_rect = text.get_rect(center=(SAVE_BUTTON_X + BUTTON_WIDTH // 2, SAVE_BUTTON_Y + BUTTON_HEIGHT // 2))
    screen.blit(text, text_rect)


def save_board_state(output='board_state.yaml'):
    board_state = {}
    for row in range(ROWS):
        for col in range(COLS):
            color = screen.get_at((col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2))
            hex_color = '#{:02x}{:02x}{:02x}'.format(color.r, color.g, color.b)
            board_state[f"({row}, {col})"] = hex_color

    if not output.endswith('.yaml'):
        output += '.yaml'

    with open(output, 'w') as file:
        yaml.dump(board_state, file)

# Main loop
def main(output):
    running = True
    colors = [RED, GREEN, BLUE, YELLOW, CYAN, VIOLET, ORANGE, BROWN, GREY, BEIGE, PINK]
    selected_color = colors[0]

    draw_board()
    draw_button()
    draw_color_squares(colors)
    draw_save_button()
    draw_color_indicator(selected_color)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if BUTTON_X <= x <= BUTTON_X + BUTTON_WIDTH and BUTTON_Y <= y <= BUTTON_Y + BUTTON_HEIGHT:
                    draw_board()
                elif SAVE_BUTTON_X <= x <= SAVE_BUTTON_X + BUTTON_WIDTH and SAVE_BUTTON_Y <= y <= SAVE_BUTTON_Y + BUTTON_HEIGHT:
                    save_board_state(output)
                else:
                    for i, color in enumerate(colors):
                        if COLOR_SQUARE_X <= x <= COLOR_SQUARE_X + COLOR_SQUARE_SIZE and COLOR_SQUARE_Y + i * (COLOR_SQUARE_SIZE + 10) <= y <= COLOR_SQUARE_Y + i * (COLOR_SQUARE_SIZE + 10) + COLOR_SQUARE_SIZE:
                            selected_color = color
                            draw_color_indicator(selected_color)
                            break
                    row = y // SQUARE_SIZE
                    col = x // SQUARE_SIZE
                    if row < ROWS and col < COLS:
                        pygame.draw.rect(screen, selected_color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                        pygame.draw.rect(screen, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)

        pygame.display.flip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a board')
    parser.add_argument('--output', type=str, default='board_state.yaml', help='Output file')
    args = parser.parse_args()
    main(output=args.output)