import pygame
import threading


def show_tips(screen, font, text, x, y):
    text = font.render(text, True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.centerx = x
    text_rect.centery = y
    screen.blit(text, text_rect)


def move_to_location(move):
    h = move // 8
    w = move % 8
    return [h, w]


def dictionary_to_list(states):
    state = [[0 for x in range(8)] for y in range(8)]
    for move in states.keys():
        i = move_to_location(move)
        state[i[0]][i[1]] = states[move]
    return state


def update_screen(screen, font, states, player, moves):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    screen.fill(WHITE)
    for i in range(10):
        pygame.draw.line(screen, BLACK, [80, 80 + i * 80], [720 - 80, 80 + i * 80], 1)
        pygame.draw.line(screen, BLACK, [80 + i * 80, 80], [80 + i * 80, 720 - 80], 1)
    state = dictionary_to_list(states)
    for i in range(8):
        for j in range(8):
            if state[j][i] == 1:
                pygame.draw.circle(screen, (255, 255, 255), [80 + i * 80, 80 + j * 80], 30, 0)
                pygame.draw.circle(screen, (0, 0, 0), [80 + i * 80, 80 + j * 80], 30, 1)
            elif state[j][i] == 2:
                pygame.draw.circle(screen, (0, 0, 0), [80 + i * 80, 80 + j * 80], 30, 0)
    if moves is not None:
        move = move_to_location(moves)
        rect = pygame.Rect(80 + move[1] * 80 - 32, 80 + move[0] * 80 - 32, 64, 64)
        pygame.draw.rect(screen, (0, 0, 255), rect, 1, 1)

    show_tips(screen, font, f"the last move is made by {player}", 360, 40)
    show_tips(screen, font, "MCT-pure is the white pieces", 230, 675)
    show_tips(screen, font, "MCT1 is the black pieces", 208, 700)

    # 刷新屏幕
    pygame.display.flip()


class UpdateScreenThread(threading.Thread):
    def __init__(self, screen, font, state, player, move=None):
        threading.Thread.__init__(self)
        self.screen = screen
        self.font = font
        self.state = state
        self.player = player
        self.move = move

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            update_screen(self.screen, self.font, self.state, self.player, self.move)


if __name__ == '__main__':
    exit(0)
