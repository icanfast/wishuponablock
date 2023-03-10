import pygame
import random
import json
import datetime
import getpass
import os
import copy

from assets import shapes, controls

pygame.font.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700
PLAY_WIDTH = 300
PLAY_HEIGHT = 600
BLOCK_SIZE = 30
LOCK_DELAY = 40  # miliseconds
PLAY_COLOUR = (30, 30, 30)
BACKGROUND_COLOUR = (20, 20, 20)
GRIDLINES_COLOUR = (64, 64, 64)
TOP_LEFT_X = (WINDOW_WIDTH - PLAY_WIDTH) // 2
TOP_LEFT_Y = WINDOW_HEIGHT - PLAY_HEIGHT - 10
MATRIX_WIDTH = 10
MATRIX_HEIGHT = 30
MATRIX_ACTIVE_HEIGHT = 20
PIECE_SPAWNING_X = 5
PIECE_SPAWNING_Y = 17
RECORD = True


def write_snapshot(snapshot, snapshot_path, turn):
    file_path = os.path.join(snapshot_path, 'turn_'+str(turn)+'.json')
    with open(file_path, 'w') as f:
        json.dump(snapshot, f)


class Piece(object):  # *

    def __init__(self, x, y, piece_name, colour=None, rotation=0):
        self.x = x
        self.y = y
        self.piece_name = piece_name
        self.shape = shapes[piece_name]['rotations']
        if colour:
            self.colour = colour
        else:
            self.colour = shapes[piece_name]['colour']
        self.rotation = rotation
        self.image = shape_to_colours(self.shape, self.colour)

    def __str__(self) -> str:
        description = 'X: ' + str(self.x)
        description += '\n'
        description += 'Y: ' + str(self.y)
        description += '\n'
        description += '\n'.join(self.shape[self.rotation % 4])
        return description

    def __copy__(self):
        return Piece(x=self.x,
                     y=self.y,
                     shape=self.shape,
                     colour=self.colour,
                     rotation=self.rotation)

    def shape_to_colours(shape, colour):
        image = []
        for rotation in shape:
            image_rotation = [
                [colour if x == '0' else PLAY_COLOUR for x in y] for y in rotation]
            image.append(image_rotation)
        return image


class Bag(object):

    def __init__(self, seed=42):  # seed functionality not implemented yet
        self.all_keys = ['S', 'Z', 'I', 'O', 'J', 'L', 'T']
        self.current_bag = copy.deepcopy(self.all_keys)
        self.next_bag = copy.deepcopy(self.all_keys)
        random.shuffle(self.current_bag)
        random.shuffle(self.next_bag)
        self.turn_no = 1

    # matrix parameter for TBI conditional spawning
    def __next__(self, verbose=True, matrix=None):
        key = self.current_bag.pop()
        if not self.current_bag:
            self.refill_bag()
        if verbose:
            print('Turn', self.turn_no)
            print('Current Piece: ', key)
            print('Bag: ', self.current_bag)
            if not self.current_bag:
                print('Refilling Bag')
            print()
        self.turn_no += 1
        return key

    def refill_bag(self):
        self.current_bag = copy.deepcopy(self.next_bag)
        self.next_bag = copy.deepcopy(self.all_keys)
        random.shuffle(self.next_bag)


def create_matrix():
    matrix = [[PLAY_COLOUR for i in range(MATRIX_WIDTH)]
              for j in range(MATRIX_HEIGHT)]
    return matrix


def valid_space(piece, matrix):
    for i, row in enumerate(piece.image[piece.rotation % 4]):
        for j, cell in enumerate(row):
            if cell != PLAY_COLOUR:
                # print('X: ', piece.x)
                # print('y: ', piece.y)
                # print('i: ', i)
                # print('j: ', j)
                # print()
                if i + piece.y > 29:
                    return False
                if matrix[i + piece.y][j+piece.x] != PLAY_COLOUR:
                    return False
    return True


def srs_kicks(matrix, piece, key):
    pass


def get_next_piece(retriever):
    return Piece(PIECE_SPAWNING_X, PIECE_SPAWNING_Y, next(retriever))


def clear_rows(matrix, verbose=True):
    rows_to_clear = []
    for i, row in enumerate(matrix):
        for item in row:
            if item == PLAY_COLOUR:
                break
        else:
            rows_to_clear.append(i)

    new_matrix = [[PLAY_COLOUR]*MATRIX_WIDTH]*len(rows_to_clear)
    for i, row in enumerate(matrix):
        if i not in rows_to_clear:
            new_matrix.append(row)

    if verbose:
        print('Matrix Height: ', len(matrix))
        rectangle = True
        for row in matrix:
            if len(row) != MATRIX_WIDTH:
                rectangle = False
        print('Matrix rectangular: ', rectangle)

    score = scoring(rows_to_clear)

    return new_matrix, score


def scoring(cleared_rows, modifier=None, combo=None, b2b=None):
    pass


def check_lost(current_piece):
    if current_piece.y < 10:
        return True
    return False


def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("sfnsmono", size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH / 2 - (label.get_width()/2),
                 TOP_LEFT_Y + PLAY_HEIGHT/2 - label.get_height()/2))


def draw_window(surface, matrix, current_piece, score=0):
    surface.fill(BACKGROUND_COLOUR)
    pygame.font.init()
    font = pygame.font.SysFont('sfnsmono', 60)
    label = font.render('Tetris', 1, (255, 255, 255))
    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH /
                 2 - (label.get_width() / 2), 30))

    # current score
    font = pygame.font.SysFont('sfnsmono', 30)
    label = font.render('Score: ' + str(score), 1, (255, 255, 255))

    sx = TOP_LEFT_X + PLAY_WIDTH + 50
    sy = TOP_LEFT_Y + PLAY_HEIGHT/2 - 100

    surface.blit(label, (sx, sy + 350))

    draw_matrix(surface, matrix)
    draw_piece(surface, current_piece)
    draw_gridlines(surface)

    pygame.draw.rect(surface, (64, 64, 64), (TOP_LEFT_X-2,
                     TOP_LEFT_Y-2, PLAY_WIDTH+4, PLAY_HEIGHT+4), 4)


def draw_matrix(surface, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            pygame.draw.rect(
                surface,
                matrix[i][j],
                (
                    TOP_LEFT_X + j*BLOCK_SIZE,
                    TOP_LEFT_Y + i*BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE
                ),
                0
            )


def draw_piece(surface, piece):
    for i, row in enumerate(piece.image[piece.rotation % 4]):
        for j in range(len(row)):
            pygame.draw.rect(
                surface,
                piece.image[piece.rotation % 4][i][j],
                (
                    TOP_LEFT_X + j*BLOCK_SIZE + piece.x*BLOCK_SIZE,
                    TOP_LEFT_Y + i*BLOCK_SIZE + (piece.y-10)*BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE
                ),
                0
            )


def draw_gridlines(surface):
    for i in range(MATRIX_ACTIVE_HEIGHT):
        pygame.draw.line(
            surface,
            GRIDLINES_COLOUR,
            (
                TOP_LEFT_X,
                TOP_LEFT_Y + i * BLOCK_SIZE
            ),
            (
                TOP_LEFT_X + PLAY_WIDTH,
                TOP_LEFT_Y + i * BLOCK_SIZE
            )
        )
    for j in range(MATRIX_WIDTH):
        pygame.draw.line(
            surface,
            GRIDLINES_COLOUR,
            (
                TOP_LEFT_X + j * BLOCK_SIZE,
                TOP_LEFT_Y
            ),
            (
                TOP_LEFT_X + j * BLOCK_SIZE,
                TOP_LEFT_Y + PLAY_HEIGHT
            )
        )


def draw_next_piece(surface, piece):
    font = pygame.font.SysFont('sfnsmono', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = TOP_LEFT_X + PLAY_WIDTH + 70
    sy = TOP_LEFT_Y + PLAY_HEIGHT/2 - 250

    for i, row in enumerate(piece.image[piece.rotation % 4]):
        for j, column in enumerate(row):
            if column != PLAY_COLOUR:
                pygame.draw.rect(
                    surface,
                    piece.colour,
                    (
                        sx + j*BLOCK_SIZE,
                        sy + i*BLOCK_SIZE,
                        BLOCK_SIZE,
                        BLOCK_SIZE
                    ),
                    0
                )

    surface.blit(label, (sx - 30, sy - 50))


def draw_ghost(surface, matrix, piece):
    pass


def main(win):
    matrix = create_matrix()
    change_piece = False
    run = True
    clock = pygame.time.Clock()
    fall_speed = 0.3
    fall_time = 0
    score = 0
    turn = 1
    bag = Bag()
    next_piece = get_next_piece(bag)
    current_piece = next_piece
    next_piece = get_next_piece(bag)
    if RECORD:
        snapshot_path = os.path.join(
            './snapshots',
            getpass.getuser() + '_snapshots',
            datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        )
        try:
            os.makedirs(snapshot_path)
        except Exception as e:
            print(e)

    while run:
        fall_time += clock.get_rawtime()
        clock.tick()
        # current_piece = next_piece
        # next_piece = get_next_piece(bag)
        lost = False

        if fall_time/1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not(valid_space(current_piece, matrix)) and current_piece.y < 10:
                current_piece.y -= 1
                print('not valid space')
                change_piece = True
                # lock_delay += clock.get_rawtime()
                # if lock_delay >= LOCK_DELAY:
                #     lock_delay = 0
                #     change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == controls['Left']:
                    current_piece.x -= 1
                    if not(valid_space(current_piece, matrix)):
                        current_piece.x += 1
                if event.key == controls['Right']:
                    current_piece.x += 1
                    if not(valid_space(current_piece, matrix)):
                        current_piece.x -= 1
                if event.key == controls['Down']:
                    current_piece.y += 1
                    if not(valid_space(current_piece, matrix)):
                        current_piece.y -= 1
                    else:
                        fall_time = 0

                if event.key == controls['Rotate Clockwise'] or event.key == controls['Rotate']:
                    current_piece.rotation += 1
                    if not(valid_space(current_piece, matrix)):
                        current_piece.rotation -= 1
                if event.key == controls['Rotate Counterclockwise']:
                    current_piece.rotation -= 1
                    if not(valid_space(current_piece, matrix)):
                        current_piece.rotation += 1
                if event.key == controls['Rotate 180']:
                    current_piece.rotation += 2
                    if not(valid_space(current_piece, matrix)):
                        current_piece.rotation += len(current_piece.shape)//2
                if event.key == controls['Hard Drop']:
                    fall_speed = 0.00001

        if change_piece:
            snapshot = {
                'matrix': matrix
            }
            if RECORD:
                write_snapshot(snapshot=snapshot,
                               snapshot_path=snapshot_path, turn=turn)
            for i, row in enumerate(current_piece.image[current_piece.rotation % 4]):
                for j, cell_ in enumerate(row):
                    matrix[current_piece.y+i][current_piece.x+j] = \
                        current_piece.image[current_piece.rotation % 4][i][j]
            lost = check_lost(current_piece)
            current_piece = next_piece
            next_piece = get_next_piece(bag)
            fall_speed = 0.3
            fall_time = 0
            turn += 1
            change_piece = False
            matrix, delta_score = clear_rows(matrix)
            score += delta_score

        # print()
        # print(next_piece)
        # print()
        draw_window(win, matrix, current_piece)
        draw_next_piece(win, next_piece)
        pygame.display.update()
        if lost:
            draw_text_middle(win, "YOU LOST!", 80, (255, 255, 255))
            pygame.display.update()
            pygame.time.delay(1500)
            run = False
            break


def main_menu(win):  # *
    run = True
    quit = False
    while run:
        win.fill(BACKGROUND_COLOUR)
        draw_text_middle(win, 'Press Any Key To Play', 60, (255, 255, 255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                quit = True
            if event.type == pygame.KEYDOWN:
                quit = main(win)
            if quit:
                pygame.display.quit()
                return None
    pygame.display.quit()


win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Tetris')
main_menu(win)
