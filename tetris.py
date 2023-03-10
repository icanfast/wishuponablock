import pygame
import random
import copy
import json
import datetime
import getpass
import os
from assets import shapes, controls

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 30 height per block
block_size = 30
bag = []
LOCK_DELAY = 40
GRID_COLOUR = (30, 30, 30)
BACKGROUND_COLOUR = (20, 20, 20)
top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height


class Piece(object):  # *
    def __init__(self, x, y, shape, color=None):
        self.x = x
        self.y = y
        self.shape = shape['rotations']
        if color:
            self.color = color
        else:
            self.color = shape['colour']
        self.rotation = 0

    def __str__(self) -> str:
        description = 'X: ' + str(self.x)
        description += '\n'
        description += 'Y: ' + str(self.y)
        description += '\n'
        description += str(self.shape[self.rotation % 4])
        return description


def create_grid(locked_pos={}):  # *
    grid = [[GRID_COLOUR for _ in range(10)] for _ in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_pos:
                c = locked_pos[(j, i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i]
                     [j] == GRID_COLOUR] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True

    return False


def bag_shuffler():
    all_keys = ['S', 'Z', 'I', 'O', 'J', 'L', 'T']
    bag_of_keys = []
    turn_no = 1
    while True:
        print('Turn', turn_no)
        if not bag_of_keys:
            print('Refilling Bag')
            bag_of_keys = copy.deepcopy(all_keys)
            random.shuffle(bag_of_keys)
        current_piece_key = bag_of_keys.pop()
        current_piece = shapes[current_piece_key]
        print('Current Piece: ', current_piece_key)
        print('Bag: ', bag_of_keys)
        print()

        turn_no += 1
        yield Piece(5, 3, current_piece)


shuffler = bag_shuffler()  # dubious idea idk


def get_shape():
    return next(shuffler)


def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("sfnsmono", size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width()/2),
                 top_left_y + play_height/2 - label.get_height()/2))


def draw_gridlines(surface, grid):
    sx = top_left_x
    sy = top_left_y

    for i in range(len(grid)):
        pygame.draw.line(surface, (64, 64, 64), (sx, sy +
                         i*block_size), (sx+play_width, sy + i*block_size))
    for j in range(len(grid[i])):
        pygame.draw.line(surface, (64, 64, 64), (sx + j *
                                                 block_size, sy), (sx + j*block_size, sy + play_height))


def clear_rows(grid, locked):

    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if GRID_COLOUR not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)

    return inc


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('sfnsmono', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = top_left_x + play_width + 70
    sy = top_left_y + play_height/2 - 250
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color, (sx + j*block_size,
                                 sy + i*block_size, block_size, block_size), 0)

    surface.blit(label, (sx - 30, sy - 50))


def shape_to_colours(piece):  # unused for now
    colours = []
    for row in piece.shape[piece.rotation % 4]:
        row_colours = []
        for item in row:
            if item == '0':
                row_colours.append(piece.colour)
            else:
                row_colours.append(GRID_COLOUR)
        colours.append(row_colours)
    return colours


def draw_ghost(surface, shape, grid):
    ghost = Piece(shape.x, 19, {'rotations': shape.shape}, shape.color)
    while not valid_space(ghost, grid):
        ghost.y += 1
    ghost.y -= 1
    # print()
    # print()
    # print(shape)
    # print()
    # print(ghost)
    # print()
    # print()
    # exit()
    # draw ghost
    ghost_colour = tuple([(i+j)/2 for i, j in zip(ghost.color, GRID_COLOUR)])
    for i, line in enumerate(ghost.shape[shape.rotation % 4]):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, ghost_colour, (top_left_x + (ghost.x-2)*block_size + j*block_size,
                                                         top_left_y + ghost.y*block_size + i*block_size,
                                                         block_size, block_size), 0)


def max_score():
    return '0'


def write_snapshot(snapshot, snapshot_path, turn):
    file_path = os.path.join(snapshot_path, 'turn_'+str(turn)+'.json')
    with open(file_path, 'w') as f:
        json.dump(snapshot, f)


def draw_window(surface, grid, score=0, last_score=0):
    surface.fill(BACKGROUND_COLOUR)

    pygame.font.init()
    font = pygame.font.SysFont('sfnsmono', 60)
    label = font.render('Tetris', 1, (255, 255, 255))

    surface.blit(label, (top_left_x + play_width /
                 2 - (label.get_width() / 2), 30))

    # current score
    font = pygame.font.SysFont('sfnsmono', 30)
    label = font.render('Score: ' + str(score), 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100

    surface.blit(label, (sx, sy + 350))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (top_left_x + j*block_size,
                             top_left_y + i*block_size, block_size, block_size), 0)

    pygame.draw.rect(surface, (64, 64, 64), (top_left_x,
                     top_left_y, play_width, play_height), 5)

    draw_gridlines(surface, grid)


def main(win):
    last_score = max_score()
    locked_positions = {}
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.27
    level_time = 0
    score = 0
    turn = 1
    record = True
    lock_delay = 0
    if record:
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
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if level_time/1000 > 5:
            level_time = 0
            if level_time > 0.12:
                level_time -= 0.005

        if fall_time/1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not(valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                lock_delay += clock.get_rawtime()
                if lock_delay >= LOCK_DELAY:
                    lock_delay = 0
                    change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                pygame.quit()
                return True

            if event.type == pygame.KEYDOWN:
                if event.key == controls['Left']:
                    current_piece.x -= 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.x += 1
                if event.key == controls['Right']:
                    current_piece.x += 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.x -= 1
                if event.key == controls['Down']:
                    current_piece.y += 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.y -= 1
                if event.key == controls['Rotate Clockwise'] or event.key == controls['Rotate']:
                    current_piece.rotation += 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.rotation -= 1
                if event.key == controls['Rotate Counterclockwise']:
                    current_piece.rotation -= 1
                    if not(valid_space(current_piece, grid)):
                        current_piece.rotation += 1
                if event.key == controls['Rotate 180']:
                    current_piece.rotation -= len(current_piece.shape)//2
                    if not(valid_space(current_piece, grid)):
                        current_piece.rotation += len(current_piece.shape)//2
                if event.key == controls['Hard Drop']:
                    fall_speed = 0.00001
                    # change_piece = True
        shape_pos = convert_shape_format(current_piece)

        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        if change_piece:
            snapshot = {
                'grid': grid,
                'score': score,
                'last_score': last_score}
            if record:
                write_snapshot(snapshot=snapshot,
                               snapshot_path=snapshot_path, turn=turn)
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            fall_speed = 0.27
            turn += 1
            change_piece = False
            score += clear_rows(grid, locked_positions) * 10

        draw_window(win, grid, score, last_score)
        draw_next_shape(next_piece, win)
        # draw_ghost(win, current_piece, grid)
        pygame.display.update()

        if check_lost(locked_positions):
            draw_text_middle(win, "YOU LOST!", 80, (255, 255, 255))
            pygame.display.update()
            pygame.time.delay(1500)
            run = False


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
            if event.type == pygame.KEYDOWN:
                quit = main(win)
            if quit:
                pygame.display.quit()
                return None
    pygame.display.quit()


win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')
main_menu(win)
