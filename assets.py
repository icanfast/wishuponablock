import pygame

controls = {
    'Left': pygame.K_LEFT,
    'Right': pygame.K_RIGHT,
    'Down': pygame.K_DOWN,
    'Rotate': pygame.K_UP,
    'Rotate Clockwise': pygame.K_e,
    'Rotate Counterclockwise': pygame.K_w,
    'Rotate 180': pygame.K_a,
    'Hard Drop': pygame.K_SPACE
}

shapes = {
    'S': {
        'rotations':
        [
            ['.00',
             '00.',
             '...'],
            ['.0.',
             '.00',
             '..0'],
            ['...',
             '.00',
             '00.'],
            ['0..',
             '00.',
             '.0.']
        ],
        'colour': (0, 244, 0)
    },

    'Z': {
        'rotations':
        [
            ['00.',
             '.00',
             '...'],
            ['..0',
             '.00',
             '.0.'],
            ['...',
             '00.',
             '.00'],
            ['.0.',
             '00.',
             '0..']
        ],
        'colour': (255, 0, 0)
    },

    'I': {
        'rotations':
        [
            ['....',
             '0000',
             '....',
             '....'],
            ['..0.',
             '..0.',
             '..0.',
             '..0.'],
            ['....',
             '....',
             '0000',
             '....'],
            ['.0..',
             '.0..',
             '.0..',
             '.0..']
        ],
        'colour': (0, 244, 242)
    },

    'O': {
        'rotations':
        [['.00.',
          '.00.',
          '....',
          '....'],
         ['.00.',
          '.00.',
          '....',
          '....'],
         ['.00.',
          '.00.',
          '....',
          '....'],
         ['.00.',
          '.00.',
          '....',
          '....'], ],
        'colour': (240, 240, 0)
    },

    'J': {
        'rotations':
        [
            ['0..',
             '000',
             '...'],
            ['.00',
             '.0.',
             '.0.'],
            ['...',
             '000',
             '..0'],
            ['.0.',
             '.0.',
             '00.']
        ],
        'colour': (0, 0, 250)
    },

    'L': {
        'rotations':
        [
            ['..0',
             '000',
             '...'],
            ['.0',
             '.0.',
             '.00'],
            ['...',
             '000',
             '0..'],
            ['00.',
             '.0.',
             '.0.']
        ],
        'colour': (254, 155, 0)
    },

    'T': {
        'rotations':
        [
            ['.0.',
             '000',
             '...'],
            ['.0',
             '.00',
             '.0.'],
            ['...',
             '000',
             '.0.'],
            ['.0.',
             '00.',
             '.0.']
        ],
        'colour': (175, 0, 249)
    }
}
