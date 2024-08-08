import pygame
import time
import random
from pygame.locals import *

class Square(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(Square, self).__init__()
        self.surf = pygame.Surface((10, 10))
        self.surf.fill((0, 200, 255))
        self.pos = [x, y]

pygame.init()

screen = pygame.display.set_mode((800, 600))
snake_block = 10
square = Square(40, 40)
font = pygame.font.SysFont("comicSansms", 40)
font_style = pygame.font.SysFont(None, 50)

def scorer(score):
    value = font.render("Score: " + str(score), True, (255, 255, 0))
    screen.fill((0, 0, 0), (0, 0, 200, 30))
    screen.blit(value, [0, 0])

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, (0, 200, 255), [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    screen.blit(mesg, [350, 250])

def main():
    gameOn = True
    clock = pygame.time.Clock()

    snake_list = []
    length_snake = 1
    direction = 'RIGHT'

    foodx = round(random.randrange(0, 800 - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, 600 - snake_block) / 10.0) * 10.0

    while gameOn:
        for event in pygame.event.get():
            if event.type == QUIT:
                gameOn = False
        
        keys = pygame.key.get_pressed()
        if keys[K_w] or keys[K_UP]:
            direction = 'UP'
        if keys[K_a] or keys[K_LEFT]:
            direction = 'LEFT'
        if keys[K_s] or keys[K_DOWN]:
            direction = 'DOWN'
        if keys[K_d] or keys[K_RIGHT]:
            direction = 'RIGHT'
        
        if direction == 'UP':
            square.pos[1] -= snake_block
        if direction == 'DOWN':
            square.pos[1] += snake_block
        if direction == 'LEFT':
            square.pos[0] -= snake_block
        if direction == 'RIGHT':
            square.pos[0] += snake_block

        if square.pos[0] >= 800 or square.pos[0] < 0 or square.pos[1] >= 600 or square.pos[1] < 0:
            gameOn = False

        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (0, 255, 0), [foodx, foody, snake_block, snake_block])

        snake_head = [square.pos[0], square.pos[1]]
        snake_list.append(snake_head)
        if len(snake_list) > length_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_head:
                gameOn = False

        our_snake(snake_block, snake_list)
        scorer(length_snake - 1)
        pygame.display.flip()

        if square.pos[0] == foodx and square.pos[1] == foody:
            foodx = round(random.randrange(0, 800 - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, 600 - snake_block) / 10.0) * 10.0
            length_snake += 1

        clock.tick(15)

    message("Game Over", (255, 0, 0))
    pygame.display.update()
    time.sleep(2)
    pygame.quit()

if __name__ == "__main__":
    main()
