import pygame
from lib.constantes import *


class Pong:
    def __init__(self):
        pygame.init()
        self.screen_config()
        self.loop()
    
    def screen_config(self) -> None:
        self.screen = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption('Pong IA')
        self.clock = pygame.time.Clock()
    
    def loop(self):
        while True:
            self.screen.fill('#202020')
            self.clock.tick(FPS)
            self.eventos()
            pygame.display.update()
    
    def eventos(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


if __name__ == '__main__':
    Pong()
