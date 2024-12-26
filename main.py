import pygame
from lib.classes import *


class Pong:
    '''Jogo Pong'''
    def __init__(self):
        '''Método construtor'''
        pygame.init()
        self.screen_config()
        self.loop()
    
    def screen_config(self):
        '''Método para configuração da tela'''
        self.screen = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption('Pong IA')
        self.clock = pygame.time.Clock()
    
    def loop(self):
        '''Loop global do jogo'''
        jogo = Game(self.screen)
        while True:
            self.screen.fill('#202020')
            self.clock.tick(FPS)
            self.eventos()
            jogo.run()
            pygame.display.update()
    
    def eventos(self):
        '''Método para os eventos'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


if __name__ == '__main__':
    Pong()
