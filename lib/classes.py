import pygame, json
from random import randrange
from lib.rede_neural import Network
from lib.constantes import *


class Player:
    def __init__(self, pos: tuple):
        self.rect = pygame.Rect(*pos, 10, 100)
    
    def update(self) -> None:
        keys = pygame.key.get_pressed()
        if not (keys[pygame.K_UP] and keys[pygame.K_DOWN]):
            if keys[pygame.K_UP] and self.rect.top > 80:
                self.rect.y -= 5
            elif keys[pygame.K_DOWN] and self.rect.bottom < ALTURA:
                self.rect.y += 5
    
    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, 'white', self.rect)


class PlayerIA(Player):
    def __init__(self, pos: tuple):
        super().__init__(pos)
        with open(os.path.join(DIRETORIO_PRINCIPAL, 'network.json'), 'r') as file:
            self.rede = Network(**json.load(file))
    
    def update(self, parametro: tuple):
        result = self.rede.feedforward([self.rect.centery - parametro[0]])
        if not all(result):
            if result[0] and self.rect.top > 80:
                self.rect.y -= 5
            if result[1] and self.rect.bottom < ALTURA:
                self.rect.y += 5


class Bola:
    def __init__(self, pos: tuple):
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.new_postion(pos)
        self.vel = VELOCIDADE
    
    def new_postion(self, pos: tuple):
        self.rect.center = pos
        self.dir_x, self.dir_y = randrange(-1, 2, 2), randrange(-1, 2, 2)
    
    def draw(self, screen: pygame.Surface):
        pygame.draw.rect(screen, 'white', self.rect)
    
    def update(self):
        self.rect.x += self.vel * self.dir_x
        self.rect.y += self.vel * self.dir_y
        if self.rect.top + self.vel * self.dir_y < 80:
            self.dir_y *= -1
        elif self.rect.bottom + self.vel * self.dir_y > ALTURA:
            self.dir_y *= -1


class Game:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.jogador1 = Player((0, 290))
        self.jogador2 = PlayerIA((LARGURA - 10, 290))
        self.bola = Bola((LARGURA // 2, (ALTURA + 80) // 2))
        self.jogador1_pontos = 0
        self.jogador2_pontos = 0
        self.start = False
    
    def draw_text(self):
        font1 = pygame.font.SysFont('04b19', 200)
        font2 = pygame.font.SysFont('04b19', 35)
        pontos = font1.render(f'{self.jogador1_pontos}:{self.jogador2_pontos}', True, '#303030')
        player = font2.render('Player', True, 'white')
        neural_network = font2.render('Neural Network', True, 'white')
        self.screen.blit(pontos, pontos.get_rect(center=(LARGURA / 2, (ALTURA + 80) / 2)))
        self.screen.blit(player, player.get_rect(center=(LARGURA / 4, 40)))
        self.screen.blit(neural_network, neural_network.get_rect(center=(3 * LARGURA / 4, 40)))
    
    def draw(self):
        pygame.draw.line(self.screen, 'white', (0, 80), (LARGURA, 80))
        pygame.draw.line(self.screen, 'white', (LARGURA / 2, 0), (LARGURA / 2, 80))
        self.draw_text()
        self.jogador1.draw(self.screen)
        self.jogador2.draw(self.screen)
        self.bola.draw(self.screen)

    def update(self):
        self.jogador1.update()
        self.jogador2.update([self.bola.rect.centery])
        if any(pygame.key.get_pressed()):
            self.start = True
        if self.start:
            self.bola.update()
        
        if self.bola.rect.right < 0:
            self.start = False
            self.bola.new_postion((LARGURA // 2, (ALTURA + 80) // 2))
            self.jogador2_pontos += 1
            self.jogador1.rect.centery = (ALTURA + 80) // 2
        elif self.bola.rect.left > LARGURA:
            self.start = False
            self.bola.new_postion((LARGURA // 2, (ALTURA + 80) // 2))
            self.jogador1_pontos += 1
            self.jogador1.rect.centery = (ALTURA + 80) // 2
        
        if self.jogador1.rect.colliderect(self.bola.rect):
            self.bola.dir_x *= -1
        elif self.jogador2.rect.colliderect(self.bola.rect):
            self.bola.dir_x *= -1

    def run(self):
        self.update()
        self.draw()
