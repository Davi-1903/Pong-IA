import pygame, json
from random import choice
from lib.rede_neural import Network
from lib.constantes import *


class Player:
    '''Classe que representa o jogador.
    
    Atributos:
    - `rect:` Retângulo tanto de colisão como de exibição;
    '''
    def __init__(self, pos: tuple):
        '''Método construtor.
        
        Parâmetros:
        - `pos:` Posição que o jogador ficará;
        '''
        self.rect = pygame.Rect(*pos, 10, 100)
    
    def update(self) -> None:
        '''Método que atualiza o jogador.'''
        keys = pygame.key.get_pressed()
        if not (keys[pygame.K_UP] and keys[pygame.K_DOWN]):
            if keys[pygame.K_UP] and self.rect.top > 80:
                self.rect.y -= VELOCIDADE
            elif keys[pygame.K_DOWN] and self.rect.bottom < ALTURA:
                self.rect.y += VELOCIDADE
    
    def draw(self, screen: pygame.Surface):
        '''Método para a exibição do jogador.
        
        Parâmetros:
        - `screen:` Tela do jogo;
        '''
        pygame.draw.rect(screen, 'white', self.rect)


class PlayerIA(Player):
    '''Subclasse de `Player` que representa do bot.'''
    def __init__(self, pos: tuple):
        super().__init__(pos)
        with open(os.path.join(DIRETORIO_PRINCIPAL, 'network.json'), 'r') as file:
            self.rede = Network(**json.load(file))
    
    def update(self, parametro: tuple):
        result = self.rede.feed_forward([self.rect.centery - parametro[0]])
        if not all(result):
            if result[0] and self.rect.top > 80:
                self.rect.y -= VELOCIDADE
            if result[1] and self.rect.bottom < ALTURA:
                self.rect.y += VELOCIDADE


class Bola:
    '''Classe que representa a bola.
    
    Atributos:
    - `rect:` Retângulo tanto de colisão como de exibição;
    - `dir_x:` Sentido na horizontal;
    - `dir_y:` Sentido na vertical;
    '''
    def __init__(self, pos: tuple):
        '''Método construtor.
        
        Parâmetros:
        - `pos:` Posição inical da bola;
        '''
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.new_direction(pos)
    
    def new_direction(self, pos: tuple):
        '''Método que define uma nova posição para a bola.
        
        Parâmetros:
        - `pos:` Nova posição;
        '''
        self.rect.center = pos
        self.dir_x, self.dir_y = choice([-1, 1]), choice([-1, 1])
    
    def draw(self, screen: pygame.Surface):
        '''Método que desenha a bola.
        
        Parâmetros:
        - `screen:` Tela do jogo;
        '''
        pygame.draw.rect(screen, 'white', self.rect)
    
    def update(self):
        '''Método que atualiza do jogo.'''
        self.rect.x += VELOCIDADE * self.dir_x
        self.rect.y += VELOCIDADE * self.dir_y
        if self.rect.top + VELOCIDADE * self.dir_y < 80 or self.rect.bottom + VELOCIDADE * self.dir_y > ALTURA:
            self.dir_y *= -1


class Game:
    '''Classe que controla o jogo em si.
    
    Atributos:
    - `screen:` Tela do jogo;
    - `jogador1:` Player;
    - `jogador2:` PlayerIA;
    - `bola:` Bola;
    - `jogador1_pontos:` Pontos do primeiro jogador;
    - `jogador2_pontos:` Pontos do segundo jogador;
    - `start:` Se o jogo começou ou não;
    '''
    def __init__(self, screen: pygame.Surface):
        '''Método construtor.

        Parâmetros:
        - `screen:` Tela do jogo;
        '''
        self.screen = screen
        self.jogador1 = Player((0, 290))
        self.jogador2 = PlayerIA((LARGURA - 10, 290))
        self.bola = Bola((LARGURA // 2, (ALTURA + 80) // 2))
        self.jogador1_pontos = 0
        self.jogador2_pontos = 0
        self.start = False
    
    def draw_text(self):
        '''Método que desenha os textos do jogo.'''
        font1 = pygame.font.SysFont('04b19', 200)
        font2 = pygame.font.SysFont('04b19', 35)
        pontos = font1.render(f'{self.jogador1_pontos}:{self.jogador2_pontos}', True, '#303030')
        player = font2.render('Player', True, 'white')
        neural_network = font2.render('Neural Network', True, 'white')
        self.screen.blit(pontos, pontos.get_rect(center=(LARGURA / 2, (ALTURA + 80) / 2)))
        self.screen.blit(player, player.get_rect(center=(LARGURA / 4, 40)))
        self.screen.blit(neural_network, neural_network.get_rect(center=(3 * LARGURA / 4, 40)))
    
    def draw(self):
        '''Método que desenha o jogo.'''
        pygame.draw.line(self.screen, 'white', (0, 80), (LARGURA, 80))
        pygame.draw.line(self.screen, 'white', (LARGURA / 2, 0), (LARGURA / 2, 80))
        self.draw_text()
        self.jogador1.draw(self.screen)
        self.jogador2.draw(self.screen)
        self.bola.draw(self.screen)

    def update(self):
        '''Método que atualiza o jogo.'''
        self.jogador1.update()
        self.jogador2.update([self.bola.rect.centery])
        if any(pygame.key.get_pressed()):
            self.start = True
        if self.start:
            self.bola.update()
        
        if self.bola.rect.right < 0:
            self.start = False
            self.bola.new_direction((LARGURA // 2, (ALTURA + 80) // 2))
            self.jogador2_pontos += 1
            self.jogador1.rect.centery = (ALTURA + 80) // 2
        elif self.bola.rect.left > LARGURA:
            self.start = False
            self.bola.new_direction((LARGURA // 2, (ALTURA + 80) // 2))
            self.jogador1_pontos += 1
            self.jogador1.rect.centery = (ALTURA + 80) // 2
        
        if self.jogador1.rect.colliderect(self.bola.rect) or self.jogador2.rect.colliderect(self.bola.rect):
            self.bola.dir_x *= -1

    def run(self):
        '''Método que executa o jogo.'''
        self.update()
        self.draw()
