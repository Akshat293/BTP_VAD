import pygame
from random import randint
# display
width=300
height=500

pygame.init()
win=pygame.display.set_mode((width,height))
pygame.display.set_caption("Dice Rolling Simulator")

# font type
font=pygame.font.SysFont("Comic Sans Ms",40)

# define colors
white= (255,255,255)
blue_violish=(153,153,255)
black=(0,0,0)
# change the color of the background into white
win.fill(white)

# get dice images

dices=[pygame.image.load("images/1.jpg"),pygame.image.load("images/2.jpg"),pygame.image.load("images/3.jpg"),pygame.image.load("images/4.jpg"),pygame.image.load("images/5.jpg"),pygame.image.load("images/6.jpg")]

# display roll button
pygame.draw.rect(win,blue_violish,pygame.Rect(75,350,150,90))
win.blit(font.render("roll",True,black),(120,360))
pygame.display.flip()

# create a function that select a random dice
def dice_random(x,y):
    if x>=75 and x<=225:
        if (y>=350 and y<=440):
            l= randint(0,5)
            win.blit(dices[l],(90,100))
            pygame.display.flip()


while True:
    for event in pygame.event.get():
        if event.type==pygame.MOUSEBUTTONDOWN:
            pos=pygame.mouse.get_pos()
            dice_random(pos[0],pos[1])

        if event.type==pygame.QUIT:
            pygame.quit()