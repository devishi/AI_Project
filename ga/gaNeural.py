from itertools import cycle
import random
import sys
import time
import datetime
import logging
import os
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import json
from genome import gene
from network import NeuralNetwork
from population import Population
from setUp import setup
import pickle
import numpy as np
import subprocess
from colorama import *

saveNet = "saved/" + str(datetime.date.today()) + "_" + time.strftime("%X")

if not os.path.exists(saveNet):
    os.makedirs(saveNet)

bestFitness = 0
fitnesses = []
bestFit = []
detectionOffset = 40

SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
IMAGES, HITMASKS = {}, {}
DIEIFTOUCHTOP = True
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

def main():
    initPygame()
    bestFitness = 0
    population = Population()
    population.randPop()
    generation = 1
    maxgeneration = setup.maxGeneration

    lastAvg = 0
    while generation <= maxgeneration :
        birdnumber = 1
        for i in xrange(population.size()):
            gene = startGame(population.getGenome(i))
            population.setGenomeFitness(i,gene.fitness)
            console = {
            'generation' : generation,
            'birdnumber' : birdnumber,
            'lastfitness' : gene.fitness,
            'lastAvg' : lastAvg,
            'bestfitness' : bestFitness
            }
            printData(console)
            if gene.fitness > bestFitness:
                global bestFitness
                bestFitness = gene.fitness
                gene.network.save(saveNet + "/bestfitness.json")
            birdnumber += 1
        global fitnesses
        fitnesses.append(population.averageFitness())
        lastAvg = population.averageFitness()
        global bestFit
        bestFit.append(population.findFittest().fitness)
        population.evolvePopulation()
        generation += 1


def initPygame():
        global SCREEN, FPSCLOCK
        pygame.init()
        init()
        FPSCLOCK = pygame.time.Clock()
        SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption("Genetic Learning")

        IMAGES['numbers'] = (
            pygame.image.load('assets/sprites/0.png').convert_alpha(),
            pygame.image.load('assets/sprites/1.png').convert_alpha(),
            pygame.image.load('assets/sprites/2.png').convert_alpha(),
            pygame.image.load('assets/sprites/3.png').convert_alpha(),
            pygame.image.load('assets/sprites/4.png').convert_alpha(),
            pygame.image.load('assets/sprites/5.png').convert_alpha(),
            pygame.image.load('assets/sprites/6.png').convert_alpha(),
            pygame.image.load('assets/sprites/7.png').convert_alpha(),
            pygame.image.load('assets/sprites/8.png').convert_alpha(),
            pygame.image.load('assets/sprites/9.png').convert_alpha()
        )

        # base (ground) sprite
        IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

def startGame(gene):
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    playerShmVals = {'val': 0, 'dir': 1}
    basex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    movementInfo = {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }
    gene.network.setGeneData(gene.genes)
    crashInfo = mainGame(movementInfo,gene)
    gene = crashInfo['gene']
    gene.fitness = crashInfo['score']

    if gene.fitness < 0:
        gene.fitness = 0
    return gene

def mainGame(movementInfo,gene):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    framesurvived = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                #Store Stat
                reportStat()
                pygame.quit()
                showStat(saveNet)
        #Evaluate the NN
        if playerx < lowerPipes[0]['x'] + detectionOffset:
            nextPipe = lowerPipes[0]
        else:
            nextPipe = lowerPipes[1]

        nextPipeY = float(SCREENHEIGHT - nextPipe['y']) / SCREENHEIGHT

        playerYcorrectAxis = float(SCREENHEIGHT - playery) / SCREENHEIGHT
        distanceBetweenPlayerAndNextPipe = float(nextPipe['x'] - playerx)/ SCREENWIDTH

        NNinput = np.array([[playerYcorrectAxis],[nextPipeY]])

        NNoutput = gene.network.feedforward(NNinput)

        if NNoutput > 0.5:
            if playery > -2 * IMAGES['player'][0].get_height():
                playerVelY = playerFlapAcc
                playerFlapped = True

        info = {'playery': playerYcorrectAxis, 'pipey': nextPipeY, 'distance': distanceBetweenPlayerAndNextPipe}


        # check for crash here
        crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
                               upperPipes, lowerPipes)
        if crashTest[0] or playery < 5:
            gene.fitness = framesurvived
            return {
                'score': score,
                'gene': gene
            }

        # check for score
        playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = playerIndexGen.next()
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        if playerVelY < playerMaxVelY and not playerFlapped:
            playerVelY += playerAccY
        if playerFlapped:
            playerFlapped = False
        playerHeight = IMAGES['player'][playerIndex].get_height()
        if playery > 5:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        framesurvived += 1
        SCREEN.blit(IMAGES['player'][playerIndex], (playerx, playery))
        pygame.display.update()
        FPSCLOCK.tick(4000)

def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

def displayInfo(info):
    ###Display useful info : the input for the ANN
    myfont = pygame.font.Font(None, 30)
    # render text
    playery = str(info['playery'])
    tubey = str(info['pipey'])
    distance = str(info['distance'])

    labelplayery = myfont.render(playery,1,(255,255,0))
    labeltubey = myfont.render(tubey,1,(0,255,255))
    labeldistance = myfont.render(distance,1,(255,255,255))

    SCREEN.blit(labelplayery, (SCREENWIDTH / 2 - 100, SCREENHEIGHT * 0.7))
    SCREEN.blit(labeltubey, (SCREENWIDTH / 2  - 100, SCREENHEIGHT * 0.8))
    SCREEN.blit(labeldistance, (SCREENWIDTH / 2 - 100, SCREENHEIGHT * 0.9))

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = NeuralNetwork(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def reportStat():
    with open(saveNet + '/fitnesses.dat', 'wb') as handle:
        pickle.dump(fitnesses, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(saveNet + '/bestFit.dat', 'wb') as handle:
        pickle.dump(bestFit, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(saveNet + '/bestfitness.dat', 'wb') as handle:
        pickle.dump(bestFitness, handle, protocol=pickle.HIGHEST_PROTOCOL)

def printData(info):
    subprocess.call(["printf", "\033c"])
    if info["generation"] > 1:
        print("Last generation:")
        print("Average fitness: %s" % str(info["lastAvg"]))
        print("\n************************************")
    if info["birdnumber"] > 1:
        print("Last Fitness: %s" % str(info["lastfitness"]))

    print("Best Fitness: %s" % str(info["bestfitness"]))
    print("Current generation:")
    print("Generation number : %s/%s" % (str(info["generation"]),str(setup.maxGeneration)))
    print("Bird number: %s/%s" % (str(info["birdnumber"]), str(setup.birdsInPlay)))
    print("\n************************************")

def showStat(folder):
    bestFit = pickle.load(open(folder + '/bestFit.dat', 'rb'))
    bestfitness = pickle.load(open(folder + '/bestfitness.dat', 'rb'))
    subprocess.call(["printf", "\033c"])
    print "Number of generation: %s" % len(bestFit)
    print("Best Fitness: %s" % bestFitness)

    plt.figure(1)
    plt.plot(bestFit)
    plt.show()
    sys.exit()

if __name__ == '__main__':
    main()
