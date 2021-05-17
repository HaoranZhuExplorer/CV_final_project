import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from scipy import signal
import math
import sys

import matplotlib.patches as patches


if sys.platform == "darwin":
    # OS X
    mpl.use('tkagg')


def createRect(r, c, w, h, n):
    nw = n*w//(w+h)//2
    nh = n//2 - nw
    rows = np.round(np.linspace(r-h, r+h, nh))
    cols = np.round(np.linspace(c-w, c+w, nw))
    rect = []
    for col in cols:
        rect.append([rows[0], col])
    for row in rows:
        rect.append([row, cols[-1]])
    for col in cols[::-1]:
        rect.append([rows[-1], col])
    for row in rows[::-1]:
        rect.append([row, cols[0]])
    rect = np.array(rect)

    return rect.astype(int)
    

def normalize(im_in):
    im_out = ((im_in - np.min(im_in)) * (1/(np.max(im_in) - np.min(im_in)) * 255))
    return im_out


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def avgDistance(snake):
    d = 0
    for i, pt_i in enumerate(snake):
        d += distance(pt_i, snake[(i+1)%len(snake)])
    d /= len(snake)
    return d


def eCont(snake, d, id, pi):
    id_before = (id-1+len(snake)) % len(snake)
    pi_1 = snake[id_before]
    return (d - distance(pi_1, pi))**2


def eCurve(snake, id, pi):
    id_before = (id-1+len(snake)) % len(snake)
    id_after = (id+1) % len(snake)
    pi_before = snake[id_before]
    pi_after = snake[id_after]
    return (pi_before[0]+pi_after[0]-2*pi[0])**2+(pi_before[1]+pi_after[1]-2*pi[1])**2


def eImage(im, snake):
    filterX = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    filterY = np.array([[0,-1,0],[0,0,0],[0,1,0]])
    dX = signal.convolve2d(im, filterX, boundary='symm', mode='same')
    dY = signal.convolve2d(im, filterY, boundary='symm', mode='same')
    return -np.sqrt(np.power(dX, 2)+np.power(dY, 2))


def activeContourLoop(im, snake, grad_mag, weights):
    neighborhood = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    d = avgDistance(snake)
    stop = True
    for i, pi in enumerate(snake):
        min_energy, min_pi = float('inf'), pi
        for dPos in neighborhood:
            new_pi = pi+dPos
            if new_pi[0] < 0 or new_pi[0] >= len(im) or new_pi[1] < 0 or new_pi[1] >= len(im[0]):
                continue
            energy = np.array([eCont(snake, d, i, new_pi), eCurve(snake, i, new_pi), grad_mag[tuple(new_pi)]])
            sum_energy = np.sum(energy * weights)
            if sum_energy < min_energy:
                min_energy = sum_energy
                min_pi = new_pi
        if any(min_pi != pi):
            stop = False
        snake[i] = min_pi
    return stop


def activeContour(im, init, weights, iterations=-1):
    snake = np.copy(init)
    grad_mag = eImage(im, snake)
    if iterations > 0:
        for _ in range(iterations):
            activeContourLoop(im, snake, grad_mag, weights)
    else:
        stop = False
        while not stop:
            stop = activeContourLoop(im, snake, grad_mag, weights)
    return snake


if __name__ == '__main__':
    fn = 'plate.png'

    im = rgb2gray(io.imread(fn))
    weights = [0.1,0.1,1000]

    rows, cols = im.shape[0], im.shape[1]
    
    center_x, center_y = rows//1.95, cols//2
    h, w = round(0.46*rows/2), round(0.94*cols/2)
    
    im_denoised = cv2.GaussianBlur(im, (7,7), 0)

    init = createRect(center_x, center_y, w, h, 100)
    snake = activeContour(im, init[:-1,:], weights, iterations=3000) # Passing int[:-1,:] to avoid disconnection
    snake = np.append(snake, [snake[0]], axis=0) # Append first point to the rear of snake
    
    # rectangle patch area
    begin = np.min(snake, axis=0)
    end = np.max(snake, axis=0)
    height, width = end[0]-begin[0]+1, end[1]-begin[1]+1
    rect = patches.Rectangle((begin[1], begin[0]), width, height, linewidth=1, edgecolor='r', facecolor='none')

    io.imsave("plate_crop.png", im[begin[0]:end[0],begin[1]:end[1]])

    fig = plt.figure(figsize=(6,6))
    plt.gray()
    # plt.title(f'{pts[param]} points. α={weights[0]}, β={weights[1]}, γ={weights[2]}. 2000 iterations')
    ax = plt.subplot(111)
    ax.imshow(im)
    ax.plot(init[:,1], init[:,0], '--b', lw=2)
    ax.plot(snake[:,1], snake[:,0], '-c.', lw=1)
    ax.add_patch(rect)
    plt.show()
