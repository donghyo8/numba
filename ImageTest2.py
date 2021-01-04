import numpy as np
import cv2
import time
from numba import jit



def canny():
    img = cv2.imread('./datasets/cat/cat.jpg', cv2.IMREAD_GRAYSCALE)
    edge1 = cv2.Canny(img, 50, 200)
    edge2 = cv2.Canny(img, 100, 200)
    edge3 = cv2.Canny(img, 170, 200)
    cv2.imshow('original', img)
    cv2.imshow('Canny Edge1', edge1)
    cv2.imshow('Canny Edge2', edge2)
    cv2.imshow('Canny Edge3', edge3)



start = time.time()
canny()
print("===============================================================")
print("Canny Edge Processing time : " + str(time.time() - start) + " " + "seconds")
print("===============================================================")


@jit(nopython=True)
def canny2():
    img = cv2.imread('./datasets/cat/cat.jpg', cv2.IMREAD_GRAYSCALE)
    edge1 = cv2.Canny(img, 50, 200)
    edge2 = cv2.Canny(img, 100, 200)
    edge3 = cv2.Canny(img, 170, 200)
    cv2.imshow('original', img)
    cv2.imshow('Canny Edge1', edge1)
    cv2.imshow('Canny Edge2', edge2)
    cv2.imshow('Canny Edge3', edge3)


start = time.time()
canny()
print("===============================================================")
print("Numba Canny Edge Processing time : " + str(time.time() - start) + " " + "seconds")
print("===============================================================")