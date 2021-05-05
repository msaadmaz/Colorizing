import cv2
import math
from random import randint
import numpy as np
import copy


def basic_agent():
    image = cv2.imread('beach.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    middle = int(len(image[0]) / 2)
    left_half = image[:, :middle]
    right_half = image[:, middle:]
