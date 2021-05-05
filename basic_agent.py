import matplotlib.pyplot as plt
from statistics import mode
import random
import numpy as np
from PIL import Image


class Cell:
    def __init__(self, classifier, red, green, blue):
        # mine
        self.classifier = classifier
        # The red value of a pixel
        self.red = red
        # The green value of a pixel
        self.green = green
        # The blue value of a pixel
        self.blue = blue


def basic_agent():
    image = Image.open('beach.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    middle = int(len(image[0]) / 2)
    left_half = image[:, :middle]
    right_half = image[:, middle:]

    # recolor the right side by finding 6 most similar in testing data

    centers, cluster_list = k_classification(left_half)
    # FINAL OUTPUT FOR LEFT (representative colors)
    colored_left = np.copy(recolor_left(left_half, centers, cluster_list))

    grayed_left = convert_to_grayscale(left_half)[0]

    grayed_right, duplicate = convert_to_grayscale(right_half)

    grayed_left_pixel_patch = three_by_three_pixel_patches(grayed_left)

    tracker = 0

    # iterate through testing
    # iterate through rows
    for i in range(1, len(grayed_right) - 1):
        # iterate through columns
        for j in range(1, len(grayed_right[0]) - 1):
            min1, min2, min3, min4, min5, min6 = 1000, 1000, 1000, 1000, 1000, 1000
            patch_list = [[], [], [], [], [], []]
            # find six patches

            # take a sample from the total training data to compare with test data
            # the higher the number the better the resulting image quality
            samples = random.sample(list(grayed_left_pixel_patch), 1000)

            for k in samples:
                dist = find_dist(k[0], grayed_right[i - 1:i + 2, j - 1:j + 2])
                if dist < min1:
                    min1 = dist
                    patch_list[1] = patch_list[0]
                    patch_list[0] = k[1]
                    continue
                if dist < min2:
                    min2 = dist
                    patch_list[2] = patch_list[1]
                    patch_list[1] = k[1]
                    continue
                if dist < min3:
                    min3 = dist
                    patch_list[3] = patch_list[2]
                    patch_list[2] = k[1]
                    continue
                if dist < min4:
                    min4 = dist
                    patch_list[4] = patch_list[3]
                    patch_list[3] = k[1]
                    continue
                if dist < min5:
                    min5 = dist
                    patch_list[5] = patch_list[4]
                    patch_list[4] = k[1]
                    continue
                if dist < min6:
                    min6 = dist
                    patch_list[5] = k[1]
                    continue

                # get color of 6 middel pixels
            for a in range(0, len(patch_list)):
                x = patch_list[a][1]
                y = patch_list[a][0]

                # replace the patches/coordinates we got with the colors they represent
                patch_list[a] = cluster_list[y][x].cluster

            try:
                freq = mode(patch_list)
                duplicate[i][j] = centers[freq]
            finally:
                x = random.randint(0, len(patch_list) - 1)
                tie = patch_list[x]
                duplicate[i][j] = centers[tie]

            tracker += 1
        print(tracker / (len(grayed_right) * len(grayed_right[0])) * 100, "%")

    plt.imshow(colored_left)
    plt.show()

    plt.imshow(duplicate)
    plt.show()


# calculate the euclidean distance
def find_dist(a, b):
    dist = np.linalg.norm(a - b)
    return dist


# turn the image format to [r,g,b,cluster] for later convenience
def reformat(arr):
    cluster_list = np.empty((len(arr), len(arr[0])), dtype=object)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            temp = Cell(-1, arr[i][j][0], arr[i][j][1], arr[i][j][2], )
            cluster_list[i][j] = temp
    return cluster_list


# the kmeans algorithm to find the centroids of our image data
def k_classification(arr):
    a = []
    # generates 5 random points
    for i in range(5):
        a.append(list(arr[random.randint(0, len(arr) - 1)][random.randint(0, len(arr[0]) - 1)]))

    x = 0

    cluster_list = reformat(arr)
    while x != 15:
        temp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        counter = [0, 0, 0, 0, 0]
        x = 0

        # goes through every pixel
        for y in range(len(arr)):
            for x in range(len(arr[0])):
                # print("x", x, "y", y)
                min_distance = 3000
                cluster = -1

                for j in range(5):
                    # finding the closest centroid
                    updated = find_dist(arr[y][x], a[j])
                    if updated < min_distance:
                        min_distance = updated
                        cluster = j

                # tag the pixel
                cluster_list[y][x].cluster = cluster
                # add to the array of sums
                temp[cluster][0] += cluster_list[y][x].r
                temp[cluster][1] += cluster_list[y][x].g
                temp[cluster][2] += cluster_list[y][x].b
                temp[cluster] += 1
        # finds the average
        for i in range(5):
            for j in range(3):
                if counter[i] != 0:
                    avg = int(temp[i][j] / counter[j])
                else:
                    avg = 0
                if abs(avg - a[i][j]) > 5:
                    a[i][j] = avg
                else:
                    x += 1

    return a, cluster_list


# recolor the left image in terms of representative colors
def recolor_left(left, center, cluster_list):
    for a in range(0, len(left)):
        for b in range(0, len(left[0])):
            left[a][b] = center[cluster_list[a][b].cluster]

    return left


# a method to turn an image to grayscale
def convert_to_grayscale(jpg):
    # recolor each pixel
    jpg_copy = np.copy(jpg)
    jpg = jpg.tolist()
    for i in range(0, len(jpg)):
        for j in range(0, len(jpg[i])):
            jpg[i][j] = 0.21 * jpg[i][j][0] + 0.72 * jpg[i][j][1] + 0.07 * jpg[i][j][2]
            jpg_copy[i][j] = 0.21 * jpg[i][j][0] + 0.72 * jpg[i][j][1] + 0.07 * jpg[i][j][2]

    return np.array(jpg), jpg_copy


# get all of the 3x3 patches in an image
def three_by_three_pixel_patches(jpg):
    pixel_patch_list = []
    # iterate through left gray patch
    # iterate through rows
    for x in range(1, len(jpg) - 1):
        # iterate through columns
        for y in range(1, len(jpg[0]) - 1):
            # grayed_left[i][j] starts on middle pixel
            # find the rest of the patch (adjacent pixels)
            pixel_patch_list.append((jpg[x - 1:x + 2, y - 1:y + 2], (x, y)))

    return pixel_patch_list
