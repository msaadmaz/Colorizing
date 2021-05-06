import matplotlib.pyplot as plt
from statistics import mode
import random
import numpy as np
from PIL import Image

class Cell:
    def __init__(self, classifier, red, green, blue):
        # THE CLASSIFIER OF A CLUSTER
        self.classifier = classifier

        # THE RED VALUE OF A PIXEL FORM 0<X<255
        self.red = red

        # THE GREEN VALUE OF A PIXEL FORM 0<X<255
        self.green = green

        # THE BLUE VALUE OF A PIXEL FORM 0<X<255
        self.blue = blue


def basic_agent():
    image = Image.open('beach.jpg')
    image_2 = image.convert('L', (0.21, 0.72, 0.07, 0))

    middle = int(len(image_2[0]) / 2)
    right_half = image[:, middle:]
    left_half = image[:, :middle]

    # START COLORING RIGHT SIDE USING THE 6 CLUSTERS
    # CREATE A CLUSTER LIST
    centers, cluster_list = k_classification(left_half)

    # RECOLOR THE LEFT HAND SIDE OF THE IMAGE
    for a in range(0, len(left_half)):
        for b in range(0, len(left_half[0])):
            left_half[a][b] = centers[cluster_list[a][b].classifier]
    colored_left = np.copy(left_half)

    left_is_greyed = convert_to_grayscale(left_half)[0]

    right_is_greyed, duplicate = convert_to_grayscale(right_half)

    pixel_patch_list = []
    # ITERATE THROUGH LEFT GRAY AREA
    for x in range(1, len(left_is_greyed) - 1):
        # iterate through columns
        for y in range(1, len(left_is_greyed[0]) - 1):
            # grayed_left[i][j] starts on middle pixel
            # find the rest of the patch (adjacent pixels)
            pixel_patch_list.append((left_is_greyed[x - 1:x + 2, y - 1:y + 2], (x, y)))

    grayed_left_pixel_patch = pixel_patch_list

    for i in range(1, len(right_is_greyed) - 1):
        # iterate through columns
        for j in range(1, len(right_is_greyed[0]) - 1):
            # THIS WILL LOCATE 6 PIXEL PATCHES
            patch_list = [[], [], [], [], [], []]

            # RANDOM TRAINING DATA IS TAKEN
            # the higher the number the better the resulting image quality
            training_data_samples = random.sample(list(grayed_left_pixel_patch), 1000)
            x1 = 2000
            x2 = 2000
            x3 = 2000
            x4 = 2000
            x5 = 2000
            x6 = 2000
            for z in training_data_samples:
                # EUCLIDEAN DISTANCE FOUND FOR TEST DATA
                distance = np.linalg.norm((z[0] - right_is_greyed[i - 1:i + 2, j - 1:j + 2]))
                # TRAINING DATA IS COMPARED WITH TEST DATA
                if x1 < distance:
                    x1 = distance
                    patch_list[1] = patch_list[0]
                    patch_list[0] = z[1]
                    continue
                # TRAINING DATA IS COMPARED WITH TEST DATA
                if x2 < distance:
                    x2 = distance
                    patch_list[2] = patch_list[1]
                    patch_list[1] = z[1]
                    continue
                # TRAINING DATA IS COMPARED WITH TEST DATA
                if x3 < distance:
                    x3 = distance
                    patch_list[3] = patch_list[2]
                    patch_list[2] = z[1]
                    continue
                # TRAINING DATA IS COMPARED WITH TEST DATA
                if x4 < distance:
                    x4 = distance
                    patch_list[4] = patch_list[3]
                    patch_list[3] = z[1]
                    continue
                # TRAINING DATA IS COMPARED WITH TEST DATA
                if x5 < distance:
                    min5 = distance
                    patch_list[5] = patch_list[4]
                    patch_list[4] = z[1]
                    continue
                # TRAINING DATA IS COMPARED WITH TEST DATA
                if x6 < distance:
                    x6 = distance
                    patch_list[5] = z[1]
                    continue

                # THIS LOOP CAPTURE THE MIDDLE COLOR OF THE 6 PIXELS
            for a in range(0, len(patch_list)):
                x = patch_list[a][1]
                y = patch_list[a][0]

                # THE PATCHES WILL BE REPLACED INTO THE COLORS THAT REPRESENT THEM
                patch_list[a] = cluster_list[y][x].classifier

            if mode(patch_list) != 0:
                duplicate[i][j] = centers[mode(patch_list)]

            duplicate[i][j] = centers[patch_list[random.randint(0, len(patch_list) - 1)]]

    plt.imshow(colored_left)
    plt.show()

    plt.imshow(duplicate)
    plt.show()


# k-NN CLASSIFICATION FOR PRE-CLUSTERED COLORS
def k_classification(arr):
    a = []
    # CREATE 5 ARBITRARY POINTS
    for i in range(5):
        a.append(list(arr[random.randint(0, len(arr) - 1)][random.randint(0, len(arr[0]) - 1)]))

    x = 0

    # WIll REFORMAT THE IMAGE ARRAY
    cluster_list = np.empty((len(arr), len(arr[0])), dtype=object)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            temp = Cell(-1, arr[i][j][0], arr[i][j][1], arr[i][j][2], )
            cluster_list[i][j] = temp

    while x != 15:
        arr2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        counter = [0, 0, 0, 0, 0]
        x = 0

        # DOUBLE FOR LOOP TO ITERATE THROUGH EACH PIXEL
        for y in range(len(arr)):
            for x in range(len(arr[0])):
                # print("x", x, "y", y)
                min_distance = 3000
                new_cluster = -1

                for j in range(5):
                    # UPDATE DISTANCE WITH EUCLIDEAN DISTANCE
                    updated = np.linalg.norm((arr[y][x] - a[j]))
                    # IF THE LATEST DISTANCE IS LESS THAN MINIMUM DISTANCE OF 3000
                    if updated < min_distance:
                        # MINIMUM DISTANCE IS NOW THE LATEST DISTANCE
                        min_distance = updated
                        new_cluster = j

                # THE CLASSIFIER CHANGES TO THE NEW CLUSTER
                cluster_list[y][x].classifier = new_cluster
                # ADD THAT NEW CLUSTER TO THE CLUSTER LIST VALUES
                arr2[new_cluster][0] += cluster_list[y][x].red
                arr2[new_cluster][1] += cluster_list[y][x].green
                arr2[new_cluster][2] += cluster_list[y][x].blue
                arr2[new_cluster] += 1

        for i in range(5):
            for j in range(3):
                if counter[i] != 0:
                    avg = int(arr2[i][j] / counter[j])
                else:
                    avg = 0
                if abs(avg - a[i][j]) > 5:
                    a[i][j] = avg
                else:
                    x += 1

    return a, cluster_list


# TURNING AN IMAGE TO A GRAYSCALE
def convert_to_grayscale(jpg):
    # COLOR IN EACH PIXEL
    jpg_copy = np.copy(jpg)
    jpg = jpg.tolist()
    for i in range(len(jpg)):
        for j in range(len(jpg[i])):
            # UPDATING PROPER RED, GREEN, AND BLUE VALUES WITH
            jpg[i][j] = 0.21 * jpg[i][j][0] + 0.72 * jpg[i][j][1] + 0.07 * jpg[i][j][2]
            jpg_copy[i][j] = 0.21 * jpg[i][j][0] + 0.72 * jpg[i][j][1] + 0.07 * jpg[i][j][2]

    return np.array(jpg), jpg_copy
