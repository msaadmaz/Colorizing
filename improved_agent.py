from PIL import Image
# from numpy.random import random
import random
import numpy as np


# Obtain the initial guess weights for Stochastic Gradient Descent (SGD)
# this will be a 1 dimensional weight vector of 9 weights
def get_random_weights():
    return np.random.rand(9)


# This is the improved agent that uses the sigmoid function on the model, and multiplies that by 255 (to give a color
# value). This is not linear regression as it is not mapping the points linearly/directly but instead uses the sigmoid
# function to keep everything out of 1, then multiplies that by 255 in order to receive a real color value between 0
# and 255
# f_w(x) = 255sigma(w dot x)
# The loss function is the squares loss function (f_w(xi) - yi)**2


def get_patches(image, grey_image, color):
    # make a list of patches
    patches = []

    # get the height and width of the image
    (width, height) = image.size

    for x in range(1, width // 2):
        for y in range(1, height - 1):
            patch = []
            middle_pixel = image.getpixel((x, y))[color] / 255

            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    pixel = grey_image.getpixel((i, j)) / 255
                    patch.append(pixel)

        patch.append(middle_pixel)
        patches.append(patch)

    return patches


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


# weights-alpha(colori - f_w(x))x
def run_new_model(weights, patch):
    # get the x
    x_vector = patch[:9]
    middle_color = patch[9]
    x_vector = np.array(x_vector)

    alpha = 0.00000000000000001
    new_weights = weights - alpha * (middle_color - 255 * sigmoid(np.dot(weights, x_vector))) * x_vector

    return new_weights


def regression_agent(image, image_pixels, grey_image, color_channel):
    # Get the initial guess weight vector
    weights = get_random_weights()

    # get the height and width of the image
    (width, height) = image.size

    # get the patches to train upon
    patches = get_patches(image, grey_image, color_channel)

    trials = 1000000

    for i in range(trials):

        patch = patches[random.randint(0, len(patches) - 1)]
        weights = run_new_model(weights, patch)
    print("done training")
    # recolor the image
    for x in range(width//2+1, width - 1):
        for y in range(1, height - 1):
            patch = []

            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    patch.append(grey_image.getpixel((i, j))/255)

            patch = np.array(patch)
            color_for_pixel = int(255*sigmoid(np.dot(weights, patch)))

            # grab the current color from the pixels of the image
            current_rgb = list(image_pixels[x, y])

            # set the new color for the pixels of that image
            current_rgb[color_channel] = color_for_pixel

            # add back in the color for that picture
            image_pixels[x, y] = (current_rgb[0], current_rgb[1], current_rgb[2])

    print("done coloring")


# This is the driver for the regression agent, as it needs to get the 3 values for each RGB channel
def regression_agent_driver():
    # open the image we want to train the data on
    image = Image.open('beach.jpg')

    # get the pixels
    image_pixels = image.load()

    # set the rgb values to xyz values to grey out desired side
    conversion_tuple = (0.21, 0.72, 0.07, 0)
    grey_image = image.convert('L', conversion_tuple)

    # call the regression agent for each channel of color, Red Green and Blue
    regression_agent(image, image_pixels, grey_image, 0)
    regression_agent(image, image_pixels, grey_image, 1)
    regression_agent(image, image_pixels, grey_image, 2)

    print("done with regression")

    image.show()
    print("done")
