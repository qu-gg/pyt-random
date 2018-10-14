"""
Simple CPPN Model that generates images in a while loop based
off of user input. This model is a simple 4 layer feed-forward net that
uses input about each pixel to output a pixel density.

Input to network: x, y, radius from center, latent vector, random cosine value
Output from network: single value between 0 and 1 representing pixel density (0: white, 1: black)

@author Ryan Missel
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import misc


BETA = np.cos(np.random.randint(0, 100))  # Value to act as a 'z' for each pixel


def create_array(img_w, img_h):
    """
    Creates the array for use as inputs into the CPPN.
    Each pixel in the array has a radius from the center, a latent vector,
    and a global cosine value
    :param img_w: Width of the image
    :param img_h: Height of the image
    :return: Array of pixels
    """
    array = [[[] for _ in range(img_h)] for _ in range(img_w)]
    for row in range(img_w):
        for pixel in range(img_h):
            W = np.random.normal(500)
            radius = np.sqrt(np.square(img_h//2 - row) + np.square(img_w//2 - pixel))
            array[row][pixel] = [row, pixel, radius, W, BETA]
    array = torch.Tensor(array)
    array = array.view(-1, 5)
    return array


class CPPN(nn.Module):
    """
    Simple feed-forward neural network that comprises of 4 fully connected
    layers with tanh as the activation function between them.

    Output is 1 value for B/W images and 3 for RGB images
    """
    def __init__(self):
        super(CPPN, self).__init__()
        self.dense1 = nn.Linear(5, 1000)
        self.dense2 = nn.Linear(1000, 1000)
        self.dense3 = nn.Linear(1000, 2000)
        self.dense4 = nn.Linear(2000, 1)

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = torch.tanh(self.dense4(x))
        return x


def generate_img(filename, img_w, img_h):
    """
    Function that handles creating an array and feeding it through the model
    in order to generate a single image. Reshapes the output array and saves it
    locally under the given filename
    :param filename: name of the output file
    :param img_w: image width
    :param img_h: image height
    :return: None
    """
    model = CPPN()
    array = create_array(img_w, img_h)
    image = model(array)
    image = image.view(img_w, img_h)
    image = image.detach().numpy()
    misc.imsave("results/{}.jpg".format(filename), image)


def main():
    """
    Main function of the program, handles the infinite loop and taking
    user input to generate CPPN images
    :return: None
    """
    while True:
        filename = input("Filename(q to quit): ")
        if filename == "q":
            break

        width = int(input("Image width: "))
        height = int(input("Image height: "))
        generate_img(filename, width, height)


main()
