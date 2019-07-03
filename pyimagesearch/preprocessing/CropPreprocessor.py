# import the necessary packages
import numpy as np
import cv2


class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        # store the target image width, height, whether or not
        # horizontal flips should be included, along with the
        # interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

        # The only required arguments are the target width and height of each cropped region. We can also optionally
        # specify whether horizontal flipping should be applied (defaults to True) along with the interpolation
        # algorithm OpenCV will use for resizing. These arguments are all stored inside the class for use within the
        # preprocess method.
    def preprocess(self, image):
        # initialize the list of crops
        crops = []

        # grab the width and height of the image then use these
        # dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]]

        # compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # The preprocess method requires only a single argument – the image which we are going to apply
        # over-sampling. We grab the width and height of the input image on Line 21, which then allows us to compute
        # the (x;y)-coordinates of the four corners (top-left, top-right, bottom-right, bottom-left, respectively) on
        # Lines 22-26. The center crop of the image is then computed on Lines 29 and 30, then added to the list of
        # coords on Line 31.

        # loop over the coordinates, extract each of the crops,
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        # On Line 35 we loop over each of the starting and ending (x;y)-coordinates of the rectangular crops. Line 36
        # extracts the crop via NumPy array slicing which we then resize on Line 37 to ensure the target width and
        # height dimensions are met. The crop is the added to the crops list. In the case that horizontal mirrors are
        # to be computed, we can flip each of the five original crops, leaving us with ten crops overall:

        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

            # return the set of crops
        return np.array(crops)
