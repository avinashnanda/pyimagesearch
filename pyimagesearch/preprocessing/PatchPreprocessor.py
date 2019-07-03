# import the necessary packages
from sklearn.feature_extraction.image import extract_patches_2d


class PatchPreprocessor:
    def __init__(self, width, height):
        # store the target width and height of the image
        self.width = width
        self.height = height

    def preprocess(self, image):
        # extract a random crop from the image with the target width
        # and height
        return extract_patches_2d(image, (self.height, self.width),
                                  max_patches=1)[0]

# Extracting a random patches of size self.width x self.height is easy using the extract_patches_2d function from the
# scikit-learn library. Given an input image, this function randomly extracts a patch from image. Here we supply
# max_patches=1, indicating that we only need a single random patch from the input image. The PatchPreprocessor class
# doesn’t seem like much, but it’s actually a very effective method to avoid overfitting by applying yet another
# layer of data augmentation.
