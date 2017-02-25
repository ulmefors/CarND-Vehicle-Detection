import numpy as np
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog


class FeatureExtractor:
    """
    Extract features from images. Code developed with inspiration from Udacity Self-Driving Car Nanodegree.

    """
    def __init__(self, feature_config, color_space):
        self.feature_config = feature_config
        self.color_space = color_space

    SPATIAL_SIZE = (64, 64)
    HIST_BINS = 32

    def bin_spatial(self, image, size=SPATIAL_SIZE):
        # Create feature vector using pixel values
        features = cv2.resize(image, size).ravel()
        return features

    # Define a function to compute color histogram features
    def color_hist(self, image, nbins=HIST_BINS, bins_range=(0, 256)):
        channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)

        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        return hist_features

    def get_hog_features(self, image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

        args = {
            'orientations': orient,
            'pixels_per_cell': (pix_per_cell,) * 2,
            'cells_per_block': (cell_per_block,) * 2,
            'transform_sqrt': True,
            'visualise': vis,
            'feature_vector': feature_vec
        }

        if vis:
            features, hog_image = hog(image, **args)
            return features, hog_image

        else:
            features = hog(image, **args)
            return features

    def __extract_features(self, feature_image, spatial_size=SPATIAL_SIZE,
                           hist_bins=HIST_BINS, orient=9, pix_per_cell=8, cell_per_block=2,
                           hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
        file_features = []

        if spatial_feat:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat:
            args_dict = {
                'vis': False,
                'feature_vec': True
            }

            args_list = [orient, pix_per_cell, cell_per_block]

            if hog_channel == 'ALL':
                channels = range(feature_image.shape[2])
            else:
                channels = [hog_channel]

            hog_features = []
            for channel in channels:
                hog_features.append(self.get_hog_features(feature_image[:, :, channel], *args_list, **args_dict))

            # Unroll features if more than 1 dimension
            # if len(np.array(hog_features).shape) > 1:
            if np.ndim(hog_features) > 1:
                hog_features = np.ravel(hog_features)

            file_features.append(hog_features)

        return np.concatenate(file_features)

    def extract_features_from_image(self, image):
        # Convert color space from RGB to configuration color space (if required)
        feature_image = self.convert_color_space(image)
        features = self.__extract_features(feature_image, **self.feature_config)
        return features

    def extract_features_from_files(self, image_files):
        features = []
        for img_file in image_files:
            # Read RGB version from disk
            image = mpimg.imread(img_file)
            # Scale values to make png and jpeg compatible
            if img_file.endswith('png'):
                image = (image * 255).astype(np.uint8)
            file_features = self.extract_features_from_image(image)
            features.append(file_features)

        return features

    def convert_color_space(self, image):
        """ Converts color space in accordance with configuration

        :param image: image in RGB color space
        :return: image in configured color space
        """
        # Load color space
        color_space = self.color_space

        # Convert from RBG to chosen color space
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        return feature_image


def main():
    pass


if __name__ == '__main__':
    main()
