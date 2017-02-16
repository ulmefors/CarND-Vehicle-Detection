import numpy as np
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog

'''
Get features from images. Code developed from Udacity Self-Driving Car Nanodegree.
'''

SPATIAL_SIZE = (64, 64)
HIST_BINS = 32


def bin_spatial(image, size=SPATIAL_SIZE):
    # Create feature vector
    features = cv2.resize(image, size).ravel()
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(image, nbins=HIST_BINS, bins_range=(0, 256)):
    channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)

    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


def get_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

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


def extract_features(imgs, color_space='RGB', spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):

    features = []

    for img_file in imgs:

        file_features = []

        image = mpimg.imread(img_file)

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

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat:
            args_dict = {
                'vis': False,
                'feature_vev': True
            }

            args_list = [orient, pix_per_cell, cell_per_block]

            if hog_channel == 'ALL':
                channels = range(feature_image.shape[2])
            else:
                channels = [hog_channel]

            hog_features = []
            for channel in channels:
                hog_features.append(get_hog_features(feature_image[:, :, channel], *args_list, **args_dict))

            # Unroll features is more than 1 dimension
            if len(np.array(hog_features).shape) > 1:
                hog_features = np.ravel(hog_features)

            file_features.append(hog_features)

        features.append(file_features)

    return features


def main():
    pass


if __name__ == "__main__":
    main()
