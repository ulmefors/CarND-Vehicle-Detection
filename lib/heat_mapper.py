import numpy as np
from scipy.ndimage.measurements import label


class HeatMapper:

    def __init__(self, image_shape):
        self.hot_windows = []
        self.heat_map = np.zeros(image_shape)
        self.threshold = 2

    def add_hot_windows(self, windows):

        heat_map = self.heat_map

        for window in windows:
            x_min, x_max, y_min, y_max = window[0][0], window[1][0], window[0][1], window[1][1]
            # Add += 1 for all pixels inside each window
            heat_map[y_min:y_max, x_min:x_max] += 1

        heat_map -= 1
        heat_map = self.apply_threshold(heat_map, self.threshold)

        labels = label(heat_map)
        bboxes = self.get_labeled_bboxes(labels)

        self.heat_map = heat_map

        # Return updated heatmap
        return bboxes

    def apply_threshold(self, heat_map, threshold):
        # Zero out pixels below the threshold
        heat_map[heat_map <= threshold] = 0
        # Return thresholded map
        return heat_map

    def get_labeled_bboxes(self, labels):
        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

            bboxes.append(bbox)

        return bboxes


def main():
    pass


if __name__ == "__main__":
    main()
