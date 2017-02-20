import numpy as np
from scipy.ndimage.measurements import label


class HeatMapper:

    def __init__(self, image_shape):
        self.nb_frames = 6
        self.hot_windows = []
        self.image_shape = image_shape
        self.heatmap_stack = np.tile(np.zeros(image_shape), (self.nb_frames, 1, 1))
        self.threshold = 8

    def add_hot_windows(self, windows):

        heatmap = np.zeros(self.image_shape)

        for window in windows:
            x_min, x_max, y_min, y_max = window[0][0], window[1][0], window[0][1], window[1][1]
            # Add += 1 for all pixels inside each window
            heatmap[y_min:y_max, x_min:x_max] += 1

        self.heatmap_stack = np.insert(self.heatmap_stack, 0, heatmap, axis=0)
        self.heatmap_stack = np.delete(self.heatmap_stack, -1, axis=0)

        heatmap = np.sum(self.heatmap_stack, axis=0)

        heatmap = self.apply_threshold(heatmap, self.threshold)

        labels = label(heatmap)
        bboxes = self.get_labeled_bboxes(labels)

        # Return updated heatmap
        return bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

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
