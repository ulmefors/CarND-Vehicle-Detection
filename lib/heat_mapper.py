import numpy as np
from scipy.ndimage.measurements import label


class HeatMapper:

    default_filter_size = (48, 48)

    def __init__(self, image_shape, nb_frames, threshold):
        self.nb_frames = nb_frames
        self.hot_windows = []
        self.image_shape = image_shape
        self.heatmap_stack = None
        self.threshold = threshold

    def add_hot_windows(self, windows):

        # Initialize heatmap and increase heat value for all pixels inside each window
        heatmap = np.zeros(self.image_shape)
        for window in windows:
            x_min, y_min, x_max, y_max = np.ravel(window)
            heatmap[y_min:y_max, x_min:x_max] += 1

        # Add heatmap to stack of heatmaps, one for each frame. Initialize stack if nonexistent.
        try:
            self.heatmap_stack = np.insert(self.heatmap_stack, 0, heatmap, axis=0)
        except ValueError:
            self.heatmap_stack = np.tile(heatmap, (1, 1, 1))
        # Limit stack depth (i.e. frame history) by removing the last entry
        if self.heatmap_stack.shape[0] > self.nb_frames:
            self.heatmap_stack = np.delete(self.heatmap_stack, -1, axis=0)

        # Calculate mean value of heatmap history, apply threshold, and return identified bounding boxes
        heatmap_mean = np.mean(self.heatmap_stack, axis=0)
        heatmap_thresholded = self.apply_threshold(heatmap_mean, self.threshold)
        detection_boxes = self.get_detection_boxes(heatmap_thresholded, filter_size=self.default_filter_size)
        return detection_boxes

    @staticmethod
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    @staticmethod
    def get_detection_boxes(heatmap, filter_size=(0, 0)):
        detection_boxes = []
        labels = label(heatmap)

        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max of x/y
            x_min, y_min, x_max, y_max = np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)

            # Filter bounding boxes to accept only boxes with minimum size
            if x_max - x_min > filter_size[0] and y_max - y_min > filter_size[1]:
                detection_box = ((x_min, y_min), (x_max, y_max))
                detection_boxes.append(detection_box)
        return detection_boxes


def main():
    pass


if __name__ == '__main__':
    main()
