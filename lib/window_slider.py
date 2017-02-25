import lib.config as cfg
import numpy as np


class WindowSlider:

    def __init__(self):
        self.window_configs = cfg.get_window_configs()
        self.detection_focus_config = cfg.get_detection_focus_config()

    def slide_window(self, img, detection_boxes=[]):
        windows = []
        for config in self.window_configs:
            standard_windows = self.__slide_window(img, **config)
            windows.extend(standard_windows)

        offset_fraction = self.detection_focus_config['offset_fraction']
        dimensions = self.detection_focus_config['dimensions']
        overlap = self.detection_focus_config['overlap']

        # Perform concentrated search with extra windows around specified bounding boxes e.g. from previous frame
        for box in detection_boxes:
            xmin, ymin, xmax, ymax = np.ravel(box)
            dominant_dim = max(xmax-xmin, ymax-ymin)
            for dim in dimensions:
                offset = int(dominant_dim*offset_fraction)
                focus_windows = self.__slide_window(img, xy_window=(dim, dim), xy_overlap=(overlap, overlap),
                                              x_start_stop=[xmin-offset, xmax+offset],
                                              y_start_stop=[ymin-offset, ymax+offset])
                windows.extend(focus_windows)
        return windows

    def __slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

        # Set x/y start/stop to edge if None, or outside image edge
        if x_start_stop[0] is None or x_start_stop[0] < 0:
            x_start_stop[0] = 0
        if x_start_stop[1] is None or x_start_stop[1] > img.shape[1]:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None or y_start_stop[0] < 0:
            y_start_stop[0] = 0
        if y_start_stop[1] is None or y_start_stop[1] > img.shape[0]:
            y_start_stop[1] = img.shape[0]

        x_width = x_start_stop[1] - x_start_stop[0]
        y_height = y_start_stop[1] - y_start_stop[0]
        x_step_size = int(xy_window[0] * (1 - xy_overlap[0]))
        y_step_size = int(xy_window[1] * (1 - xy_overlap[1]))
        x_nb_windows = int((x_width - xy_window[0]) / x_step_size + 1)
        y_nb_windows = int((y_height - xy_window[1]) / y_step_size + 1)

        windows = []
        for x_i in range(x_nb_windows):
            for y_j in range(y_nb_windows):
                top_left = (x_start_stop[0] + x_i * x_step_size, y_start_stop[0] + y_j * y_step_size)
                bottom_right = (top_left[0] + xy_window[0], top_left[1] + xy_window[1])
                windows.append((top_left, bottom_right))
        return windows


def main():
    pass


if __name__ == '__main__':
    main()
