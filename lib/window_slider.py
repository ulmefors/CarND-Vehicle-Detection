import lib.config as cfg
import numpy as np


class WindowSlider:

    def __init__(self):
        self.window_configs = cfg.get_window_configs()

    def slide_window(self, img, bounding_boxes=[]):
        window_list = []
        for config in self.window_configs:
            windows = self.__slide_window(img, **config)
            window_list.extend(windows)

        # Perform concentrated search with extra windows around specified bounding boxes e.g. from previous frame
        for bbox in bounding_boxes:
            xmin, ymin, xmax, ymax = np.ravel(bbox)
            for dim in [64, 96, 128]:
                offset = int(dim/2)
                windows = self.__slide_window(img, xy_window=(dim, dim), xy_overlap=(7/8, 7/8),
                                              x_start_stop=[xmin-offset, xmax+offset],
                                              y_start_stop=[ymin-offset, ymax+offset])
                window_list.extend(windows)

        return window_list

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

        window_list = []

        x_width = x_start_stop[1] - x_start_stop[0]
        y_height = y_start_stop[1] - y_start_stop[0]

        x_step_size = int(xy_window[0] * (1 - xy_overlap[0]))
        y_step_size = int(xy_window[1] * (1 - xy_overlap[1]))

        x_nb_windows = int((x_width - xy_window[0]) / x_step_size + 1)
        y_nb_windows = int((y_height - xy_window[1]) / y_step_size + 1)

        # List of windows
        for x_i in range(x_nb_windows):
            for y_j in range(y_nb_windows):
                top_left = (x_start_stop[0] + x_i * x_step_size, y_start_stop[0] + y_j * y_step_size)
                bottom_right = (top_left[0] + xy_window[0], top_left[1] + xy_window[1])
                window_list.append((top_left, bottom_right))

        return window_list


def main():
    pass


if __name__ == "__main__":
    main()
