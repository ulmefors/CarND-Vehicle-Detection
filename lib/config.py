x_width = 1280
y_height = 720


def get_color_space():
    # 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'
    return 'YUV'


def get_feature_config():
    feature_config = {
        'spatial_size': (64, 64),
        'hist_bins': 32,
        'orient': 9,
        'pix_per_cell': 8,
        'cell_per_block': 2,
        'hog_channel': 1,
        'spatial_feat': True,
        'hist_feat': True,
        'hog_feat': True
    }
    return feature_config


def get_window_config():

    window_config = {
        'x_start_stop': [0, x_width],
        'y_start_stop': [int(y_height/2), y_height],
        'xy_window': (128, 128),
        'xy_overlap': (5/8, 5/8)
    }

    return window_config


def main():
    pass


if __name__ == "__main__":
    main()
