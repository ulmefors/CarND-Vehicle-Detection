x_width = 1280
y_height = 720


def get_color_space():
    # 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'
    return 'HSV'


def get_feature_config():
    feature_config = {
        'spatial_size': (64, 64),
        'hist_bins': 32,
        'orient': 12,
        'pix_per_cell': 8,
        'cell_per_block': 2,
        'hog_channel': 'ALL',
        'spatial_feat': True,
        'hist_feat': True,
        'hog_feat': True
    }
    return feature_config


def get_window_configs():
    overlap = 6/8
    window_configs = []

    window_configs.append({
        'x_start_stop': [int(x_width * 2.5 / 8), int(x_width * 8 / 8)],
        'y_start_stop': [int(y_height * 4.5 / 8), int(y_height * 6 / 8)],
        'xy_window': (64, 64),
        'xy_overlap': (overlap, overlap)
    })
    window_configs.append({
        'x_start_stop': [int(x_width * 2 / 8), int(x_width * 8 / 8)],
        'y_start_stop': [int(y_height * 4.5 / 8), int(y_height * 7 / 8)],
        'xy_window': (96, 96),
        'xy_overlap': (overlap, overlap)
    })
    window_configs.append({
        'x_start_stop': [int(x_width * 2 / 8), int(x_width * 8 / 8)],
        'y_start_stop': [int(y_height * 4.5 / 8), int(y_height * 8 / 8)],
        'xy_window': (128, 128),
        'xy_overlap': (overlap, overlap)
    })
    return window_configs


def get_bbox_focus_config():
    bbox_focus_config = {
        'overlap': 7/8,
        'dimensions': [64, 96, 128],
        'offset_fraction': 2/8
    }
    return bbox_focus_config


def main():
    pass


if __name__ == "__main__":
    main()
