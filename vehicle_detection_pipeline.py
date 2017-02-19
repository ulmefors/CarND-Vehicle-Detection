import os
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import lib.data_reader as data_reader
import lib.config as config
from lib.window_slider import WindowSlider
from lib.feature_extractor import FeatureExtractor
from lib.data_preprocessor import PreProcessor
from lib.classifier import Classifier
from lib.heat_mapper import HeatMapper
from moviepy.editor import VideoFileClip


class Pipeline:

    def __init__(self):
        self.classifier = Classifier()
        self.feature_extractor = FeatureExtractor(config.get_feature_config(), config.get_color_space())
        self.window_slider = WindowSlider()
        self.preprocessor = PreProcessor()
        self.heat_mapper = HeatMapper((config.y_height, config.x_width))
        self.count = 0

    def train_classifier(self):
        car_files, non_car_files = data_reader.read_data(small_sample=True)

        car_features = self.feature_extractor.extract_features_from_files(car_files)
        non_car_features = self.feature_extractor.extract_features_from_files(non_car_files)

        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(car_features, non_car_features)

        self.classifier.train(X_train, X_test, y_train, y_test)

    def run_pipeline(self, image):

        windows = self.window_slider.slide_window(image)

        hot_windows = []

        for window in windows:
            x_min, x_max, y_min, y_max = window[0][0], window[1][0], window[0][1], window[1][1]

            resized = cv2.resize(image[y_min:y_max, x_min:x_max], (64, 64))

            # Get features
            features = self.feature_extractor.extract_features_from_image(resized)

            # Scale
            scaled_features = self.preprocessor.scale_features(features)

            if self.classifier.predict(scaled_features):
                # cv2.imwrite('output_images/window_{}.jpeg'.format(self.count), cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                hot_windows.append(window)
            self.count += 1

        bboxes = self.heat_mapper.add_hot_windows(hot_windows)

        for bbox in hot_windows:
            cv2.rectangle(image, bbox[0], bbox[1], (255,) * 3, 6)

        return image


def main():
    pipeline = Pipeline()
    pipeline.train_classifier()

    # Run video or single image
    video = False

    # Specify inputs and outputs
    video_file = 'test_video'
    video_output_dir = 'output_videos/'

    # Create output folder if missing
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    if video:
        # Create video with lane zone overlay
        output = video_output_dir + video_file + '.mp4'
        input_clip = VideoFileClip(video_file + '.mp4')
        output_clip = input_clip.fl_image(pipeline.run_pipeline)
        output_clip.write_videofile(output, audio=False)

    # Plot image with detected lanes
    for image_file in glob.glob('test_images/test*.jpg'):
        image = mpimg.imread(image_file)
        result = pipeline.run_pipeline(image)
        cv2.imwrite(image_file.replace('test_images', 'output_images'), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
