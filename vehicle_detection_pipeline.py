import os
import cv2
import glob
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import lib.data_reader as data_reader
import lib.config as config
from lib.window_slider import WindowSlider
from lib.feature_extractor import FeatureExtractor
from lib.data_preprocessor import PreProcessor
from lib.classifier import Classifier
from lib.heat_mapper import HeatMapper
from moviepy.editor import VideoFileClip
import pickle


class Pipeline:

    def __init__(self, clf=None, scaler=None):
        self.classifier = Classifier(clf=clf)
        self.feature_extractor = FeatureExtractor(config.get_feature_config(), config.get_color_space())
        self.window_slider = WindowSlider()
        self.preprocessor = PreProcessor(scaler=scaler)
        self.heat_mapper = HeatMapper((config.y_height, config.x_width))
        self.bboxes = []
        self.save_data = False
        self.load_data = False

    def load_features(self):
        car_files, non_car_files = data_reader.read_data(small_sample=False)

        print('Extract features from {0} cars and {1} non-cars'.format(len(car_files), len(non_car_files)))
        car_features = self.feature_extractor.extract_features_from_files(car_files)
        non_car_features = self.feature_extractor.extract_features_from_files(non_car_files)

        return car_features, non_car_features

    def train_classifier(self, car_features, non_car_features):

        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(car_features, non_car_features)

        accuracy_test = self.classifier.train(X_train, X_test, y_train, y_test)

        pickle.dump({'scaler': self.preprocessor.get_scaler()}, open("scaler.p", "wb"))
        pickle.dump({'clf': self.classifier.get_classifier()}, open("clf.p", "wb"))

        return accuracy_test

    def run_pipeline(self, image):

        windows = self.window_slider.slide_window(image, bounding_boxes=self.bboxes)

        hot_windows = []

        for window in windows:
            x_min, x_max, y_min, y_max = window[0][0], window[1][0], window[0][1], window[1][1]

            resized = cv2.resize(image[y_min:y_max, x_min:x_max], (64, 64))

            # Get features
            features = self.feature_extractor.extract_features_from_image(resized)

            # Scale
            scaled_features = self.preprocessor.scale_features(features)

            if self.classifier.predict(scaled_features):
                hot_windows.append(window)

        self.bboxes = self.heat_mapper.add_hot_windows(hot_windows)

        for bbox in self.bboxes:
            cv2.rectangle(image, bbox[0], bbox[1], (0, 127, 255), 4)

        return image


def main():
    if True:
        clf = pickle.load(open("clf.p", "rb"))
        scaler = pickle.load(open("scaler.p", "rb"))
        pipeline = Pipeline(clf=clf['clf'], scaler=scaler['scaler'])
    else:
        pipeline = Pipeline()
        car_features, non_car_features = pipeline.load_features()
        pipeline.train_classifier(car_features, non_car_features)

    # Run video or single image
    video = True

    # Specify inputs and outputs
    video_file = 'project_video'
    video_output_dir = 'bin/'

    # Plot image with detected lanes
    for image_file in glob.glob('test_images/test*.jpg'):
        image = mpimg.imread(image_file)
        result = pipeline.run_pipeline(image)
        cv2.imwrite(image_file.replace('test_images', 'bin'), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    if video:
        # Create output folder if missing
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # Create video with lane zone overlay
        output = video_output_dir + video_file + '.mp4'
        input_clip = VideoFileClip(video_file + '.mp4')
        output_clip = input_clip.fl_image(pipeline.run_pipeline)
        output_clip.write_videofile(output, audio=False)


if __name__ == "__main__":
    main()
