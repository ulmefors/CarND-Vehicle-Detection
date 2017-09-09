# Avoid bug: Python is not installed as a framework (OS X)
import matplotlib
matplotlib.use('TkAgg')

import os
import cv2
import glob
import matplotlib.image as mpimg
import numpy as np
import lib.data_reader as data_reader
import lib.config as config
import pickle
from moviepy.editor import VideoFileClip
from lib.window_slider import WindowSlider
from lib.feature_extractor import FeatureExtractor
from lib.preprocessor import PreProcessor
from lib.classifier import Classifier
from lib.heat_mapper import HeatMapper

class Pipeline:

    model_file = './model.p'

    def __init__(self, clf=None, scaler=None, video=False):
        hm_cfg = config.get_heatmap_config(video)
        self.heat_mapper = HeatMapper((config.y_height, config.x_width), hm_cfg['nb_frames'], hm_cfg['threshold'])
        self.classifier = Classifier(clf=clf)
        self.feature_extractor = FeatureExtractor(config.get_feature_config(), config.get_color_space())
        self.window_slider = WindowSlider()
        self.preprocessor = PreProcessor(scaler=scaler)
        self.detection_boxes = []

    def load_features(self):
        car_files, non_car_files = data_reader.read_data(small_sample=False)
        print('Extract features from {0} cars and {1} non-cars'.format(len(car_files), len(non_car_files)))
        car_features = self.feature_extractor.extract_features_from_files(car_files)
        non_car_features = self.feature_extractor.extract_features_from_files(non_car_files)
        return car_features, non_car_features

    def train_classifier(self, car_features, non_car_features):
        # Scale features. Split into train/test sets. Train Classifier. Save model (classifier + scaler) to disk.
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(car_features, non_car_features)
        self.classifier.train(X_train, X_test, y_train, y_test)
        pickle.dump({'scaler': self.preprocessor.get_scaler(), 'clf': self.classifier.get_classifier()},
                    open(Pipeline.model_file, 'wb'))

    def run_pipeline(self, image):
        # Choose windows where vehicles will be searched
        windows = self.window_slider.slide_window(image, detection_boxes=self.detection_boxes)

        # Create detection boxes based on SVM predictions from content of each window
        hot_windows = []
        for window in windows:
            x_min, y_min, x_max, y_max = np.ravel(window)
            resized = cv2.resize(image[y_min:y_max, x_min:x_max], (64, 64))
            features = self.feature_extractor.extract_features_from_image(resized)
            scaled_features = self.preprocessor.scale_features(features)
            if self.classifier.predict(scaled_features):
                hot_windows.append(window)
        self.detection_boxes = self.heat_mapper.add_hot_windows(hot_windows)

        # Draw detection boxes around every car
        for box in self.detection_boxes:
            cv2.rectangle(image, box[0], box[1], (0, 127, 255), 4)

        # Sign the art :)
        cv2.putText(image, 'Marcus Ulmefors, Udacity Self-Driving Car Nanodegree', (415, 710),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,) * 3, 1, cv2.LINE_AA)
        return image


def main():
    # Load model from disk instead of retraining classifier
    load_classifier_and_scaler = True
    # Run video instead of single image
    video = True

    if load_classifier_and_scaler:
        print('Loading Classifier and Scaler from disk')
        model = pickle.load(open(Pipeline.model_file, 'rb'))
        pipeline = Pipeline(clf=model['clf'], scaler=model['scaler'], video=video)
    else:
        pipeline = Pipeline(video=video)
        car_features, non_car_features = pipeline.load_features()
        pipeline.train_classifier(car_features, non_car_features)

    # Specify inputs and outputs
    video_file = 'project_video.mp4'
    video_output_dir = 'bin/'

    if video:
        # Create output folder if missing
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # Create video with detected vehicles
        output = video_output_dir + video_file
        input_clip = VideoFileClip(video_file)
        output_clip = input_clip.fl_image(pipeline.run_pipeline)
        output_clip.write_videofile(output, audio=False)
    else:
        # Create image with detected vehicles
        image_files = glob.glob('test_images/test*.jpg')
        print('Detect vehicles in {0} image(s)'.format(len(image_files)))
        for image_file in image_files:
            image = mpimg.imread(image_file)
            result = pipeline.run_pipeline(image)
            cv2.imwrite(image_file.replace('test_images', 'bin'), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    main()
