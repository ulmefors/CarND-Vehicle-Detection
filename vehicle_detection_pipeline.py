import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from lib.window_slider import WindowSlider
import lib.data_reader as data_reader
from lib.feature_extractor import FeatureExtractor
from lib.data_preprocessor import  PreProcessor
from lib.classifier import Classifier


class Pipeline:

    def __init__(self):
        self.classifier = Classifier()
        self.feature_extractor = FeatureExtractor()
        self.window_slider = WindowSlider()
        self.preprocessor = PreProcessor()

    def train_classifier(self):
        car_files, non_car_files = data_reader.read_data(nb_data=250, small_sample=True)

        car_features = self.feature_extractor.extract_features_from_files(car_files)
        non_car_features = self.feature_extractor.extract_features_from_files(non_car_files)

        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(car_features, non_car_features)

        self.classifier.train(X_train, X_test, y_train, y_test)

    def run_pipeline(self, image):

        windows = self.window_slider.slide_window(image)

        result = 0
        count = 0
        for window in windows:
            x_min, x_max, y_min, y_max = window[0][0], window[1][0], window[0][1], window[1][1]

            resized = cv2.resize(image[y_min:y_max, x_min:x_max], (64, 64))

            # Get features
            features = self.feature_extractor.extract_features_from_image(resized)
            # Scale
            scaled_features = self.preprocessor.scale_features(features)

            fit = self.classifier.predict(scaled_features)

            count += 1
            if fit:
                result +=1
                cv2.rectangle(image, window[0], window[1], (255,)*3)

        plt.imshow(image)
        plt.show()


def main():
    pipeline = Pipeline()
    pipeline.train_classifier()
    pipeline.run_pipeline(mpimg.imread('test_images/test1.jpg'))


if __name__ == "__main__":
    main()
