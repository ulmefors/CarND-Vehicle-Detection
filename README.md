
# Vehicle Detection
This project is a submission for the Udacity [Self-Driving Car Engineer Nanodegree](http://udacity.com/drive).
Vehicles in a video are detected and tracked using computer vision (OpenCV) and Machine Learning techniques.

[//]: # (Image References)
[image1]: ./output_images/org_vehicle/412.jpeg "Vehicle"
[image2]: ./output_images/org_vehicle/529.jpeg "Vehicle"
[image3]: ./output_images/org_vehicle/703.jpeg "Vehicle"
[image4]: ./output_images/org_non_vehicle/extra108.jpeg "Non-vehicle"
[image5]: ./output_images/org_non_vehicle/extra198.jpeg "Non-vehicle"
[image6]: ./output_images/org_non_vehicle/extra283_64.jpeg "Non-vehicle"
[image7]: ./output_images/
[image8]: ./output_images/
[image9]: ./output_images/
[image10]: ./output_images/
[image11]: ./output_images/
[image12]: ./output_images/
[image13]: ./output_images/
[image14]: ./output_images/
[image15]: ./output_images/

[video1]: ./project_video.mp4

## Read data
Labeled images in resolution 64 x 64 pixels with the classes `vehicle` and `non-vehicle` are loaded.  Examples for the both classes below.

Vehicle | Vehicle | Vehicle
:---: | :---: | :---:
![alt_text][image1] | ![alt_text][image2] |![alt_text][image3]

Non-vehicle | Non-vehicle | Non-vehicle
:---: | :---: | :---:
![alt_text][image4] | ![alt_text][image5] |![alt_text][image6]

Some of the images labeled `non-vehicle` do in fact contain vehicles. In order to qualify as a `vehicle` image the entire vehicle must be visible within the bounds of the image and, in addition, make a close fit to the image borders.

Download the labeled dataset for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) respectively. The data is composed from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

[file: `data_reader.py`](./lib/data_reader.py)
## Feature extraction

### Color Histograms

### Color space

### Histogram of Oriented Gradients (HOG)
HOG features are extracted using `hog()` from `scikit-image`.

```python
from skimage.feature import hog

hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
    visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
```

A number of experiments were run to select HOG color space, image channels, and optimization parameters. The best performance in these benchmark tests was achieved using all three channels in the `HSV` color space in combination with the following parameters for feature extraction using `hog()`

    orientations=12, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True

For the sake a brevity in this document, the full methodology can be found in [Appendix](./docs/APPENDIX.md).

## Normalization and Classification

HOG features are extracted along with spatial features and color histogram features. The features are normalized together using `StandardScaler` and split into training/test set.

```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    self.X_scaler = StandardScaler().fit(X)
    scaled_X = self.X_scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2)
```
[file: `preprocessor.py`](./lib/preprocessor.py)

The normalized data is used to train and test the classifier, in this case a Linear Support Vector Classifier.

```python
    from sklearn.svm import LinearSVC

    def train(self, X_train, X_test, y_train, y_test):
        self.clf = LinearSVC()
        self.clf.fit(X_train, y_train)
        accuracy_test = self.clf.score(X_test, y_test)
```

[file: `classifier.py`](./lib/classifier.py)

## Sliding windows
_A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen._

_Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)_

[file: `window_slider.py`](./lib/window_slider.py)
## Video implementation
_The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video._

### Heat map
_A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur._

[file: `heat_mapper.py`](./lib/heat_mapper.py)
## Discussion
_Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail._
