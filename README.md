
# Vehicle Detection
This project is Project 5 of the [Udacity Self-Driving Car Engineer Nanodegree](http://udacity.com/drive).
Vehicles on video are detected and tracked using computer vision (OpenCV) and Machine Learning techniques such as Support Vector Machines.

[//]: # (Image References)
[image1]: ./output_images/org_vehicle/412.jpeg 'Vehicle'
[image2]: ./output_images/org_vehicle/529.jpeg 'Vehicle'
[image3]: ./output_images/org_vehicle/703.jpeg 'Vehicle'
[image4]: ./output_images/org_non_vehicle/extra108.jpeg 'Non-vehicle'
[image5]: ./output_images/org_non_vehicle/extra198.jpeg 'Non-vehicle'
[image6]: ./output_images/org_non_vehicle/extra283_64.jpeg 'Non-vehicle'
[image7]: ./output_images/sliding_windows/windows-128.jpg 'Windows 128px'
[image8]: ./output_images/sliding_windows/windows-96.jpg 'Windows 96px'
[image9]: ./output_images/sliding_windows/windows-64.jpg 'Windows 64px'
[image10]: ./output_images/sliding_windows/hot-windows.jpg 'Hot windows'
[image11]: ./output_images/sliding_windows/heatmap.jpg 'Heatmap'
[image12]: ./output_images/sliding_windows/heatmap-thresholded-detection.jpg 'Heatmap Thresholded Detection'
[image13]: ./output_images/detection_focus/detection-focus-search.jpg
[image14]: ./output_images/detection_focus/hot-windows-focus.jpg
[image15]: ./output_images/detection_focus/hot-windows-standard.jpg
[image16]: ./output_images/hog/301_hsv_2.jpeg
[image17]: ./output_images/hog/301_hog_hsv_2.jpeg
[image18]: ./output_images/hog/image0176_hsv_2.jpeg
[image19]: ./output_images/hog/image0176_hog_hsv_2.jpeg
[image22]: ./output_images/

[video1]: ./output_videos/project_video.mp4 'Project video'

Main script: `vehicle_detection_pipeline.py`

Output video: `output_videos/project_video.mp4` [Link](./output_videos/project_video.mp4)

1. Describe pipeline
1. Cover rubric points
1. False positives
1. Describe label-function combining blobs into vehicles + visualization
3. Heatmap to Detection box
4. HOG visualization
------

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
---

## Read data
Labeled images in resolution 64 x 64 pixels with the classes `vehicle` and `non-vehicle` are loaded.  Examples for the both classes below.

Vehicle | Vehicle | Vehicle
:---: | :---: | :---:
![alt_text][image1] | ![alt_text][image2] |![alt_text][image3]

Non-vehicle | Non-vehicle | Non-vehicle
:---: | :---: | :---:
![alt_text][image4] | ![alt_text][image5] |![alt_text][image6]

Some of the images labeled `non-vehicle` do in fact contain vehicles. In order to qualify as a `vehicle` image the entire vehicle must be visible within the bounds of the image and, in addition, make a close fit to the image borders. Alternatively, this could also be an example of data set errors.

Download the labeled dataset for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) respectively. The data is composed from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

[file: `data_reader.py`](./lib/data_reader.py)
## Feature extraction

### Histogram of Oriented Gradients (HOG)
HOG features are extracted using `hog()` from `scikit-image`.

```python
from skimage.feature import hog

hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
    visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
```

A number of experiments were run to select HOG color space, image channels, and optimization parameters. The best performance in these benchmark tests was achieved using all three channels in the `HSV` color space in combination with the following parameters for feature extraction using `hog()`

    orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2)

For the sake a brevity in this document, the full methodology can be found in [Appendix](./docs/APPENDIX.md).

Examples of visualized HOG features for vehicle and non-vehicle classes are presented below. Only the HSV V-channel is used for visualization whereas all three HSV channels are included in the training and prediction.

| HSV V-Channel | HSV V-CH HOG|  HSV V-Channel | HSV V-CH HOG |
| :-----: | ----- | ----- | ----- |
| ![alt_text][image16] | ![alt_text][image17] | ![alt_text][image18] | ![alt_text][image19] |

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
In order to find vehicles, we scan each image (video frame) with vehicle-sized rectangular windows. The size and position of the window must be sufficiently similar to the vehicle in order to generate a match. Three sizes (128, 96, and 64 pixels) are used and snapshots are taken at regular intervals such that the windows overlap. Smaller windows scan near the center of the image (close to the horizon), and larger boxes near the base of the image (close to the det`ecting car). We take care to avoid sliding windows in areas where we do not expect to find vehicles in order to save computing resources and reduce number of false positives.

| 128x128 | 96x96 | 64x64 | 
| :-----: | ----- | ----- |
| ![alt_text][image7] | ![alt_text][image8] | ![alt_text][image9] |

The window slider code takes as input the scan area coordinates `x_start_stop, y_start_stop`, window size `xy_window`and window overlap fraction `xy_overlap`. The number of windows and their corresponding coordinates are calculated.
The calculated windows are defined by the two points top_left and bottom_right i.e. `((x_min, y_min), (x_max, y_max))`.

```python
def __slide_window(x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    
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
```
[file: `window_slider.py`](./lib/window_slider.py)

The content of each window is resized (64, 64) to match the classifier. Features are extracted and values normalized before prediction is performed. If prediction is positive, he window is considered `hot` and can be added to the heatmap.

```python
hot_windows = []
for window in windows:
    x_min, y_min, x_max, y_max = np.ravel(window)
    resized = cv2.resize(image[y_min:y_max, x_min:x_max], (64, 64))
    features = self.feature_extractor.extract_features_from_image(resized)
    scaled_features = self.preprocessor.scale_features(features)
    if self.classifier.predict(scaled_features):
        hot_windows.append(window)
    self.heat_mapper.add_hot_windows(hot_windows)
```
[file: `vehicle_detection_pipeline.py`](./lib/vehicle_detection_pipeline.py)

### Single frame heatmap
The `hot_windows` that the SVM predicts as vehicles are added to a heatmap where the encompassed pixels contribute by increasing the heat value. The heatmap can be filtered so that all values below a certain threshold are set to zero.

```python
def add_hot_windows(self, hot_windows):
    heatmap = np.zeros(self.image_shape)
    for window in hot_windows:
        x_min, y_min, x_max, y_max = np.ravel(window)
        heatmap[y_min:y_max, x_min:x_max] += 1

def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap
``` 
[file: `heat_mapper.py`](./lib/heat_mapper.py)

| Hot windows | Heatmap | Thresholded detection | 
| :-----: | ----- | ----- |
| ![alt_text][image10] | ![alt_text][image11] | ![alt_text][image12] |


Pixels in the thresholded heatmap pertaining to vehicles are labelled and grouped into separate units using `label()`. For each of the areas the top-left pixel and the bottom-right pixel together define the detection box. An optional filter is added that can be used if false positives are more likely to produce thin detection boxes.

```python
from scipy.ndimage.measurements import label

def get_detection_boxes(heatmap, filter_size=(0, 0)):
    detection_boxes = []
    labels = label(heatmap)

    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        x_min, y_min, x_max, y_max = np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)

        if x_max - x_min > filter_size[0] and y_max - y_min > filter_size[1]:
            detection_box = ((x_min, y_min), (x_max, y_max))
            detection_boxes.append(detection_box)
    return detection_boxes
```
[file: `heat_mapper.py`](./lib/heat_mapper.py)

_Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)_

## Video implementation
_The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video._

During processing of a video stream we have the advantage of being able to use previous frames in addition to the current frame. One important benefit of this opportunity is that false positives can be reduced since they are often classification errors and are unlikely to remain with high confidence in multiple sequential frames. Frame-to-frame optimization is achieved by _Heatmap averaging_ and _Detection box window focus_.

### Heatmap averaging
Heatmap averaging is achieved by adding the current frame heatmap to a stack of previous heatmaps. The stack depth (and thus detection inertia/stability) is configurable by parameter `nb_frames`. A longer history reduces risk of losing track of a detected vehicle but also slows down initial detection. 

```python
self.heatmap_stack = np.insert(self.heatmap_stack, 0, heatmap, axis=0)
if self.heatmap_stack.shape[0] > self.nb_frames:
    self.heatmap_stack = np.delete(self.heatmap_stack, -1, axis=0)
heatmap_mean = np.mean(self.heatmap_stack, axis=0)
```
[file: `heat_mapper.py`](./lib/heat_mapper.py)

### Detection box window focus
In order to avoid losing track of a detected vehicle we increase search intensity in the area around the detection box defined in the previous frame. The higher density of search windows close to the previous detection will increase the number of true positive vehicle predictions in the subsequent frame. The increased rate of correct vehicle detection can also allow for increase in heatmap thresholding, potentially reducing number of false positives.

In the left-most image (frame _f<sub>n</sub>_ ) we see the blue detection box from frame _f<sub>n-1</sub>_ and the resulting extended search windows marked in red. The center image shows the hot windows resulting from the extended focus search and the right-most image presents the hot windows from the standard search in frame _f<sub>n</sub>_ without knowledge from the previous frame.

| Extended search around detection box | Hot windows extended | Hot windows standard |
|:---:|:---:|:---:|
| ![alt_text][image13] | ![alt_text][image14] | ![alt_text][image15] |
 
The difference in results comparing extended search (based on the detection from the previous frame) to standard search (from scratch) provides redundancy and stability to the detection algorithm.

```python
def slide_window(self, img, detection_boxes=[]):
    for box in detection_boxes:
        xmin, ymin, xmax, ymax = np.ravel(box)
        dominant_dim = max(xmax-xmin, ymax-ymin)
        for dim in dimensions:
            offset = int(dominant_dim*offset_fraction)
            focus_windows = self.__slide_window(img, xy_window=(dim, dim), xy_overlap=(overlap, overlap),
                                          x_start_stop=[xmin-offset, xmax+offset],
                                          y_start_stop=[ymin-offset, ymax+offset])
            windows.extend(focus_windows)
```
[file: `window_slider.py`](./lib/window_slider.py)


### Heat map
_A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur._

[file: `heat_mapper.py`](./lib/heat_mapper.py)
## Discussion
_Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail._

The first problem to address is speed. The code runs on a single CPU-core and results are measured on a Macbook Pro 15" Mid 2015 in its basic configuration of i7 quad-core Haswell CPU [(full technical specs)](https://support.apple.com/kb/SP719). Features are extracted for each sliding window individually. For each window we have the following breakdown.
 
|   | Spatial | Color histogram | HOG | Total |
| :----- | :-----: | :-----: | :-----: | -----: |
| Feature count  | 12288 | 96 | 7056 | 19440 |
| Extraction time (ms) | 0.006 | 0.4 | 3 | 3.41 |

In a basic frame we have 644 windows, resulting in total pipeline time 2.43 s, of which 2.23 s is feature extraction. Performing the feature extraction once per frame and sliding windows over the features would increase speed greatly. 

It would be interesting to try using a CNN-classifier instead of a SVM to see if performance improves further, both accuracy and speed. 

Accuracy of >99% on test set might give room for faster operation