
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

file: `data_reader.py`

## Feature extraction

### Color Histograms

### Color space

### Histogram of Oriented Gradients (HOG)
_Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why._

HOG features are extracted using `hog()` from scikit-image with the following signature.
 
```python
from skimage.feature import hog

hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
    visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)
```
The starting point for HOG parameters were `orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False, transform_sqrt=True, feature_vector=True, normalise=None` with `RGB` color space using all three color channels.

An experiment was run to select HOG color space and channel. The large data set (~9000 images per class) was used for training with only HOG features (no spatial bin, or color histogram). The remaining HOG parameters were set as `orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)`. The features were extracted and a linear Support Vector Machine was trained and evaluated at 80%/20% training/test set.

```python
def train(self, X_train, X_test, y_train, y_test, type='SVC'):
    from sklearn.svm import LinearSVC
    svc= LinearSVC()
    svc.fit(X_train, y_train)
    accuracy_test = svc.score(X_test, y_test)
```

The test accuracy was evaluated 10 times and the average value saved. For each iteration the data was normalized, randomized, and split anew. The result is presented in the table below.


| HOG CH| 0       | 1       | 2       | ALL     |
| :---  | :-----: | :-----: | :-----: | :-----: | 
| RGB   | 93.989% | 94.882% | 94.673% | 96.478% |
| HSV   | 90.507% | 89.659% | 94.713% | 97.917% |
| LUV(t)| 94.938% |     N/A |     N/A |     N/A |
| LUV(f)| 94.690% | 91.951% | 90.104% | 97.275% |
| HLS   | 90.287% | 94.794% | 89.274% | 97.990% |
| YUV(t)| 94.899% | 92.903% |     N/A |     N/A |
| YUV(f)| 95.037% | 92.784% | 90.521% | 98.021% |
| YCrCb | 94.947% | 93.136% | 91.059% | 98.114% |


_negative values_ for `LUV[1]` (-65, +171), `LUV[2]` (-134, 105), `YUV[2]`(-0.05 20) gives NaN after `hog()`.
sqrt_transform was turned on

HSV ALL 10 x, sqrt true

| Orient/CPB   | 2       | 3       | 4       | 
| :---  | :-----: | :-----: | :-----: |  
| 8     | 97.773% | 97.852% | 97.534% | 
| 10    | 98.066% | 98.086% | 97.790% | 
| 12    | 98.359% | 98.316% | 98.074% | 
| 14    | 98.328% | 98.308% | 98.162% | 

# Make sure to set sqrt flag correctly. Try all colors with 12 orientations.
Sqrt false/true, 12 orientations, ALL

| Color| !Sqrt | Sqrt |     
| :---  | :-----: | :-----: |
| RGB   | 96.568% | 96.585% |
| HSV   | 98.421% | 98.249% |
| LUV   | 97.075% |     N/A |     
| HLS   | 98.091% | 98.221% |     
| YUV   | 98.457% |     N/A |     
| YCrCb | 98.246% | 98.359% |   


Add color histogram, spatial
  
| Color| !Sqrt | Sqrt |     
| :---  | :-----: | :-----: |
| HSV   | 99.178% | 99.110% |   
| YUV   | 98.978% | N/A |     
| YCrCb | 98.891% | 98.947% |

Final
spatial, clr_hist (32)
HOG: HSV, ALL, Sqrt, 12 orient, 8 px_cell, 2 cpb, 

_The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier._


## Support Vector Machine

## Sliding windows
_A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen._

_Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)_


## Video implementation
_The sliding-window search plus classifier has been used to search for and identify vehicles in the videos provided. Video output has been generated with detected vehicle positions drawn (bounding boxes, circles, cubes, etc.) on each frame of video._

### Heat map
_A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur._

## Discussion
_Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail._