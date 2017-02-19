
# Vehicle Detection
This project is a submission for the Udacity [Self-Driving Car Engineer Nanodegree](http://udacity.com/drive).
Vehicles in a video are detected and tracked using computer vision (OpenCV) and Machine Learning techniques.



[//]: # (Image References)
[image1]: ./docs/org_vehicle/412.jpeg "Vehicle"
[image2]: ./docs/org_vehicle/529.jpeg "Vehicle"
[image3]: ./docs/org_vehicle/703.jpeg "Vehicle"
[image4]: ./docs/org_non_vehicle/extra108.jpeg "Non-vehicle"
[image5]: ./docs/org_non_vehicle/extra198.jpeg "Non-vehicle"
[image6]: ./docs/org_non_vehicle/extra283_64.jpeg "Non-vehicle"
[image7]: ./docs/
[image8]: ./docs/
[image9]: ./docs/
[image10]: ./docs/
[image11]: ./docs/
[image12]: ./docs/
[image13]: ./docs/
[image14]: ./docs/
[image15]: ./docs/

[video1]: ./project_video.mp4

## Feature extraction

### Color Histograms

### Color space

### Histogram of Oriented Gradients (HOG)
_Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why._

_The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier._

#### Read data
Labeled images in resolution 64 x 64 pixels with the classes `vehicle` and `non-vehicle` are loaded.  Examples for the both classes below.

Vehicle | Vehicle | Vehicle
:---: | :---: | :---:
![alt_text][image1] | ![alt_text][image2] |![alt_text][image3]

Non-vehicle | Non-vehicle | Non-vehicle
:---: | :---: | :---:
![alt_text][image4] | ![alt_text][image5] |![alt_text][image6]

Some of the images labeled `non-vehicle` do in fact contain vehicles. In order to qualify as a `vehicle` image the entire vehicle must be visible within the bounds of the image and, in addition, make a close fit the image borders. 

file: `data_reader.py`

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