# Determination of HOG parameters
In order to determine the optimal HOG parameters a number of experiments was run. The larger data set (~9000 images per class) was used for training using only HOG features (neither spatial bins nor color histogram). Features were extracted and a linear Support Vector Machine was trained and evaluated at 80%/20% training/test set.

```python
def train(self, X_train, X_test, y_train, y_test):
    from sklearn.svm import LinearSVC
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    accuracy_test = svc.score(X_test, y_test)
```
[file: classifier.py](../lib/classifier.py)

## Test 1 - Color space and channel
The starting point for HOG parameters was `orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, normalise=None` with `RGB` color space using all three color channels. HOG features were extracted using different color spaces, using different channels. The test accuracy was evaluated 10 times and the average value saved. For each iteration the data was normalized, randomized, and split anew. The result is presented in the table below.

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

Channels 1, 2 in LUV and channel 2 in YUV color spaces can contain negative values which is why these test were run setting `transform_sqrt=False` indicated by (f) as opposed to (t) in the table.

The benefit of using all three channels was clear and also that some color spaces performed better than others.

## Test 2 - Orientations and Cells per block
In this experiment all channels of HSV color space is used and `transform_sqrt=True`. The number of orientations as well as `cells_per_block` are allowed to vary.

| Orient/CPB   | 2       | 3       | 4       |
| :---  | :-----: | :-----: | :-----: |  
| 8     | 97.773% | 97.852% | 97.534% |
| 10    | 98.066% | 98.086% | 97.790% |
| 12    | 98.359% | 98.316% | 98.074% |
| 14    | 98.328% | 98.308% | 98.162% |

It is concluded that 12 orientations with 2 cells per block is the ideal combination.

## Test 3 - Color space
Using the results from previous test we set parameters to 12 orientations, 2 cells per block and all channels while varying the color space and `transform_sqrt`.

| Color| !Sqrt | Sqrt |     
| :---  | :-----: | :-----: |
| RGB   | 96.568% | 96.585% |
| HSV   | 98.421% | 98.249% |
| LUV   | 97.075% |     N/A |     
| HLS   | 98.091% | 98.221% |     
| YUV   | 98.457% |     N/A |     
| YCrCb | 98.246% | 98.359% |

The strongest performance is seen in HSV, YUV, and YCrCb.

## Test 4 - Include color histogram and spatial features
Color histogram and spatial features were added to the best performing color spaces and accuracy measured.

| Color| !Sqrt | Sqrt |     
| :---  | :-----: | :-----: |
| HSV   | 99.178% | 99.110% |   
| YUV   | 98.978% | N/A |     
| YCrCb | 98.891% | 98.947% |

All three color spaces show strong performance at roughly 99% (mean 99.02%, std 0.11%). The potential advantage/disadvantage of `transform_sqrt` is very small.

## Conclusion
The final configuration chosen after the range of tests is HSV with all channels, 12 orientations, 2 cells_per_block, 8 px per cell, transform_sqrt=True, spatial features enabled, color histograms enabled.
