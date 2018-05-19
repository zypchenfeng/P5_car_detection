## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./submit_imgs/car_not_car.png
[image2]: ./submit_imgs/HOG_example.png
[image3]: ./submit_imgs/bin_example.png
[image4]: ./submit_imgs/sliding_windows.png
[image5]: ./submit_imgs/search_window.png
[image6]: ./submit_imgs/heat_map.png
[image7]: ./submit_imgs/label_box.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 3rd to 6th code cell (titled `Example of HOG`) of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.feature.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I have also explored the spacial bin and here is a comparison between car and not-car sample.

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

In order to tune the combination of parameter, I have made a parameter tuning function `parameter_tuning`.
First, I have selected 2000 random samples from cars and not-cars (out of ~9000).
I tried various combinations of parameters and did the matrix scan one by one. By looking at the test accuracy, make the decision to choose the best one. 

Here are the list of parameters I have scanned:



| Name        | Search Range   |  Best Choice | Test Accuracy |
|:-------------:|:-------------:|:-----------:|:-----------:|
| Color Space      | RGB, HSV, YCrCb, LUV, YUV, HLS     |     YCrCb      |0.9925|
| pix_per_cell      | 8, 16, 32     |    8       |0.975|
| cell_per_block     | 1, 2, 3, 4      |       2    |0.98|
| hog_channel     | 0, 1, 2, ALL       |     ALL       |0.9825|
| spatial_size | (8, 8), (16, 16), (32, 32),  (64, 64) | (32, 32)|0.97|
| hist_bins | 8, 16, 32 | 32|0.9625|


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `LinearSVC` from `sklearn.svm`. In the session _Extract Features and Store to Prevent Repeat Calculation_
I prepared the data needed for training. Then saved the data into a `pickle` file for reuse of data.

Session _Load Saved Feature Data and Train The Model_ is where I trained my model. After training the model, the `X_scale`
as well as the model and parameters are saved in `pickle` file for future usage.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the bottom half of the image at 96x96 window and came up with this:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from one frame:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took:
1. Prepare functions to train the svc model
2. Decide what features to use, I used combination of color, bin spatial and gradient(HOG)
3. Choose and train a classifier: I used linear SVC, but will check if other method can improve the accuracy
4. Sliding windows to search vehicle in test images: use 96x96 to sliding over the image
5. Use heatmap to filter out the false positive
6. Take average of the input video frames and apply large heat filter to get stable output.
  
The techniques I used:
1. I used matrix for the parameters to scan for the best combination
2. I followed the suggestion from the lesson and did HOG for part of the image at one time, then extract and flatten the features.
 
What worked and why:
1. The average of multiple frames from image worked. It can greatly filter out the noise
2. The heat map worked. It can also help filter the false positive.


Where the pipeline might fail:
1. If there are two cars very close to each other, the pipeline will fail.
2. If the car is only partly presented, the pipeline cannot recognize the car. 
3. The method is super slow, makes it impossible to implement for real time car detection.

How I might improve it if I were going to pursue this project further:
I have started to implementing the YOLO method trying to improve the accuracy and do real time detection. 

