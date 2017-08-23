## Advanced Lane Detection Project
---

The primary goals of this project are as follows:
* Create a lane detection pipeline that can handle curved lanes, shadows on road as well as changes in pavement color
* Apply the lane detection pipeline on sample images succesfully
* Apply the developed pipeline on a udacity provided test video successfully

[//]: # (Image References)

[image1]: ./output_images/CameraCal.PNG "CameraCal"
[image2]: ./output_images/Distortion.PNG "Distortion"
[image3]: ./output_images/GradientThresholding.PNG "Gradient Threshold"
[image4]: ./output_images/ColorThresholding.PNG "Color Threshold"
[image5]: ./output_images/CombinedThresholding.PNG "Combined Threshold"
[image6]: ./output_images/MaskedCombinedThreshold.PNG "MaskedCombinedThreshold"
[image7]: ./output_images/Perspective.PNG "PerspectiveTransform"
[image8]: ./output_images/histogram.PNG "histogram"
[image9]: ./output_images/SlidingWindows.PNG "SlidingWindows"
[image10]: ./output_images/LaneDetectMargin.PNG "LaneDetectMargin"
[image11]: ./output_images/RadiusOfCurvature.PNG "RadiusOfCurvature"
[image12]: ./output_images/WidthCurrMA.PNG "WidthCurrMA"


**Description of files**

* AdvancedLaneDetection_Sankar.ipynb is the jupyter notebook that contains the code
* project_video_output.mp4 is the video generated via the developed image processing pipeline

**Algorithm**
---

The pin-hole camera that is used to generate the images needs to be calibrated to work in the real world. Therefore the first step was to use a set of calibration images to obtain the camera calibration matrices. The opencv library was used to perform the calibration. Sample image shown below identifying all the corners on the chess board.

![alt text][image1]

Once the camera calibration matrices are generated, the images are undistorted. Distortion addresses the curved edges that are captured by the pin hole camera and flattens the content of the image. Sample of an image being undistorted is shown below.

![alt text][image2]

**Thresholding**

The next step is to start detecting lanes. The two primary knobs used for lane detection is edge detection and color detection. The first step was to experiment with the edge detection scheme. The sobel operators are used which calculate the gradient in x and y directions. In addition to just absolute thresholds in a single direction (x or y), direction and magnitude of the detected line segments are also useful. Image below shows the effect of gradient thresholding on a raw image. 

![alt text][image3]

In addition to the gradient thresholding, it is possible to use the RGB or HSV (Hue, Lightness and Saturation) channels of an image to perform some color thresholding. Below is an example of lane detection purely by color thresholding. 

![alt text][image4]

Logical operations between the two thresholded images  are done to make the scheme more robust. Image below shows the gradient and color thresholded images shown above merged together by an AND operation. 

![alt text][image5]

In addition to merging the two thresholded images, a region of interest masking also helps to filter out unwanted pixels. Image below shows the noise reduction between the combined thresholded image when masked by a region of interest.

![alt text][image6]

Once the lane lines are identified in an image, a perspective transform is applied. The idea of a perspective transform is to get a bird's eye view of the road. This is very useful in determining curvature between the lanes. The opencv perspective transform function is used to obtain a transformation matrix that maps a set of source and destination points. Image below shows a perspective transform being applied on our image. 

![alt text][image7]

**Lane Detection**

Using the above techniques, the set of pixels that contribute to a line are identified. The next step is to take this further and apply a histogram filter to detect sharp changes in pixel density as shown below. 

![alt text][image8]

The peaks of the histogram provides a good starting point to look for the line pixels. A moving window scheme is implemented to scan through the image looking for non-zero pixels. The window re-positions based on pixel density. Once all the pixels are obtained, the numpy polyfit function is used to fit a second order polynomial. Image below shows implementation of the window scheme. 

![alt text][image9]

The sliding windows although is an expensive operation. Once the lanes are detected using the sliding windows, the next image will not change radically. It is possible to search around a specified margin from the previous lane to detect the lanes from new image. 

![alt text][image10]

For visualization purposes, the obtained lanes are transformed back to the original image. The inverse of the perspective transform is calculated to do this. Also the cv2.fillpoly function is used to clearly mark the lane width. The radius of curvature of the road and distance of the vehicle from the center of the road is calculated and displayed on the image. 

![alt text][image11]

The above pipeline was configured to run on a video that was provided by Udacity. The video has straight lanes, curved lanes, changes in pavement color as well as shadows on road. The established pipeline effectively identified lane markings and ensures no divergence. The video is also available here: https://youtu.be/d5KURcDlJ-s

**Challenges**

While the above pipeline was tested successfully on a subset of images, the key challenge while operating on a video is the fact that there are frames where lane markings are not clearly visible. This could result in a case where no lanes are detected. Also, it is possible that no lane pixels were detected in the specified margin from the previous image. In order to address these issues, few mechanisms were established:

- Calculate lane width from the current frame. Establish a moving average of the lane width using a simple mechanism as mentioned below:
	
	```sh
        if lane_width_movingavg==0:
            lane_width_movingavg=lane_width_curr
                        
        tol=0.25 #tolerance
        if (lane_width_curr>(1+tol)*lane_width_movingavg) or (lane_width_curr<(1-tol)*lane_width_movingavg):
            badlines=True
        else:
            badlines=False
            lane_width_movingavg=0.8*lane_width_curr+0.2*lane_width_movingavg
	```
- This served as an excellent tool to identify problems in lane detection. If lane width changes dramatically, that clearly identifies a problem in the thresholding scheme. Sample image shown below. Note the sudden drop in lane width around frame 162. 
	
![alt text][image12]
	
	
- If no line pixels are detected, just return the original image.
- One of the challenging areas in the video is when the pavement color changes and shadows falling on the road. A video demonstrating the problem is shown [here](https://www.youtube.com/watch?v=daJ_YtJVrBg)

[![Project Video](https://img.youtube.com/vi/LJc_GhtzSCY/0.jpg)](https://www.youtube.com/watch?v=daJ_YtJVrBg)

- When no lane pixels are detected, two key things were done:
	1) Perform sliding window search on the histogram filter again
	2) Pass the last known good lane to the algorithm for stability
	

**Area of Improvement**

- Thresholding does not seem to work well on the challenge videos. Could refine it more
- Sharper turns need to be handled better as in the harder challenge video



