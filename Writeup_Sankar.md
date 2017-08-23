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
[image4]: ./output_images/CombinedThresholding.PNG "Combined Threshold"

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

Logical operations between the two thresholded images  are done to make the scheme more robust.


In addition to merging the two thresholded images, a region of interest masking also helps to filter out unwanted pixels. Image below shows an example of an image that com

1. An IMG directory that captures left, center and right camera images mounted on the car while driving around the track
2. A drivinglog.csv file that includes path to images captured above along with measurements like steer angle, brake position, pedal and vehicle speed

As mentioned before, this project primarily focuses on using an image to predict what the steer angle needs to be. Simply put, 

```sh
X_train=images
Y_train=steering_angle
```

**Data Augmentation**

While initial testing with center images alone was done to verify basic functionality, it became obvious that more training data is needed to make the network predict driving behavior better. A simple way to do that was to use images from all cameras. 

While images from left and right camera were being analyzed, a small correction factor indicating steer angle that will drive it to the center was added. For example, for an image from the left camera, the correction factor would steer the car slightly to the right. Code that performed this correction is shown below

```sh

for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='/home/carnd/P3_sankar/myData/data/IMG/'+ filename
        image=cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image == None:
            print("Invalid image:" , current_path)
        else:
            images.append(image)
            measurement = float(line[3])
            if i==1:
                measurements.append(measurement+0.2) ## for a left image, steer right a bit
            elif i==2:
                measurements.append(measurement-0.2) ## for a right image, steer left a bit
            else:
                measurements.append(measurement)   ## for a center image, do nothing     
```

An interesting thing to note was the driving direction on the track - clockwise vs anti-clockwise could bias the steer towards left or right. It is important to add this data to the training set to help the network perform better. One way to acquire this data is to actually drive around the track in anti-clockwise fashion. Cv2 has a very useful feature in the "flip" method that performs the same task in software. The figure below demonstrates use of this technique. The steer angle being a mirror image could simply be negated to create the correct label.

```sh
for image,measurement in zip(images,measurements):
    images_aug.append(image)    
    measurements_aug.append(measurement)
    images_aug.append(cv2.flip(image,1))
    measurements_aug.append(measurement*-1.0)
```


In total, the baseline data set size was 24108. With image augmentation via the flip technique, the size doubled to 48216. This was sufficient to train the network.

**Network Architecture**
---

Various network architectures were tested all the way from simple linear models to slightly complex architectures via convolutions. Nvidia published a paper that details their convolutional network architecture for mimicing human behavior.

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The architecture had the non-linearities needed for solving the problem and being tried and tested, I implemented this architecture for the problem. 

**Pre-Processing**

A "lambda" layer was added to normalize the image before training the model. The lambda layer in keras is essentially similar to adding python code that does the normalizing. An important advantage to using the lambda layer is that while testing the network on validation images, it goes through the same pre-processing without having to explicitly pre-process the feed images again. Normalizing code shown below:

```sh
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```

In addition to normalizing, image was also cropped to remove surrounding environment data that just added to noise. The base input image was of shape 160x320x3 (RGB). The keras cropping2D function was used to reduce the image down to 70x25x3. 

```sh
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

The Nvidia CNN architecture is shown below.

![alt text][image2]

The architecture was implemented in keras. 

```sh
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

Post normalizing, the network consists of 4 convolutional layers and 3 fully connected layers. The training parameters chosen were:

```sh
Optimizer=Adam with no specific learning rate
Loss Function='mse' as in mean squared error
validation split=0.2
Num of epochs=3

```
All the training was done on Amazon Web Sever using a GPU. Therefore, no generators were used. If done on a local machine without GPU, generators would have been necessary.

**Final Results**
---

The trained model parameters were saved and transferred to the local machine. By using the drive.py function and using "autonomous mode" on the simulator, the car was driven on the track using network predicted steer angles. The results were good and the vehicle did not leave the track even once. 

The video.py function was used to create the "MyRun".mp4 video. The frames per second parameter was set at 30.