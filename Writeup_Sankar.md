## Behavioral Cloning Project
---

The primary goals of this project are as follows:
* Use the udacity provided simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that mimics good driving behavior
* Based on image obtained from center, left and right cameras, steer angle is predicted
* Test that the model successfully drives car around the track without leaving the road in fully autonomous fashion

[//]: # (Image References)

[image1]: ./examples/FlipDemonstrate.PNG "ImageAugmentation"
[image2]: ./examples/NvidiaCNNarch.PNG "NvidiaCNNArchitecture"
[image3]: ./examples/placeholder_small.png "Recovery Image"

**Description of files**

* model.ipynb contains the script to create and train the model
* model.h5 contains the final trained convolution neural network model - weights, parameters etc...
* drive.py is a utility function for driving the car in autonomous mode on udacity simulator
* MyRun.mp4 is a video demonstrating the car driving in fully autonomous mode by predicting steer angles based on input images via the model contained in "model.h5"

**Building the driver model**
---

As a first step, the car was driven in training mode using the Udacity simulator to record data for optimal driving behavior. Optimal in this case is to keep the car in the center of the lane as much as possible. Not being a gamer myself plus not having a joystick, it was actually very hard to drive around the test track being in the center of the lane :-) 

A sample training data set was provided by Udacity. Since bad training data would result in a garbage in - garbage out case and to get to the crux of the problem, I  relied on the Udacity provided data-set for the project. The dataset primarily consists of two files:

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

![alt text][image1]


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