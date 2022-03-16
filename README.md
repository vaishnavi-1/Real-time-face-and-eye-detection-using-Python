# Real-time-face-and-eye-detection-using-Python

Face Recognition is a technology in computer vision. In Face recognition / detection we locate and visualize the human faces in any digital image.
Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) and negative images (images without faces) to train the classifier. Then we need to extract features from it. For this, haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.
OpenCV already contains many pre-trained classifiers for face, eyes, smile etc.

## Steps to write code for real time face and eye detection in Python
### Step 1: Importing all required modules
1. numpy
2. opencv
### Step 2: Defining functionalities
In this step, we will define all the functions which will be linked afterward to different Buttons.
`faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')`
This is the line that loads the "classifier" (that must be in a directory named "Cascades/", under your project directory).
Then, we will set our camera and inside the loop, load our input video in grayscale mode (same we saw before).
Now we must call our classifier function, passing it some very important parameters, as scale factor, number of neighbors and minimum size of the detected face./

`faces = faceCascade.detectMultiScale(
 gray, 
 scaleFactor=1.2,
 minNeighbors=5, 
 minSize=(20, 20)
 )`

Where,
gray is the input grayscale image.
scaleFactor is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
minNeighbors is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives.
minSize is the minimum rectangle size to be considered a face.
The function will detect faces on the image. Next, we must "mark" the faces in the image, using, for example, a blue rectangle.
