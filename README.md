# Assisted driving using YOLO

A driving assistant (a dashboard app) to advise the driver on optimal speed , vehicles and pedestrian detection using YOLO.

![unnamed](https://user-images.githubusercontent.com/18646185/94140063-50091000-fe88-11ea-9096-f970ac3404f7.png)


### Tech stack
- python
- imageAI
- pygame
- numpy (numba -  with CUDA acceleration)
- OpenCV

### What does the app do?

The app continously tracks vehicles, pedestrians, signals using a pretrained common objects detection YOLO model (imageAI) and alerts the user if required. The app suggests optimal speed for the driver based on distance between the objects in front of the car (calculated using the diagonal width of the bounding boxes - needs to be calibrated). The calculations are accelerated using Nvidia GPU taking advantage of the CUDA cores witht help of numba (Python with CUDA Acceleration).

### To do
- Custom train model for Indian road signs
- Integrate with GPS

