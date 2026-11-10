"""Sample Webots controller for highway driving benchmark."""

from vehicle import Driver
import cv2
import threading
import numpy as np
from controller import Display


if __name__ == "__main__":
    # get the camera
    driver = Driver()
    driver.setSteeringAngle(0.0)  # go straight

    camera = driver.getDevice('camera')
    camera.enable(50)
    camera.recognitionEnable(50)
    display = driver.getDevice("display")
    

    while driver.step() != -1:

        driver.setSteeringAngle(0.0)
        driver.setCruisingSpeed(80)




        maxSpeed = 80
