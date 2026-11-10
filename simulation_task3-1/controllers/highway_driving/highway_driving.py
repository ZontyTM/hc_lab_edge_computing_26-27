from vehicle import Driver

if __name__ == "__main__":
    driver = Driver()
    driver.setSteeringAngle(0.0)  # go straight

    camera = driver.getDevice('camera')
    camera.enable(50)
    camera.recognitionEnable(50)

    while driver.step() != -1:

        img = camera.getImage()
        pass
