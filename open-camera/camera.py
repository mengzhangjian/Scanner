import os
import cv2
import json
import requests  # NOTE: Only used for forceful reconnection
import time  # NOTE: Only used for throttling down printing when connection is lost
from pathlib import Path
from collections import OrderedDict

def open_cam_rtsp(uri, width, height, latency):
    """
    xavier gstreamer camera config
    """
    gst_str = ("rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def read_json(fname):
    """
    read config from json file
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

pwd_path = os.path.dirname(__file__)
camera_config = Path('camera_config.json')
if camera_config.is_file():
    config = read_json(camera_config)
else:
    print("Warning: camera configuration file is not found in {}.".format(camera_config))
    raise Exception("Could find config file: {}".format(camera_config))
    
class IPVideoCapture:
    def __init__(self, blocking=False):
        """
        :param cam_address: ip address of the camera feed
        :param cam_force_address: ip address to disconnect other clients (forcefully take over)
        :param blocking: if true read() and reconnect_camera() methods blocks until ip camera is reconnected
        """

        self.cam_address = config["camera"]["rtsp_address"]
        self.blocking = blocking
        self.capture = None
        self.config  = config
        self.RECONNECTION_PERIOD = 0.5  # NOTE: Can be changed. Used to throttle down printing

        self.reconnect_camera()

  
    def reconnect_camera(self):
        while True:
            try:
                if self.config is not None:
                    camera_config = self.config["camera"]
                    self.capture = open_cam_rtsp(self.cam_address, camera_config["width"],
                                                camera_config["height"], camera_config["latency"])
                else:
                    self.capture = open_cam_rtsp("rtsp://admin:abc12345678@192.168.1.64:554/", 1920, 1080, 50)

                if not self.capture.isOpened():
                    raise Exception("Could not connect to a camera: {0}".format(self.cam_address))

                print("Connected to a camera: {}".format(self.cam_address))

                break
            except Exception as e:
                print(e)

                if self.blocking is False:
                    break

                time.sleep(self.RECONNECTION_PERIOD)

    def read(self):
        """
        Reads frame and if frame is not received tries to reconnect the camera

        :return: ret - bool witch specifies if frame was read successfully
                 frame - opencv image from the camera
        """

        ret, frame = self.capture.read()

        if ret is False:
            self.reconnect_camera()

        return ret, frame
    def get(self, x):
        """
        get fps
        """
        return self.capture.get(x)

if __name__ == "__main__":
     CAM_ADDRESS = "http://192.168.8.102:4747/video"  # NOTE: Change
     CAM_FORCE_ADDRESS = "http://192.168.8.102:4747/override"  # NOTE: Change or omit
     cap = IPVideoCapture(blocking=True)
     # cap = IPVideoCapture(CAM_ADDRESS)  # Minimal init example

     while True:
         ret, frame = cap.read()
         print("*"*10)
         #if ret is True:
         #    cv2.imshow(CAM_ADDRESS, frame)

         #if cv2.waitKey(1) == ord("q"):
         #    break
