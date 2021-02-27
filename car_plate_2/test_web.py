import requests
import base64
import json
import cv2
import time
import os
path = "./save"
for index in os.listdir(path):

    name = os.path.join(path, index) 
    print(name)
    img = cv2.imread(name)
    imencoded = cv2.imencode('.jpg', img)[1].tostring()
    r = requests.post('http://0.0.0.0:8083/plate', data=imencoded, timeout=50)
    print(r.status_code)
    rs = json.loads(r.text)
    print(rs)
