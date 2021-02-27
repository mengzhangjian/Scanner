import requests
import base64
import json
import cv2
import time
a = time.time()
img = cv2.imread('1.jpg')
imencoded = cv2.imencode('.jpg', img)[1].tostring()
#data = {"image": base64.b64encode(imencoded).decode("utf-8"), "threshold": 0.5}
r = requests.post('http://0.0.0.0:8082/shape', data=imencoded, timeout=50)
b = time.time()
print(b - a)
print(r.status_code)
rs = json.loads(r.text)
print(rs)
