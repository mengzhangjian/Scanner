from flask import Flask, Blueprint
from flask import request
import json
import cv2
import numpy as np
#import infer_color
from typeDistinguish import SimplePredict

color_server = Blueprint('color_server', __name__)

global_index = 0

@color_server.route('/color', methods=['GET', 'POST'])
def detect():
    """
    web request
    """

    res = {'results': []}

    try:
        val_image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        conf, name = infer.get_result(val_image)
        res['results'].append({
            'name': str(name),
            'score': float(conf),
        })
        return json.dumps(res)
    except Exception as err:
        print(err)
        return str(err), 403

def color_detect(cv_img):
    res = {'results': []}
    # conf , name = infer_color.get_result(cv_img)

    # img = cv2.imread(str(global_index) + ".jpg")
    name , conf = SimplePredict(cv_img)
    res['results'].append({
            'name': str(name),
            'score': float(conf)})
    return res


# if __name__ == "__main__":
#     img = cv2.imread('../receive-car/0.jpg')
#     # img = cv2.cvtColor(img , c)
#     res = color_detect(img)
#     print(res)

