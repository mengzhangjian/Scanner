from flask import Flask, Blueprint
from flask import request
import json
import cv2
import numpy as np
import infer


car_model_server = Blueprint('car_server', __name__)


@car_model_server.route('/shape', methods=['GET', 'POST'])
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

#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8082, debug=False)

