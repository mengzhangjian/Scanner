# coding=utf-8
from flask import Flask
from flask import request
import json
import ssd
import cv2
import numpy as np
app = Flask(__name__)


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """
    web request
    """

    result = {"results": {}}
    try:
        val_image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
        stats, char, score = ssd.main(val_image)
        result["results"]["code"] = char
        result["results"]["score"] = score
        result["results"]["state"] = stats
        print(score)
        if stats == 1:
            print(stats, char)
        return json.dumps(result), 200
    except Exception as e:
        print(e)
        return json.dumps(result), 403


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8084, debug=False)
