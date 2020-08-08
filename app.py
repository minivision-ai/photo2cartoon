# app.py
from flask import Flask
from flask import request
from utils import Photo2Cartoon
import cv2
import numpy as np
import base64
app = Flask(__name__)


def toCartoon(base64_data):
    imgData = base64.b64decode(base64_data)
    nparr = np.fromstring(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)
    c2p = Photo2Cartoon()
    cartoon = c2p.inference(img_np)
    # print('cartoon', cartoon)
    if cartoon is not None:
        cv_result = cv2.imencode('.jpg', cartoon)[1]
        # print('str(base64.b64encode(cv_result)', str(base64.b64encode(cv_result)))
        base64_result = (str(base64.b64encode(cv_result))[2:-1])
        return {
            'data': {
                'Image': base64_result
            },
            'status': 0,
            'message': ''
        }
    else:
        image_error = 'can not detect face!!!'
        return {
            'data': {
                'Image': image_error
            },
            'status': 0,
            'message': ''
        }

@app.route("/cartoon", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        data = request.get_json()
        imageData = data['Image']
        image_result = toCartoon(imageData)
        print('image_result', image_result)
        return image_result
    else:
        return 'do not GET'


if __name__ == "__main__":
    app.run()
