# app.py
from flask import Flask
from flask import request
# from utils import Photo2Cartoon
# import cv2
# import numpy as np
# import base64

# flask app
app = Flask(__name__)


# # base64 转换为 cv2 数据
# def base64ToCV2(base64_data):
#     imgData = base64.b64decode(base64_data)
#     nparr = np.fromstring(imgData, np.uint8)
#     img_cv2 = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#     return img_cv2

# # cv2 转换为 base64 数据
# def cv2ToBase64(image):
#     cv_result = cv2.imencode('.png', image)[1]
#     base64_result = (str(base64.b64encode(cv_result))[2:-1])
#     return base64_result

# # 图片转换为卡通
# def toCartoon(base64_data):
#     cv2Data = base64ToCV2(base64_data)
#     c2p = Photo2Cartoon()
#     cartoon = c2p.inference(cv2Data)
#     if cartoon is not None:
#         base64_result = cv2ToBase64(cartoon)
#         return {
#             'data': {
#                 'Image': base64_result
#             },
#             'status': 0,
#             'message': ''
#         }
#     else:
#         image_error = 'can not detect face!!!'
#         return {
#             'data': {
#                 'Image': image_error
#             },
#             'status': 0,
#             'message': ''
#         }

@app.route("/")
def hello():
    return 'use “/cartoon”'

# @app.route("/cartoon", methods=['GET', 'POST'])
# def cartoon():
#     if request.method == 'POST':
#         data = request.get_json()
#         imageData = data['Image']
#         image_result = toCartoon(imageData)
#         print('image_result', image_result)
#         return image_result
#     else:
#         return 'do not GET'


if __name__ == "__main__":
    app.run()
