'''
Flask API to make predictions
'''
import logging
import os
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import base64
import prediction
import time
import json
import os
import cv2

app = Flask(__name__)
cors = CORS(app)


@app.route('/predict', methods=['POST'])
def get_predictions():
    '''Function to call when a POST request is made.

        Parameters:
            None
        Return Value:
            Predictions List.
    '''

    if request.method == 'POST':
        image_data = json.loads(request.data)['data']
        prediction.save_image(image_data)
        prediction.crop_face()
        faceprint = prediction.get_faceprint()
    return faceprint


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port, debug=True)
