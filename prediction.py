'''
Script for Predictions
'''

import tensorflow as tf
import base64
import cv2
import numpy as np


model = None


def save_image(image_data):
    encoded_data = image_data.split(',')[1]
    with open("img.jpeg", "wb") as fh:
        fh.write(base64.b64decode(encoded_data))


def crop_face():
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    img = cv2.imread('img.jpeg')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        img = img[y-30:y+h+30, x-30:x+w+30]
    resized_image = cv2.resize(img, (160, 160))
    cv2.imwrite('img.jpeg', resized_image)


def get_faceprint():
    '''Function to read JPEG Images from GCP GCS bucket.

        Parameters:
            path        - Image files path.
            shape       - Image Shape.
        Return Value:
            Processed image and label tensors.
    '''
    model = tf.keras.models.load_model('faceprint.h5', compile=False)
    img = cv2.imread('img.jpeg')
    image = tf.expand_dims(img, 0, name=None)
    print(image.shape)
    faceprint = model.predict(image)[0].tolist()
    output = (faceprint / np.sqrt(np.maximum(np.sum(np.square(faceprint),
                                                    axis=-1, keepdims=True), 1e-10))).tolist()
    print(output)
    return str(output)
