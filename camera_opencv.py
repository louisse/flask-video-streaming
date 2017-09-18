import cv2
import numpy as np
from base_camera import BaseCamera
from pylepton.Lepton3 import Lepton3


class Camera(BaseCamera):
    video_source = 0
    head_cscd = cv2.CascadeClassifier('data/cascadeH5.xml')
    body_cscd = cv2.CascadeClassifier('data/cascadG.xml')
#    face_cscd = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    #fbody_cscd = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def read(device = "/dev/spidev0.0"):
        with Lepton3(device) as l:
            a,_ = l.capture()
        cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(a, 8, a)
        return np.uint8(a)

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            success, img = camera.read()

            if not success:
                continue
            gray = Camera.read()
            gray = gray[10:110,10:150]
            gray = cv2.resize(gray, (160,120))
            lower = np.array([90])
            upper = np.array([255])

            mask = cv2.inRange(gray, lower, upper)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            res = cv2.bitwise_and(img,img, mask=mask)
            res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            bodies = Camera.body_cscd.detectMultiScale(res_gray, 1.05, 5)
            for (x,y,w,h) in bodies:
                cv2.rectangle(res, (x,y), (x+w, y+h), (0, 0, 255), 2)

            heads = Camera.head_cscd.detectMultiScale(res_gray, 1.05, 5)
            for (x,y,w,h) in heads:
                cv2.rectangle(res, (x,y), (x+w, y+h), (0, 255, 0), 2)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', res)[1].tobytes()
