from flask import Flask, render_template, Response
from camera import VideoCamera
import cv2
import imutils
from imutils import face_utils
import dlib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/DetectionPage.html')
def detectionPage():
    return render_template('DetectionPage.html')

@app.route('/about.html')
def aboutPage():
    return render_template('about.html')

def gen(camera):
    while True:

        frame = camera.get_frame()
       
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)