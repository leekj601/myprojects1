import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Dlib의 얼굴 감지기와 눈 감지기 초기화
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# OpenCV 웹캠 비디오 스트림 초기화
video_capture = cv2.VideoCapture(0)

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 0:
        return False
    
    face = faces[0]
    landmarks = landmark_predictor(gray, face)
    
    left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                         (landmarks.part(37).x, landmarks.part(37).y),
                         (landmarks.part(38).x, landmarks.part(38).y),
                         (landmarks.part(39).x, landmarks.part(39).y),
                         (landmarks.part(40).x, landmarks.part(40).y),
                         (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
    
    right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                         (landmarks.part(43).x, landmarks.part(43).y),
                         (landmarks.part(44).x, landmarks.part(44).y),
                         (landmarks.part(45).x, landmarks.part(45).y),
                         (landmarks.part(46).x, landmarks.part(46).y),
                         (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
    
    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)
    
    return True

def gen_frames():
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        has_eyes = detect_eyes(frame)
        if has_eyes:
            cv2.putText(frame, 'Eyes Open', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Eyes Closed', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
