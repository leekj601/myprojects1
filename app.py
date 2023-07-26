import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Dlib의 얼굴 감지기와 눈 감지기 초기화
face_detector = dlib.get_frontal_face_detector()
eye_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# OpenCV 웹캠 비디오 스트림 초기화
video_capture = cv2.VideoCapture(0)

def detect_pupil(eye_region_gray, eye_region_color):
    # 눈동자 추적을 위한 허프 원 검출
    circles = cv2.HoughCircles(eye_region_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                               param1=50, param2=30, minRadius=5, maxRadius=30)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # 가장 크기가 큰 원(눈동자) 선택
        for (x, y, r) in circles:
            cv2.circle(eye_region_color, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(eye_region_color, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            break

    return eye_region_color

def gen_frames():
    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 감지
        faces = face_detector(gray)

        for face in faces:
            landmarks = landmark_predictor(gray, face)
            left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

            right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                         (landmarks.part(43).x, landmarks.part(43).y),
                                         (landmarks.part(44).x, landmarks.part(44).y),
                                         (landmarks.part(45).x, landmarks.part(45).y),
                                         (landmarks.part(46).x, landmarks.part(46).y),
                                         (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

            # 눈동자 추적
            left_eye_gray = gray[left_eye_region[0][1]:left_eye_region[5][1],
                                 left_eye_region[0][0]:left_eye_region[3][0]]
            left_eye_color = frame[left_eye_region[0][1]:left_eye_region[5][1],
                                   left_eye_region[0][0]:left_eye_region[3][0]]

            right_eye_gray = gray[right_eye_region[0][1]:right_eye_region[5][1],
                                  right_eye_region[0][0]:right_eye_region[3][0]]
            right_eye_color = frame[right_eye_region[0][1]:right_eye_region[5][1],
                                    right_eye_region[0][0]:right_eye_region[3][0]]

            detect_pupil(left_eye_gray, left_eye_color)
            detect_pupil(right_eye_gray, right_eye_color)

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
