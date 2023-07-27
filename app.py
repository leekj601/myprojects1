import cv2
import dlib
from flask import Flask, render_template, Response, redirect
import time
import webbrowser

app = Flask(__name__)

# Dlib의 눈동자 추적기와 얼굴 검출기 초기화
predictor_path = 'shape_predictor_68_face_landmarks.dat'
eye_predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.get_frontal_face_detector()
blink_start_time = None
blink_end_time = None

# 웹캠 캡처 객체 생성
video_capture = cv2.VideoCapture(0)

# 이전 눈동자 좌표 초기화
prev_left_eye_center = (0, 0)
prev_right_eye_center = (0, 0)

# 눈동자 추적 함수
def track_eye(frame):
    global blink_start_time, blink_end_time 
    global prev_left_eye_center, prev_right_eye_center

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    for rect in rects:
        shape = eye_predictor(gray, rect)

        # 눈동자 좌표 계산
        left_eye_center = ((shape.part(36).x + shape.part(39).x) // 2, (shape.part(36).y + shape.part(39).y) // 2)
        right_eye_center = ((shape.part(42).x + shape.part(45).x) // 2, (shape.part(42).y + shape.part(45).y) // 2)

        # 눈동자를 원으로 표시
        cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)

        # 눈동자 움직임 시각화
        if prev_left_eye_center != (0, 0):
            cv2.line(frame, prev_left_eye_center, left_eye_center, (0, 0, 255), 2)
        if prev_right_eye_center != (0, 0):
            cv2.line(frame, prev_right_eye_center, right_eye_center, (0, 0, 255), 2)

        # 눈동자 좌표 출력
        print("Left Eye:", left_eye_center)
        print("Right Eye:", right_eye_center)

        # 눈 감김 감지
        left_eye_aspect_ratio = calculate_eye_aspect_ratio([shape.part(i) for i in range(36, 42)])
        right_eye_aspect_ratio = calculate_eye_aspect_ratio([shape.part(i) for i in range(42, 48)])

        # 눈 감김 비율(threshold) 조정
        eye_closed_threshold = 0.25  # 적절한 값을 설정해주세요.

        if left_eye_aspect_ratio < eye_closed_threshold and right_eye_aspect_ratio < eye_closed_threshold:
            if blink_start_time is None:  # 눈 감은 순간을 기록
                blink_start_time = time.time()
            cv2.putText(frame, "Blinking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if blink_start_time is not None:
                blink_end_time = time.time()
                blink_duration = blink_end_time - blink_start_time
                print("Blink Duration:", blink_duration)
                if blink_duration>=2:
                     webbrowser.open('https://www.naver.com', new=2)
                blink_start_time = None  # 초기화하여 다음 눈 감김을 기록할 수 있도록 함

        # 이전 눈동자 좌표 업데이트
        prev_left_eye_center = left_eye_center
        prev_right_eye_center = right_eye_center

    return frame

def calculate_eye_aspect_ratio(eye_points):
    # 눈꺼플 높이 계산 (수직 눈꺼플 간의 거리)
    vertical_dist = abs(eye_points[1].y - eye_points[5].y)
    
    # 눈꺼플 너비 계산 (수평 눈꺼플 간의 거리)
    horizontal_dist1 = abs(eye_points[0].x - eye_points[3].x)
    horizontal_dist2 = abs(eye_points[1].x - eye_points[4].x)
    horizontal_dist = (horizontal_dist1 + horizontal_dist2) / 2.0

    # 눈동자의 가로 세로 비율 계산
    aspect_ratio = vertical_dist / horizontal_dist

    return aspect_ratio

# Flask 라우트: 눈동자 추적 페이지
@app.route('/')
def index():
    return render_template('index.html')

# Flask 라우트: 웹캠 스트리밍 페이지
def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame_with_eyes = track_eye(frame)

            ret, buffer = cv2.imencode('.jpg', frame_with_eyes)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
