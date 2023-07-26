from flask import Flask, render_template, Response, redirect, url_for
import cv2
import time
import webbrowser

app = Flask(__name__)

# Haar Cascade 분류기 로드
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Time threshold for redirection (in seconds)
TIME_THRESHOLD = 2  # 눈이 감지되지 않은 상태를 2초로 설정

# Variable to keep track of the time of the last eye detection
last_eye_detected_time = 0

def is_eye_closed(eyes):
    if len(eyes) == 0:
        return True
    else:
        return False

def eye_tracking():
    global last_eye_detected_time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 그레이스케일로 변환 (Haar Cascade는 흑백 이미지에서 동작)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 눈 검출
        eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if is_eye_closed(eyes):
            # If no eyes are detected, check if the threshold time is exceeded
            current_time = time.time()
            if current_time - last_eye_detected_time > TIME_THRESHOLD:
                # Redirect the user to a new page (e.g., 'no_eyes_detected')
                return redirect('no_eyes_detected')

        # If eyes are detected, update the last_eye_detected_time
        else:
            last_eye_detected_time = time.time()

        for (ex, ey, ew, eh) in eyes:
            # 눈 주변에 사각형 그리기
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(eye_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/no_eyes_detected')
def no_eyes_detected():
    # 사용자를 네이버 웹사이트로 리다이렉트합니다.
    webbrowser.open('https://www.naver.com', new=2)
    return "Redirecting to Naver..." 

if __name__ == '__main__':
    app.run(debug=True)
