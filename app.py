# from model_predict import damage_detect
import cv2
import flask
from ultralytics import YOLO
from random import randint

app = flask.Flask(__name__)
camera = cv2.VideoCapture(1)  # Access webcam

model = YOLO("best.onnx")  # Load YOLOv8 model (replace with your model path)

def damage_detect(frame):
    
    results = model(frame)  # Perform object detection

    # Render bounding boxes and labels using OpenCV
    print(len(results))
    for box in results[0].boxes:
        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        cls = box
        label = f'{results[0].names[int(box.cls)]}: {float(box.conf):.2f}'
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw green rectangles
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Add labels

    return cv2.imencode('.jpg', frame)  # Encode frame as JPEG

@app.route('/')
def index():
    return flask.render_template('index.html')

def generate_frames():
    while True:
        success, frame = camera.read()  # Read a frame from the camera
        if not success:
            break  # Handle camera failure gracefully

        ret, buffer = damage_detect(frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return flask.Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
