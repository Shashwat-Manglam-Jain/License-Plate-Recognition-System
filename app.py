from flask import Flask, render_template, request, Response, send_from_directory, jsonify
import cv2
import os
from ultralytics import YOLO
from werkzeug.utils import safe_join

app = Flask(__name__)

# Create necessary folders
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")  # Ensure you have your trained YOLOv8 model

# Global variable to control webcam streaming
webcam_active = True

def detect_objects(image_path, save_path):
    """ Runs YOLO detection on an image and saves the output. """
    image = cv2.imread(image_path)
    results = model.predict(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Plate {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload_image', methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    processed_path = os.path.join(PROCESSED_FOLDER, "processed_" + file.filename)

    file.save(image_path)  
    detect_objects(image_path, processed_path)

    return render_template("index.html", image_path=processed_path)

@app.route('/upload_video', methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    processed_filename = "processed_" + file.filename
    processed_video_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    return render_template("index.html", video_path=processed_filename)

@app.route('/processed_video/<filename>')
def processed_video(filename):
    file_path = safe_join(PROCESSED_FOLDER, filename)  # Prevents path traversal issues
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
def generate_frames():
    global webcam_active
    cap = cv2.VideoCapture(0)

    while True:
        if not webcam_active:
            cap.release()
            break
        
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    global webcam_active
    webcam_active = True
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/stop_webcam', methods=["POST"])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({"status": "Webcam turned off"})

if __name__ == '__main__':
    app.run(debug=True)
