from flask import Flask, request, render_template, redirect, url_for, Response
import os
from werkzeug.utils import secure_filename
from aspose.slides import Presentation
from aspose.pydrawing import Size
from aspose.pydrawing.imaging import ImageFormat
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PRESENTATION_FOLDER = 'Presentation'
ALLOWED_EXTENSIONS = {'ppt', 'pptx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PRESENTATION_FOLDER'] = PRESENTATION_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRESENTATION_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_ppt_to_images(ppt_path, output_folder):
    pres = Presentation(ppt_path)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through slides
    for index, slide in enumerate(pres.slides):
        # Set custom size for thumbnails
        size = Size(960, 720)

        # Save each slide as a PNG in the output directory
        slide_path = os.path.join(output_folder, f"slide_{index + 1}.png")
        slide.get_thumbnail(size).save(slide_path, ImageFormat.png)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Convert PPT to images
            convert_ppt_to_images(filepath, PRESENTATION_FOLDER)
            if not os.listdir(PRESENTATION_FOLDER):
                return "Error: Slide conversion failed. Please check the uploaded file."

            return redirect(url_for('presentation'))
    return render_template('upload.html')

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

def generate_frames():
    width, height = 1280, 720  # Resolution
    gestureThreshold = 700

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Debug: Camera not found. Streaming slides only.")
        cap = None

    if cap:
        cap.set(3, width)
        cap.set(4, height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
    mp_draw = mp.solutions.drawing_utils

    imgNumber = 0
    pathImages = sorted(os.listdir(PRESENTATION_FOLDER), key=len)
    if not pathImages:
        print("Error: No images found in the presentation folder.")
        return

    annotations = []
    annotation_temp = []
    annotation_color = (0, 0, 255)  # Default red
    last_command_time = 0
    command_interval = 1  # Minimum time between commands

    try:
        while True:
            img = None
            if cap:
                success, img = cap.read()
                if not success:
                    img = None
                else:
                    img = cv2.flip(img, 1)

            # Load slide
            pathFullImage = os.path.join(PRESENTATION_FOLDER, pathImages[imgNumber])
            imgSlide = cv2.imread(pathFullImage)

            if imgSlide is None:
                print(f"Error: Slide {imgNumber} could not be loaded.")
                continue

            imgSlide = cv2.resize(imgSlide, (width, height))

            if img is not None:
                # Process the image with MediaPipe Hands
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Extract landmark positions
                        lmList = []
                        for id, lm in enumerate(hand_landmarks.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append((cx, cy))

                        if len(lmList) >= 8:  # Index finger tip
                            cx, cy = lmList[8]
                            current_time = time.time()

                            # Draw cursor at the tip of the index finger
                            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                            # Left swipe gesture
                            if cy < gestureThreshold and len(lmList) >= 8:
                                if lmList[4][0] > lmList[8][0] and current_time - last_command_time > command_interval:
                                    if imgNumber > 0:
                                        print(f"Debug: Swiping left, going to slide {imgNumber - 1}")
                                        imgNumber -= 1
                                        annotations.clear()
                                        annotation_temp.clear()
                                        last_command_time = current_time

                            # Right swipe gesture
                            if cy < gestureThreshold and len(lmList) >= 8:
                                if lmList[4][0] < lmList[8][0] and current_time - last_command_time > command_interval:
                                    if imgNumber < len(pathImages) - 1:
                                        print(f"Debug: Swiping right, going to slide {imgNumber + 1}")
                                        imgNumber += 1
                                        annotations.clear()
                                        annotation_temp.clear()
                                        last_command_time = current_time

                            # Index Finger Up: Start or continue annotation
                            elif len(lmList) >= 8:
                                annotation_temp.append((cx, cy))

                            # Store completed annotation
                            if annotation_temp:
                                annotations.append(annotation_temp.copy())
                                annotation_temp.clear()

                # Draw annotations on the slide
                for annotation in annotations:
                    for i in range(1, len(annotation)):
                        cv2.line(imgSlide, annotation[i - 1], annotation[i], annotation_color, 5)

            # Combine camera and slide
            if img is not None:
                combined = np.hstack((img, imgSlide))
            else:
                combined = imgSlide

            # Encode and stream
            _, buffer = cv2.imencode('.jpg', combined)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        if cap:
            cap.release()
        hands.close()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Debug: Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)
