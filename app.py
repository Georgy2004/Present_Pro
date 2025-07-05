from flask import Flask, request, render_template, redirect, url_for, jsonify, Response,send_from_directory
import os
import threading
import queue
from werkzeug.utils import secure_filename
from aspose.slides import Presentation
from aspose.pydrawing import Size
from aspose.pydrawing.imaging import ImageFormat
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import speech_recognition as sr
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/Presentation'
ALLOWED_EXTENSIONS = {'ppt', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER



# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)



current_slide = 0
slide_queue = queue.Queue()


# File validation and conversion functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_ppt_to_images(ppt_path, output_folder):
    pres = Presentation(ppt_path)
    os.makedirs(output_folder, exist_ok=True)
    for index, slide in enumerate(pres.slides):
        size = Size(960, 720)
        slide_path = os.path.join(output_folder, f"slide_{index + 1}.png")
        slide.get_thumbnail(size).save(slide_path, ImageFormat.png)


# Routes for uploading and converting presentations
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
            convert_ppt_to_images(filepath, STATIC_FOLDER)
            return redirect(url_for('presentation'))
    return render_template('upload.html')


# Gesture Control Presentation
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

    detectorHand = HandDetector(detectionCon=0.6, maxHands=1)
    imgNumber = 0
    pathImages = sorted(os.listdir(STATIC_FOLDER), key=len)
    if not pathImages:
        print("Error: No images found in the presentation folder.")
        return

    annotations = []
    annotation_temp = []
    annotation_color = (0, 0, 255)  # Default red
    last_command_time = 0
    command_interval = 1  # Minimum time between commands
    frame_counter = 0
    frame_skip = 2  # Process every second frame

    try:
        while True:
            img = None
            if cap:
                success, img = cap.read()
                if not success:
                    img = None
                else:
                    img = cv2.flip(img, 1)

            if frame_counter % frame_skip == 0:
                # Load slide
                pathFullImage = os.path.join(STATIC_FOLDER, pathImages[imgNumber])
                imgSlide = cv2.imread(pathFullImage)

                if imgSlide is None:
                    print(f"Error: Slide {imgNumber} could not be loaded.")
                    continue

                imgSlide = cv2.resize(imgSlide, (width, height))

                # Gesture detection
                if img is not None:
                    hands, img = detectorHand.findHands(img)
                    if hands:
                        hand = hands[0]
                        lmList = hand['lmList']
                        fingers = detectorHand.fingersUp(hand)  # Ensure fingers is initialized

                        cx, cy = lmList[8][0], lmList[8][1]  # Tip of index finger
                        current_time = time.time()

                        # Draw cursor at the tip of the index finger
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                        # Left swipe gesture
                        if cy <= gestureThreshold and fingers == [1, 0, 0, 0, 0]:
                            if imgNumber > 0 and current_time - last_command_time > command_interval:
                                imgNumber -= 1
                                annotations.clear()
                                annotation_temp.clear()
                                last_command_time = current_time
                            cv2.putText(img, "Swipe Left", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # Right swipe gesture
                        elif cy <= gestureThreshold and fingers == [0, 1, 1, 1, 1]:
                            if imgNumber < len(pathImages) - 1 and current_time - last_command_time > command_interval:
                                imgNumber += 1
                                annotations.clear()
                                annotation_temp.clear()
                                last_command_time = current_time
                            cv2.putText(img, "Swipe Right", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # Index Finger Up: Start or continue annotation
                        elif fingers == [0, 1, 0, 0, 0]:
                            annotation_temp.append((cx, cy))
                            cv2.putText(img, "Annotating", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # Index + Middle Finger Up: Draw red circle
                        elif fingers == [0, 1, 1, 0, 0]:
                            cv2.circle(imgSlide, (cx, cy), 20, annotation_color, cv2.FILLED)
                            cv2.putText(img, "Drawing Circle", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # Three Fingers Up: Delete last annotation
                        elif fingers == [0, 1, 1, 1, 0]:
                            if annotations:
                                annotations.pop(-1)
                            annotation_temp.clear()
                            cv2.putText(img, "Undo Annotation", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Store completed annotation
                if 'fingers' in locals() and (not fingers or fingers == [0, 0, 0, 0, 0]):
                    if annotation_temp:
                        annotations.append(annotation_temp.copy())
                        annotation_temp.clear()

                # Draw annotations on the slide
                for annotation in annotations:
                    for i in range(1, len(annotation)):
                        cv2.line(imgSlide, annotation[i - 1], annotation[i], annotation_color, 5)
                for i in range(1, len(annotation_temp)):
                    cv2.line(imgSlide, annotation_temp[i - 1], annotation_temp[i], annotation_color, 5)

            frame_counter += 1

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Voice Control Presentation
@app.route('/voice_presentation')
def voice_presentation():
    global current_slide
    slide_images = sorted(os.listdir(app.config['STATIC_FOLDER']))

    if slide_images:
        slides = [slide_images[current_slide]]
    else:
        slides = []

    return render_template('voice_presentation.html', slides=slides, current_slide=current_slide,
                           total_slides=len(slide_images))


@app.route('/next_slide', methods=['POST'])
def next_slide():
    """Move to the next slide."""
    global current_slide
    slide_images = sorted(os.listdir(app.config['STATIC_FOLDER']))

    if current_slide < len(slide_images) - 1:
        current_slide += 1

    return jsonify({"status": "success", "current_slide": current_slide})

@app.route('/previous_slide', methods=['POST'])
def previous_slide():
    """Move to the previous slide."""
    global current_slide
    if current_slide > 0:
        current_slide -= 1

    return jsonify({"status": "success", "current_slide": current_slide})

@app.route('/get_current_slide')
def get_current_slide():
    """Get the current slide index and total slides."""
    slide_images = sorted(os.listdir(app.config['STATIC_FOLDER']))
    return jsonify({
        "current_slide": current_slide,
        "total_slides": len(slide_images)
    })



# Voice recognition thread
def listen_for_voice_commands():
    """Continuously listen for voice commands in a non-blocking way."""
    r = sr.Recognizer()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for voice commands...")

        while True:
            try:
                # Listen for a command with a smaller timeout to avoid long delays
                audio = r.listen(source, timeout=5, phrase_time_limit=5)  # Adjust timeout as needed
                command = recognizer.recognize_google(audio)
                print(f"Recognized command: {command}")

                # Immediately process the command (without delay)
                slide_queue.put(command)

                # Handle voice command immediately
                process_voice_command(command)

            except sr.UnknownValueError:
                print("Could not understand the audio. Please try again.")
            except sr.RequestError as e:
                print(f"Speech Recognition API error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

            time.sleep(1.0)  # A short pause between listening attempts, can be adjusted


def process_voice_command(command):
    """Process the recognized voice command and update the slide index."""
    global current_slide
    command = command.lower()

    # Map number words to integers
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20
    }

    slide_images = sorted(os.listdir(app.config['STATIC_FOLDER']))

    if "next" in command:
        if current_slide < len(slide_images) - 1:
            current_slide += 1
            print(f"Moved to slide {current_slide + 1}")
    elif "previous" in command:
        if current_slide > 0:
            current_slide -= 1
            print(f"Moved to slide {current_slide + 1}")
    elif "go to" in command:
        # Command like "go to 6" or "go to six"
        try:
            words = command.split("go to")[1].strip().split()
            if words:
                number_str = words[0]  # Assume the first word is the number
                slide_number = parse_slide_number(number_str, number_words)

                if slide_number is not None and 1 <= slide_number <= len(slide_images):
                    current_slide = slide_number - 1
                    print(f"Moved to slide {current_slide + 1}")
                else:
                    print(f"Slide number '{number_str}' is out of range.")
            else:
                print("Could not parse the slide number.")
        except ValueError:
            print("Could not parse the slide number.")
    else:
        # Direct number command (like "6" or "six")
        slide_number = parse_slide_number(command, number_words)

        if slide_number is not None and 1 <= slide_number <= len(slide_images):
            current_slide = slide_number - 1
            print(f"Moved to slide {current_slide + 1}")
        else:
            print(f"Slide number '{command}' is out of range or unrecognized.")

def parse_slide_number(command, number_words):
    """Helper function to parse slide numbers from command."""
    # Check if the command is a number word or a numeric string
    if command.isdigit():
        return int(command)
    elif command in number_words:
        return number_words[command]
    else:
        return None


# Start voice recognition in the background
voice_thread = threading.Thread(target=listen_for_voice_commands, daemon=True)
voice_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
