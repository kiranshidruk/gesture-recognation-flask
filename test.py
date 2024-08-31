from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
from collections import deque, Counter
import base64
from utils import CvFpsCalc
import tensorflow as tf
from model import KeyPointClassifier, PointHistoryClassifier
from tensorflow.keras.models import load_model


app = Flask(__name__)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer models
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# keypoint_model = tf.keras.models.load_model('model/keypoint_classifier/keypoint_classifier.hdf5')
# point_history_model = tf.keras.models.load_model('model/point_history_classifier/point_history_classifier.hdf5')

# model = load_model('model/keypoint_classifier/keypoint_classifier.tflite')

interpreter = tf.lite.Interpreter(model_path="model/keypoint_classifier/keypoint_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load class names
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = csv.reader(f)
    point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

# Global variables
camera_active = False
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)
cvFpsCalc = CvFpsCalc(buffer_len=10)

def process_frame(frame):
    global point_history, finger_gesture_history

    fps = cvFpsCalc.get()

    frame = cv2.flip(frame, 1)  # Mirror display
    debug_image = copy.deepcopy(frame)

    # Detection implementation
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == 2:  # Point gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            finger_gesture_id = 0
            point_history_len = len(pre_processed_point_history_list)
            if point_history_len == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()

            debug_image = draw_bounding_rect(True, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(
                debug_image,
                brect,
                handedness,
                keypoint_classifier_labels[hand_sign_id],
                point_history_classifier_labels[most_common_fg_id[0][0]],
            )
    else:
        point_history.append([0, 0])

    debug_image = draw_point_history(debug_image, point_history)
    debug_image = draw_info(debug_image, fps, 0, -1)

    return debug_image

def generate_frames():
    global camera_active
    cap = cv2.VideoCapture(0)
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    global camera_active
    if camera_active:
        return jsonify({'status': 'Camera already active'})
    camera_active = True
    return jsonify({'status': 'Camera started'})

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    if not camera_active:
        return jsonify({'status': 'Camera already stopped'})
    camera_active = False
    return jsonify({'status': 'Camera stopped'})

# @app.route('/api/infer_keypoint', methods=['POST'])
# def infer_keypoint():
#     try:
#         data = request.json
#         keypoints = np.array(data['keypoints']).reshape(1, -1)  # Ensure the data is in the correct shape
#         prediction = keypoint_model.predict(keypoints)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         return jsonify({'gesture': int(predicted_class)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/api/infer_point_history', methods=['POST'])
# def infer_point_history():
#     try:
#         data = request.json
#         point_history = np.array(data['point_history']).reshape(1, -1)
#         prediction = point_history_model.predict(point_history)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         return jsonify({'gesture': int(predicted_class)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# Include other necessary functions from script 1
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    return temp_point_history

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 4)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 4)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 4)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 4)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 4)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 4)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 8)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 4)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # Wrist
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 2)
        if index == 1:  # Palm
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 2)
        if index in [2, 5, 9, 13, 17]:  # Finger bases
            cv2.circle(image, (landmark[0], landmark[1]), 12, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 12, (0, 0, 0), 2)
        if index in [4, 8, 12, 16, 20]:  # Finger tips
            cv2.circle(image, (landmark[0], landmark[1]), 12, (255, 255, 255), -1)
            cv2.circle(image, (landmark[0], landmark[1]), 12, (0, 0, 0), 2)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle with thicker border
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 4)
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (255, 255, 255), 2)
    return image



def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if finger_gesture_text != "":
        cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

if __name__ == "__main__":
    app.run(debug=True)
