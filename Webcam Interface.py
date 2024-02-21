import cv2
import numpy as np
import tensorflow as tf
import time

# Function to switch webcam
def switch_webcam(index, cap):
    if cap is not None:
        cap.release()
    return cv2.VideoCapture(index)

# Load the TensorFlow Lite model and allocate tensors
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (255, 0, 255),
    'happy': (0, 255, 255),
    'neutral': (255, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 255, 0)
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
prediction_interval = 0.5  # Predict every 0.5 seconds
last_emotion = None  # Track the last detected emotion

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    current_time = time.time()
    if current_time - last_prediction_time >= prediction_interval:
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (input_shape[1], input_shape[2]))
            img_array = np.expand_dims(resized_face.astype(np.float32) / 255.0, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            emotion = class_names[np.argmax(predictions)]
            box_color = colors[emotion]

            # Update the emotion and box color only if the emotion changes
            if emotion != last_emotion:
                last_emotion = emotion
                last_prediction_time = current_time

        # Draw the box and show the emotion text constantly
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            if last_emotion is not None:
                cv2.putText(frame, last_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    cv2.imshow('Emotion Recognition', frame)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break
    elif key & 0xFF == ord('n'):
        cap = switch_webcam(current_cam_index + 1, cap)
    elif key & 0xFF == ord('p'):
        cap = switch_webcam(max(0, current_cam_index - 1), cap)

cap.release()
cv2.destroyAllWindows()
