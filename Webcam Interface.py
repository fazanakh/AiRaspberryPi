import cv2
import numpy as np
import tensorflow as tf
import time

# Function to switch webcam
def switch_webcam(index, cap):
    if cap is not None:
        cap.release()
    return cv2.VideoCapture(index)

current_cam_index = 0
cap = cv2.VideoCapture(current_cam_index)  # Initial webcam

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

last_prediction_time = time.time()
prediction_interval = 0.5  # Predict every 0.5 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)

    current_time = time.time()
    if current_time - last_prediction_time >= prediction_interval and len(faces) > 0:
        # Predict only for the first detected face for simplicity
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (input_shape[1], input_shape[2]))
        img_array = np.expand_dims(resized_face.astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        emotion_probabilities = tf.nn.softmax(predictions).numpy()  # Apply softmax to get probabilities

        last_prediction_time = current_time

    # Draw the box and show the emotion text constantly for each detected face
    for (x, y, w, h) in faces:
        if emotion_probabilities is not None:
            most_probable_emotion = np.argmax(emotion_probabilities)
            box_color = colors[class_names[most_probable_emotion]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)

            # Adjust text position to ensure it stays on the screen
            text_position = max(y - 10, 20)  # Keep text above the box or at least 20 pixels from the top
            for i, probability in enumerate(emotion_probabilities):
                text = f"{class_names[i]}: {probability*100:.2f}%"
                cv2.putText(frame, text, (x, text_position - (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        else:
            # Draw a default box if no predictions have been made yet
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

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