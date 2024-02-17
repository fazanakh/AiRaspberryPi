import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# Define class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detector model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to switch webcam
def switch_webcam(index, cap):
    if cap is not None:
        cap.release()
    return cv2.VideoCapture(index)


# Initialize variables for the webcam and the current index
current_cam_index = 0
cap = cv2.VideoCapture(current_cam_index)  # Directly initialize the first webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract face ROI
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Resize the face ROI to match the model's expected input size
        resized_face = cv2.resize(face_roi, (input_shape[1], input_shape[2]))

        # If necessary, add channel dimension (if model expects grayscale, no need to add dimension)
        if input_shape[3] == 3:  # Model expects color images
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2BGR)

        # Normalize the image data and add a batch dimension
        img_array = resized_face.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Set the model input
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run the inference
        interpreter.invoke()

        # Retrieve the model output
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Process the output
        emotion = class_names[np.argmax(predictions)]
        
        # Display the predicted emotion next to the face bounding box
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
        # Quit the program
        break
    elif key & 0xFF == ord('n'):
        # Switch to the next webcam
        current_cam_index += 1
        cap = switch_webcam(current_cam_index, cap)
        if not cap.isOpened():
            print(f"No webcam found at index {current_cam_index}, resetting to default webcam.")
            current_cam_index = 0
            cap = switch_webcam(current_cam_index, cap)
    elif key & 0xFF == ord('p'):
        # Switch to the previous webcam
        current_cam_index = max(0, current_cam_index - 1)
        cap = switch_webcam(current_cam_index, cap)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
