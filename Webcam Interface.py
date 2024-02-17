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

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
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
    cv2.imshow('frame', frame)
    
    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
