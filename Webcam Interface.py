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

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match your model's expected input size
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))

    # Convert frame to the color space expected by the model if necessary
    if input_shape[3] == 1:  # Model expects grayscale images
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        resized_frame = resized_frame[..., np.newaxis]  # Add channel dimension

    # Normalize the image data to the 0-1 range if your model expects that
    img_array = resized_frame.astype(np.float32) / 255.0

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Verify img_array shape and type
    print(f"Input shape: {img_array.shape}, Expected shape: {input_shape}, Data type: {img_array.dtype}")

    # Ensure the data type matches the model's expected input type
    if img_array.dtype != input_dtype:
        img_array = img_array.astype(input_dtype)
    
    # Set the model input
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run the inference
    interpreter.invoke()

    # Retrieve the model output
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Process the output
    emotion = class_names[np.argmax(predictions)]
    
    # Display the resulting frame with the predicted emotion
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    
    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()