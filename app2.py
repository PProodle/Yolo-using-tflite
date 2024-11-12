# import cv2
# import numpy as np
# import tensorflow as tf

# # Path to the TensorFlow Lite model
# model_path = r'C:\Users\prana\.cache\kagglehub\models\kaggle\yolo-v5\tfLite\tflite-tflite-model\1\1.tflite'

# # Load the TFLite model
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# # Get input and output tensor details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the webcam
#     ret, frame = cap.read()
    
#     # If no frame is captured, break
#     if not ret:
#         print("Failed to grab frame")
#         break
    
#     # Pre-process the frame: Resize, convert to RGB and normalize
#     input_shape = input_details[0]['shape']
#     image_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
#     input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)
    
#     # Run the interpreter on the input data
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     # Get the output data
#     output_data = interpreter.get_tensor(output_details[0]['index'])
    
#     # Debug: print the shape of the output data
#     print("Output data shape:", output_data.shape)

#     # Assuming output_data is 3D and contains detection probabilities
#     # Modify the following part according to the model's output structure

#     # Example: If output_data has shape (1, num_detections, 4) and contains the probability
#     if len(output_data.shape) == 3:
#         # You may need to adjust this based on your model's output format
#         for detection in output_data[0]:
#             # Assuming detection contains a confidence score in the first element
#             confidence = detection[0]
#             if confidence > 0.5:  # 0.5 is the confidence threshold
#                 print("Person detected with confidence:", confidence)
#             else:
#                 print("No person detected")
#     else:
#         print("Unexpected output format. Please check the model's output.")

#     # Display the webcam frame
#     cv2.imshow('Webcam Feed', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf
import time

# Path to the TensorFlow Lite model
model_path = r'C:\Users\prana\.cache\kagglehub\models\kaggle\yolo-v5\tfLite\tflite-tflite-model\1\1.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variable to track time
last_check_time = time.time()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If no frame is captured, break
    if not ret:
        print("Failed to grab frame")
        break

    # Get current time
    current_time = time.time()

    # Check if 5 seconds have passed
    if current_time - last_check_time >= 5:
        last_check_time = current_time  # Reset the timer

        # Pre-process the frame: Resize, convert to RGB, and normalize
        input_shape = input_details[0]['shape']
        image_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(image_resized, axis=0).astype(np.float32)

        # Run the interpreter on the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the output data
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Debug: print the shape of the output data
        print("Output data shape:", output_data.shape)

        # Assuming output_data is 3D and contains detection probabilities
        if len(output_data.shape) == 3:
            detected = False
            for detection in output_data[0]:
                # Assuming detection contains a confidence score in the first element
                confidence = detection[0]
                if confidence > 0.3:  # 0.5 is the confidence threshold
                    print("Person detected with confidence:", confidence)
                    detected = True
                    break  # Exit loop once a person is detected
            if not detected:
                print("No person detected")
        else:
            print("Unexpected output format. Please check the model's output.")

    # Display the webcam frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
