import cv2

import numpy as np


# Load pre-trained model and configuration
model_path = "static/models/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "static/models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

def blur_faces_dnn(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    # Pass the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.5:
            # Compute the coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the dimensions of the image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Extract the face region of interest (ROI)
            face = image[startY:endY, startX:endX]

            # Apply Gaussian blur to the face
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)

            # Replace the original face with the blurred face in the image
            image[startY:endY, startX:endX] = blurred_face

    # Save the output image with blurred faces
    cv2.imwrite(output_path, image)
    print(f"Blurred face image saved as {output_path}")

# Example usage (comment out when not running):
blur_faces_dnn("static/uploads/input.png", "output_image.jpg")
