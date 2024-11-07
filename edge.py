import cv2
import numpy as np

def edge_detection_dnn(image_path, output_path):
    # Load the pre-trained model for HED (Holistically-Nested Edge Detection)
    model_path = "static/models/hed_pretrained_bsds.caffemodel"
    config_path = "static/models/deploy1.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Prepare the image for the network
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)

    # Pass the blob through the network and get the edge map
    net.setInput(blob)
    edge_map = net.forward()

    # Reshape the edge map to the original image size
    edge_map = edge_map[0, 0]
    edge_map = cv2.resize(edge_map, (w, h))

    # Normalize the edge map for proper visualization
    edge_map = (255 * edge_map).astype("uint8")

    # Save the edge map to the output path
    cv2.imwrite(output_path, edge_map)
    print(f"Edge detection image saved as {output_path}")

# Example usage (comment out when not running):
edge_detection_dnn("static/uploads/input.png", "output_edges.jpg")
