import cv2
import numpy as np

def change_background_grabcut(image_path, background_path, rect):
    # Load input image and background image
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    # Check if the images are successfully loaded
    if image is None:
        raise FileNotFoundError(f"Error: Gambar tidak ditemukan di jalur {image_path}")
    if background is None:
        raise FileNotFoundError(f"Error: Gambar latar belakang tidak ditemukan di jalur {background_path}")

    # Resize background to match input image size
    image_height, image_width = image.shape[:2]
    background = cv2.resize(background, (image_width, image_height))

    # Initialize mask, bgdModel, and fgdModel for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify mask to extract the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = image * mask2[:, :, np.newaxis]

    # Extract the background where the mask is 0
    background_part = background * (1 - mask2[:, :, np.newaxis])

    # Combine the foreground and new background
    result = cv2.add(foreground, background_part)

    return result

# Example usage
image_path = 'image/2.jpg'
background_path = 'image/4.jpg'

# Define a rectangle around the foreground object (user-defined)
# (x, y, width, height) - should surround the object
rect = (50, 50, 450, 290)

# Call the function
result_image = change_background_grabcut(image_path, background_path, rect)

# Save the resulting image
cv2.imwrite('output_image_grabcut.jpg', result_image)
