import cv2
import numpy as np

def change_background_traditional(image_path, background_path, lower_bound, upper_bound):
    # Load input image and background image
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    # Check if the images are successfully loaded
    if image is None:
        raise FileNotFoundError(f"Error: Gambar tidak ditemukan di jalur {image_path}")
    if background is None:
        raise FileNotFoundError(f"Error: Gambar latar belakang tidak ditemukan di jalur {background_path}")
    
    # Resize background to match input image size, or vice versa
    image_height, image_width = image.shape[:2]
    background_height, background_width = background.shape[:2]
    
    # Resize background to match input image size
    if (image_height != background_height) or (image_width != background_width):
        background = cv2.resize(background, (image_width, image_height))

    # Convert input image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask using color thresholding for specified color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Invert mask to get the foreground
    mask_inv = cv2.bitwise_not(mask)
    
    # Extract the foreground from the input image
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    
    # Extract the background where the mask is true
    background_part = cv2.bitwise_and(background, background, mask=mask)
    
    # Combine the foreground and the new background
    result = cv2.add(foreground, background_part)
    
    return result

# Define HSV color ranges for common colors (green, blue, red)
color_ranges = {
    'green': (np.array([40, 40, 40]), np.array([80, 255, 255])),
    'blue': (np.array([100, 50, 50]), np.array([140, 255, 255])),
    'red': ([np.array([0, 50, 50]), np.array([170, 50, 50])], [np.array([10, 255, 255]), np.array([180, 255, 255])])
}

# Example user input to choose color to remove (green, blue, red)
chosen_color = 'red'  # You can change this based on user input

# Call the function with chosen color range
if chosen_color == 'red':
    result_image = change_background_traditional('image/CRE_5791 Merah.jpg', '/mnt/data/1.jpg', color_ranges[chosen_color][0], color_ranges[chosen_color][1])
else:
    result_image = change_background_traditional('/mnt/data/CRE_5791 Merah.jpg', '/mnt/data/1.jpg', color_ranges[chosen_color][0], color_ranges[chosen_color][1])

# Save the resulting image
cv2.imwrite('CRE_5791 Merah.jpg', result_image)
